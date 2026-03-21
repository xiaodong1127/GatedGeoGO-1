import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from gvp.go_embedding_utils import GOTreeEncoder
import torch.nn.functional as F
from torch_geometric.utils import softmax

# ------------------------ 工具函数 ------------------------
def transpose_qkv(X, num_heads):
    # X: (batch, seq_len, num_hiddens)
    batch, seq_len, hidden = X.shape
    assert hidden % num_heads == 0, "hidden must be divisible by num_heads"
    head_dim = hidden // num_heads
    X = X.reshape(batch, seq_len, num_heads, head_dim)  # (B, L, H, D_h)
    X = X.permute(0, 2, 1, 3)                           # (B, H, L, D_h)
    return X.reshape(-1, seq_len, head_dim)           # (B*H, L, D_h)

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)

def transpose_output(X, num_heads):
    # X: (B*H, L, D_h) -> (B, L, H*D_h)
    B_times_H, L, D_h = X.shape
    assert B_times_H % num_heads == 0
    B = B_times_H // num_heads
    X = X.reshape(B, num_heads, L, D_h)     # (B, H, L, D_h)
    X = X.permute(0, 2, 1, 3)               # (B, L, H, D_h)
    return X.reshape(B, L, num_heads * D_h) # (B, L, H*D_h)

def reshape_tensors(seq, seq_lens, pad_idx):
    """
    seq: flattened sequence items shaped (total_items, feat_dim)
    seq_lens: list/tuple/tensor of per-batch lengths
    returns: padded tensor (batch, max_len, feat_dim)
    """
    if len(seq_lens) != 1:
        batch_size = len(seq_lens)
        seq_max_len = max(seq_lens)
        total = 0
        shape = list(seq.shape)
        feat_dim = shape[1] if len(shape) > 1 else 1
        seqs = torch.full((batch_size, seq_max_len, feat_dim), pad_idx, dtype=seq.dtype, device=seq.device)
        for i, seq_len in enumerate(seq_lens):
            if seq_len == 0:
                continue
            seqs[i, :seq_len] = seq[total:total + seq_len]
            total += seq_len
    else:
        seqs = seq.unsqueeze(0)
    return seqs

# ------------------------ 模块定义 ------------------------
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # normalized_shape may be int or list; pass int
        if isinstance(normalized_shape, (list, tuple)):
            normalized_shape = normalized_shape[0]
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        # queries/keys/values: (B*H, L, D_h)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, queries, keys, values, valid_lens):
        # queries/keys/values: (B, L, dim)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)  # (B*H, L, D_h)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # ensure tensor on same device
            if not torch.is_tensor(valid_lens):
                valid_lens = torch.tensor(valid_lens, device=queries.device)
            valid_lens = valid_lens.to(dtype=torch.long, device=queries.device)
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)  # (B, L, num_hiddens)
        return self.W_o(output_concat)

class FunctionPredictor(nn.Module):
    def __init__(self, input_size, output_size, drop_rate=0.1):
        super().__init__()
        hidden_size = 4 * output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    def forward(self, X, valid_lens):
        # X: (B, L, dim)
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class GatedPPISelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, ppi_tensor, num_ppi):
        """
        ppi_tensor: (batch, seq_len, dim)
        num_ppi: list/tensor of valid lengths per batch
        returns gated_tensor same shape, gate_scores (batch, seq_len)
        """
        gate_scores = self.gate_mlp(ppi_tensor)  # (B, L, 1)
        gated_tensor = gate_scores * ppi_tensor
        return gated_tensor, gate_scores.squeeze(-1)

# ---------------- graph/geometry helpers ----------------
def make_block_diagonal_knn_edges(pos, num_nodes_list, k=8, device=None):
    """
    pos: (total_N, 3)
    num_nodes_list: list of ints
    returns edge_index (2, total_edges) and a list of per-edge (src_idx_within_graph, dst_idx_within_graph, graph_id)
    We construct k-NN edges per node within its graph (directed from neighbor -> node for message passing).
    """
    device = device if device is not None else pos.device
    rows = []
    cols = []
    offset = 0
    for graph_id, n in enumerate(num_nodes_list):
        if n == 0:
            continue
        p = pos[offset:offset + n]  # (n, 3)
        # pairwise distances
        # torch.cdist yields (n, n)
        dist = torch.cdist(p, p, p=2)  # (n, n)
        # set self-dist large so topk excludes self if k < n
        diag_inf = torch.eye(n, device=device) * 1e6
        dist = dist + diag_inf
        kk = min(k, n - 1) if n > 1 else 0
        if kk > 0:
            # for each node (dst), find kk nearest neighbors (src)
            knn_dist, knn_idx = torch.topk(dist, kk, largest=False, dim=-1)  # (n, kk)
            # edges: src = knn_idx[:, j], dst = node index
            dst_idx = torch.arange(n, device=device).unsqueeze(1).repeat(1, kk)  # (n, kk)
            rows.append((knn_idx + offset).reshape(-1))
            cols.append((dst_idx + offset).reshape(-1))
        else:
            # single node: no neighbors -> no edges
            pass
        offset += n
    if len(rows) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)
    return torch.stack([row, col], dim=0)  # shape (2, E)

def per_graph_rel_pos(edge_index, num_nodes_list):
    """
    compute rel_pos index per edge (clamped within [0,511]) based on positions inside each graph.
    Assumes edge_index is block-diagonal by graph in order.
    Returns tensor of rel_pos length E.
    """
    rels = []
    offset = 0
    e_idx = 0
    for n in num_nodes_list:
        if n == 0:
            continue
        # collect edges within this block
        # find edges where both src and dst in [offset, offset+n)
        # caller might already provide block-wise edges; for efficiency assume edges follow blocks in same order.
        # We'll compute per-edge relative indices by mapping global index -> local index via modulo
        e_idx += 0
        offset += n
    # Simpler: compute rel as (col - row).clamp(0,511)
    rel = (edge_index[1] - edge_index[0]).clamp(min=0, max=511)
    return rel.long()

def compute_edge_attr_from_pos(pos, edge_index, rbf_proj, pos_proj, rbf_gamma=10.0):
    """
    pos: (total_N, 3)
    edge_index: (2, E) global indices
    returns edge_attr: (E, rbf_k + pos_emb_dim + 3)
    """
    src = edge_index[0]
    dst = edge_index[1]
    vec = pos[src] - pos[dst]  # (E, 3)  note: consistent with original vec = pos[edge_index[0]] - pos[edge_index[1]]
    dist = torch.norm(vec, dim=-1, keepdim=True)  # (E,1)
    direction = F.normalize(vec, dim=-1)  # (E,3)
    # rbf: apply linear projection on dist then gaussian basis
    rbf_raw = rbf_proj(dist)  # (E, rbf_k)
    rbf_feat = torch.exp(-rbf_raw ** 2)  # same idea as original
    # positional embedding based on rel index
    rel_pos = (dst - src).clamp(min=0, max=511)
    pos_feat = pos_proj(rel_pos)  # (E, pos_emb_dim)
    edge_attr = torch.cat([rbf_feat, pos_feat, direction], dim=-1)
    return edge_attr

# ---------------- MessagePassing block ----------------
class FANHybridBlock(MessagePassing):
    def __init__(self, in_dim, hidden_dim, edge_attr_type='rbf_pos', rbf_k=16, pos_emb_dim=16, k_nn=8):
        super().__init__(aggr='add')
        self.edge_attr_type = edge_attr_type
        self.rbf_k = rbf_k
        self.pos_emb_dim = pos_emb_dim
        self.k_nn = k_nn
        if edge_attr_type == 'rbf_pos':
            edge_in_dim = rbf_k + pos_emb_dim + 3
            # rbf_proj maps scalar distance -> rbf_k dims
            self.rbf_proj = nn.Linear(1, rbf_k, bias=False)
            self.pos_proj = nn.Embedding(512, pos_emb_dim)
        else:
            raise ValueError(f"Unknown edge_attr_type: {edge_attr_type}")
        self.frame_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.x_proj = nn.Linear(in_dim, hidden_dim)
        self.update_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.attn_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, x, pos, edge_index=None, num_aa=None):
        """
        x: (total_N, in_dim)
        pos: (total_N, 3)
        num_aa: list of ints describing nodes per graph in order
        edge_index: optional precomputed (2, E)
        Returns: updated node embeddings (total_N, hidden_dim)
        """
        device = x.device
        if edge_index is None:
            if num_aa is None:
                # fallback: fully connect global graph (not recommended)
                N = pos.size(0)
                row, col = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')
                edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
            else:
                edge_index = make_block_diagonal_knn_edges(pos, num_aa, k=self.k_nn, device=device)

        # compute edge_attr per-edge
        edge_attr = compute_edge_attr_from_pos(pos, edge_index, self.rbf_proj, self.pos_proj)

        # propagate using torch_geometric MessagePassing
        # Note: message signature consumes x_j (source node features), edge_attr, edge_index, x_i (dest features)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr, edge_index, x_i):
        # x_j: (E, in_proj_dim) projected later
        f_frame = self.frame_mlp(edge_attr)    # (E, hidden_dim)
        x_j_proj = self.x_proj(x_j)            # (E, hidden_dim)
        attn_score = (x_j_proj * f_frame).sum(dim=-1)  # (E,)
        # edge_index[1] are destination node indices for each edge
        attn_weight = softmax(attn_score, edge_index[1])  # softmax over incoming edges per destination node
        return attn_weight.unsqueeze(-1) * (x_j_proj + f_frame)

    def update(self, aggr_out, x):
        # aggr_out: (total_N, hidden_dim)
        x_proj = self.x_proj(x)  # (total_N, hidden_dim)
        gate_input = torch.cat([aggr_out, x_proj], dim=-1)
        gate = self.attn_gate(gate_input)  # (total_N, 1)
        updated = self.update_mlp(gate * aggr_out + (1 - gate) * x_proj)
        return updated

# -------------------- HybridFANNet --------------------
class HybridFANNet(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, edge_attr_type='rbf_pos', rbf_k=16, pos_emb_dim=16, k_nn=8):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            FANHybridBlock(hidden_dim, hidden_dim, edge_attr_type=edge_attr_type, rbf_k=rbf_k, pos_emb_dim=pos_emb_dim, k_nn=k_nn)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        # readout query small init to avoid collapse
        self.readout_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.readout_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

    def forward(self, x, pos, edge_index, num_aa):
        x = self.input_proj(x)
        for block, norm in zip(self.layers, self.norms):
            residual = x
            x = block(x, pos, edge_index=edge_index, num_aa=num_aa)
            x = norm(x + residual)
        # readout per graph
        graph_features = []
        start = 0
        for n in num_aa:
            end = start + n
            if n == 0:
                graph_features.append(torch.zeros(self.readout_query.shape[-1], device=x.device).unsqueeze(0))
            else:
                node_feats = x[start:end].unsqueeze(0)  # (1, n, hidden)
                query = self.readout_query.unsqueeze(0)  # (1, 1, hidden)
                pooled, _ = self.readout_attn(query, node_feats, node_feats)
                graph_features.append(pooled.squeeze(0))
            start = end
        return torch.cat(graph_features, dim=0)  # (batch, hidden_dim)


# ---------------- node feature generation ----------------
def generate_node_features_from_ca(ca_coords, num_aa, k=4, rbf_k=16, rbf_gamma=10.0):
    """
    ca_coords: (total_N, 3)
    num_aa: list of ints stating node counts per graph (length batch)
    returns: (total_N, feat_dim) where feat_dim = 3 (mean) + 3 (std) + rbf_k
    NOTE: we compute neighbor mean/std based on k-NN within each graph.
    """
    device = ca_coords.device
    out_list = []
    start = 0
    mu = torch.linspace(0, 20, rbf_k, device=device)
    for n in num_aa:
        if n == 0:
            start += 0
            continue
        coords = ca_coords[start:start + n]  # (n, 3)
        if n == 1:
            # no neighbors: zeros
            neighbor_mean = torch.zeros((1, 3), device=device)
            neighbor_std = torch.zeros((1, 3), device=device)
            rbf_mean = torch.zeros((1, rbf_k), device=device)
            out_list.append(torch.cat([neighbor_mean, neighbor_std, rbf_mean], dim=-1))
            start += n
            continue
        # pairwise
        dist = torch.cdist(coords, coords, p=2)  # (n, n)
        diag_inf = torch.eye(n, device=device) * 1e6
        dist = dist + diag_inf
        kk = min(k, n - 1)
        knn_dist, knn_idx = torch.topk(dist, kk, largest=False, dim=-1)  # (n, kk)
        # neighbor vectors: for each node i, gather coords[j] where j in knn_idx[i]
        # we can use gather
        idx = knn_idx.unsqueeze(-1).expand(-1, -1, 3)  # (n, kk, 3)
        neighbor_vecs = torch.gather(coords.unsqueeze(0).expand(n, -1, -1), 1, idx)  # (n, kk, 3)
        neighbor_mean = neighbor_vecs.mean(dim=1)  # (n, 3)
        neighbor_std = neighbor_vecs.std(dim=1)    # (n, 3)
        # RBF on distances: use knn_dist (n, kk) -> compute rbf basis per neighbor then average across neighbors
        rbf = torch.exp(-rbf_gamma * (knn_dist.unsqueeze(-1) - mu) ** 2)  # (n, kk, rbf_k)
        rbf_mean = rbf.mean(dim=1)  # (n, rbf_k)
        out = torch.cat([neighbor_mean, neighbor_std, rbf_mean], dim=-1)  # (n, 3+3+rbf_k)
        out_list.append(out)
        start += n
    if len(out_list) == 0:
        return torch.empty((0, 6 + rbf_k), device=device)
    return torch.cat(out_list, dim=0)


class PredGONet(nn.Module):
    def __init__(self, num_class, aa_node_in_dim, aa_edge_index, ppn_num_heads, num_ppn_layers,
                 num_layers, hidden_dim=512, drop_rate=0.2, edge_attr_type='rbf_pos', k_nn=8, esm_dim=1280):
        super().__init__()
        self.aa_node_in_dim = aa_node_in_dim
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        self.hybrid_fan_net = HybridFANNet(aa_node_in_dim, hidden_dim, num_layers, edge_attr_type=edge_attr_type, k_nn=k_nn)
        self.gated_selector = GatedPPISelector(input_dim=self.esm_dim)
        self.addnorm = AddNorm(normalized_shape=self.esm_dim, dropout=drop_rate)
        self.ppn_net = nn.ModuleList([
            EncoderBlock(key_size=self.esm_dim, query_size=self.esm_dim, value_size=self.esm_dim,
                         num_hiddens=self.esm_dim, norm_shape=self.esm_dim,
                         ffn_num_input=self.esm_dim, ffn_num_hiddens=4*self.esm_dim,
                         num_heads=ppn_num_heads, dropout=drop_rate)
            for _ in range(num_ppn_layers)
        ])
        self.output_layer = FunctionPredictor(self.esm_dim + hidden_dim, num_class, drop_rate=drop_rate)
        self.mse_loss = nn.MSELoss()

        # -------------------- struct2seq MLP --------------------
        self.struct2seq = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.esm_dim)
        )
        # 初始化每个 Linear 层
        for m in self.struct2seq:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.aa_edge_index = aa_edge_index

    def forward(self, sequence_embedding, aa_ca_coords, num_aa, ppi_tensor, num_ppi, aa_edge_index=None, debug=False):
        # ------------------ 结构分支 ------------------
        aa_h_V = generate_node_features_from_ca(aa_ca_coords, num_aa, k=4, rbf_k=16)
        aa_edge_index_used = aa_edge_index if aa_edge_index is not None else self.aa_edge_index
        struct_feat = self.hybrid_fan_net(aa_h_V, aa_ca_coords, edge_index=aa_edge_index_used, num_aa=num_aa)

        # ------------------ 序列+PPI分支 ------------------
        ppi_tensor = reshape_tensors(ppi_tensor, num_ppi, 0)
        if ppi_tensor.size(-1) != self.esm_dim:
            proj = nn.Linear(ppi_tensor.size(-1), self.esm_dim).to(ppi_tensor.device)
            ppi_tensor = proj(ppi_tensor)
        ppi_tensor, gate_scores = self.gated_selector(ppi_tensor, num_ppi)

        if sequence_embedding.dim() == 1:
            sequence_embedding = sequence_embedding.unsqueeze(0)
        if sequence_embedding.dim() == 2 and sequence_embedding.shape[1] != self.esm_dim:
            total = sequence_embedding.numel()
            if total % self.esm_dim != 0:
                raise ValueError(f"sequence_embedding cannot reshape to (-1, {self.esm_dim})")
            sequence_embedding = sequence_embedding.reshape(-1, self.esm_dim)
        elif sequence_embedding.dim() == 3:
            sequence_embedding = sequence_embedding.mean(dim=1)

        if not torch.is_tensor(num_ppi):
            valid_lens = torch.tensor(num_ppi, dtype=torch.long, device=ppi_tensor.device)
        else:
            valid_lens = num_ppi.to(device=ppi_tensor.device, dtype=torch.long)

        for layer in self.ppn_net:
            ppi_tensor = layer(ppi_tensor, valid_lens)
        ppi_tensor = ppi_tensor[:, 0, :]
        seq_feat = self.addnorm(sequence_embedding, ppi_tensor)

        # ------------------ 结构 -> 序列 对齐 loss (cl_loss) ------------------
        min_b = min(struct_feat.shape[0], seq_feat.shape[0])
        if min_b == 0:
            cl_loss = torch.tensor(0.0, device=struct_feat.device if struct_feat.numel() > 0 else seq_feat.device,
                                   requires_grad=True)
        else:
            struct_feat_trim = struct_feat[:min_b]
            seq_feat_trim = seq_feat[:min_b]
            seq_target = seq_feat_trim.detach()  # 不更新 seq 分支

            struct_mapped = self.struct2seq(struct_feat_trim)

            struct_norm = F.normalize(struct_mapped, dim=1)
            seq_norm = F.normalize(seq_target, dim=1)

            cl_loss = 1 - (struct_norm * seq_norm).sum(dim=1).mean()

            if debug and min_b > 0:
                print("DEBUG shapes: struct_mapped", struct_mapped.shape, " seq_target", seq_target.shape)
                mse_per_sample = torch.mean((struct_norm - seq_norm) ** 2, dim=1)
                print("DEBUG mse_per_sample: mean {:.6f}, std {:.6f}".format(mse_per_sample.mean().item(),
                                                                            mse_per_sample.std().item()))
                cos_sim = (struct_norm * seq_norm).sum(dim=1)
                print("DEBUG cos_sim mean:", cos_sim.mean().item())

        # ------------------ 最终预测 ------------------
        if struct_feat.shape[0] != seq_feat.shape[0]:
            struct_feat = struct_feat[:min_b]
            seq_feat = seq_feat[:min_b]

        if seq_feat.numel() > 0 and struct_feat.numel() > 0:
            final_input = torch.cat([seq_feat, struct_feat], dim=1)
            output = self.output_layer(final_input)
        else:
            device = seq_feat.device if seq_feat.numel() > 0 else struct_feat.device if struct_feat.numel() > 0 else next(self.parameters()).device
            in_feat = self.output_layer.input_layer.in_features
            out_feat = self.output_layer.output_layer.out_features
            final_input = torch.empty((0, in_feat), device=device)
            output = torch.empty((0, out_feat), device=device)

        return output, final_input, struct_feat, seq_feat, cl_loss



