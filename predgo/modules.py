import math

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing
from gvp.go_embedding_utils import GOTreeEncoder
import torch.nn.functional as F
from torch_geometric.utils import softmax



def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

#其主要功能是对一个三维张量 X 执行带有掩码的 softmax 操作，其中 X 的最后一维被视为有效长度的序列。
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

#其目的是逆转前面 transpose_qkv 函数所做的操作，具体来说，是将多头注意力机制的输出从多头的格式转换回原始的格式
def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def mean_tensor(combined_tensor, lens):
    """
    tensor will be divided according to lens, and then merge with means
    :param combined_tensor:
    :param lens: tensor lens
    :return:
    """
    means = [combined_tensor[i, :l, :].mean(0) for i, l in enumerate(lens)]
    return torch.stack(means, dim=0)

def reshape_tensors(seq, seq_lens, pad_idx):
    """
    Combine tensors of different lengths into one tensor, where the excess is filled with pad_idx
    :param seq:
    :param seq_lens:
    :param pad_idx:
    :return:
    """
    if len(seq_lens) != 1:
        batch_size = len(seq_lens)
        seq_max_len = max(seq_lens)
        total = 0#用于跟踪当前正在处理的序列长度
        shape = list(seq.shape)
        shape.insert(0, batch_size)
        shape[1] = seq_max_len
        seqs = torch.full(shape, pad_idx, dtype=seq.dtype).to(seq.device)
        for i, seq_len in enumerate(seq_lens):
            seqs[i, :seq_len] = seq[total:total + seq_len]
            total += seq_len
    else:
        seqs = seq.unsqueeze(0)
    return seqs


class AddNorm(nn.Module):
    """Layer normalization after residual concatenation"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    """Location-based feedforward networks"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class DotProductAttention(nn.Module):
    """Scaled dot product attention.
        点积注意力机制
    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        #应用掩码并计算注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        #如果给定了 valid_lens，那么它会将每个长度重复 num_heads 次，以便与多个头的形状相匹配。
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class FunctionPredictor(torch.nn.Module):
    def __init__(self, input_size, output_size, drop_rate=0.1):
        super(FunctionPredictor, self).__init__()

        hidden_size = 4 * output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class EncoderBlock(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class GatedPPISelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(GatedPPISelector, self).__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 ∈ [0, 1]，表示每个ppi节点的重要性
        )

    def forward(self, ppi_tensor, num_ppi):
        gate_scores = self.gate_mlp(ppi_tensor)  # [B, N, 1]
        gated_tensor = gate_scores * ppi_tensor  # 权重缩放
        return gated_tensor, gate_scores.squeeze(-1)  # 返回缩放后的节点嵌入 & 重要性分数

class FANHybridBlock(MessagePassing):
    def __init__(self, in_dim, hidden_dim, edge_attr_type='rbf_pos', rbf_k=16, pos_emb_dim=16):
        super(FANHybridBlock, self).__init__(aggr='add')
        self.edge_attr_type = edge_attr_type
        self.rbf_k = rbf_k
        self.pos_emb_dim = pos_emb_dim

        if edge_attr_type == 'rbf_pos':
            edge_in_dim = rbf_k + pos_emb_dim + 3
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
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.attn_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, pos, edge_index):
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.norm(vec, dim=-1, keepdim=True)
        direction = F.normalize(vec, dim=-1)
        rbf_feat = torch.exp(-self.rbf_proj(dist) ** 2)
        rel_pos = (edge_index[1] - edge_index[0]).clamp(min=0, max=511)
        pos_feat = self.pos_proj(rel_pos)
        edge_attr = torch.cat([rbf_feat, pos_feat, direction], dim=-1)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr, edge_index, x_i):
        f_frame = self.frame_mlp(edge_attr)
        x_j_proj = self.x_proj(x_j)
        attn_score = (x_j_proj * f_frame).sum(dim=-1)  # [E]
        attn_weight = softmax(attn_score, edge_index[1])  #
        return attn_weight.unsqueeze(-1) * (x_j_proj + f_frame)

    def update(self, aggr_out, x):
        x_proj = self.x_proj(x)
        gate_input = torch.cat([aggr_out, x_proj], dim=-1)
        gate = self.attn_gate(gate_input)
        gated_out = gate * aggr_out + (1 - gate) * x_proj
        return self.update_mlp(gated_out)

class HybridFANNet(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_layers, edge_attr_type='rbf_pos', rbf_k=16, pos_emb_dim=16):
        super(HybridFANNet, self).__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            FANHybridBlock(hidden_dim, hidden_dim, edge_attr_type=edge_attr_type, rbf_k=rbf_k, pos_emb_dim=pos_emb_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.readout_query = nn.Parameter(torch.randn(1, hidden_dim))
        self.readout_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, x, pos, edge_index, num_aa):
        x = self.input_proj(x)
        for block, norm in zip(self.layers, self.norms):
            residual = x
            x = block(x, pos, edge_index)
            x = norm(x + residual)

        graph_features = []
        start = 0
        for n in num_aa:
            end = start + n
            node_feats = x[start:end].unsqueeze(0)  # [1, N, D]
            query = self.readout_query.unsqueeze(0).expand(1, 1, -1)  # [1, 1, D]
            pooled, _ = self.readout_attn(query, node_feats, node_feats)
            graph_features.append(pooled.squeeze(0))
            start = end
        return torch.cat(graph_features, dim=0)


class PredGONet(nn.Module):

    def __init__(self, num_class, aa_node_in_dim, aa_ca_coords,aa_edge_index,ppn_num_heads,num_ppn_layers,
                 num_layers,hidden_dim,egnn_out_dim,go_tree_embedding=None, drop_rate=0.2,edge_attr_type='rbf_pos'):
        super(PredGONet, self).__init__()
        self.hybrid_fan_net = HybridFANNet(aa_node_in_dim, hidden_dim, num_layers,edge_attr_type=edge_attr_type)
        self.go_tree_encoder = GOTreeEncoder()
        self.esm_dim = 1280
        self.go_dim = 512
        self.gated_selector = GatedPPISelector(input_dim=self.esm_dim)
        self.ppn_mlp = nn.ModuleList()
        self.addnorm = AddNorm(normalized_shape=[self.esm_dim], dropout=drop_rate)
        self.cross_attn_go_to_seq = nn.MultiheadAttention(
            embed_dim=self.go_dim,kdim=self.esm_dim,vdim=self.esm_dim,
            num_heads=16,dropout=drop_rate,batch_first=True
        )
        self.ppn_net = nn.ModuleList([EncoderBlock(key_size=self.esm_dim, query_size=self.esm_dim,
                                                   value_size=self.esm_dim, num_hiddens=self.esm_dim,
                                                   norm_shape=[self.esm_dim], ffn_num_input=self.esm_dim,
                                                   ffn_num_hiddens=4 * self.esm_dim, num_heads=ppn_num_heads,
                                                   dropout=drop_rate) for _ in range(num_ppn_layers)])
        self.output_layer = FunctionPredictor(self.esm_dim + hidden_dim + self.go_dim, num_class,
                                              drop_rate=drop_rate)

    @staticmethod
    def batch_tag_tensor(num_tensor):
        r = []
        for i, v in enumerate(num_tensor):
            r.extend([i] * v)
        return torch.tensor(r, device=num_tensor.device)

    def forward(self, sequence_embedding, aa_h_V, aa_ca_coords, num_aa, ppi_tensor, num_ppi, aa_edge_index):
        ppi_tensor = reshape_tensors(ppi_tensor, num_ppi, 0)
        ppi_tensor, gate_scores = self.gated_selector(ppi_tensor, num_ppi)
        sequence_embedding = sequence_embedding.view(sequence_embedding.shape[0] // self.esm_dim, self.esm_dim)
        for layer in self.ppn_net:
            ppi_tensor = layer(ppi_tensor, num_ppi)
        ppi_tensor = ppi_tensor[:, 0, :]
        sequence_embedding = self.addnorm(sequence_embedding, ppi_tensor)
        aa_h_V = self.hybrid_fan_net(aa_h_V, aa_ca_coords, aa_edge_index, num_aa)

        go_tree_embedding = self.go_tree_encoder()
        batch_size = sequence_embedding.size(0)
        go_tree_embedding = go_tree_embedding.expand(batch_size, -1)
        seq_input = sequence_embedding.unsqueeze(1)
        go_input = go_tree_embedding.unsqueeze(1)
        go_attn_out, _ = self.cross_attn_go_to_seq(go_input, seq_input, seq_input)
        go_attn_out = go_attn_out.squeeze(1)

        final_input = torch.cat([sequence_embedding, aa_h_V, go_attn_out], dim=1)
        output = self.output_layer(final_input)
        return output, final_input


