# predgo/model1.py (MSE + cl_loss 版本，cl_loss单独反向传播，两个 optimizers)
import glob
import os
import obonet
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

from esm.extract import extract_esm_features_in_memory
from predgo.data import generate_PredGOData
from predgo.modules1 import PredGONet
from tools.log import log, logging_params
from tools.metrics import fmax_pytorch, pair_aupr, auc_pytorch, SminCalculatorPytorch
from tools.structure_data_parser import StructureDataParser

# ======================== 工具函数 ========================
def move_to_device(obj, device):
    """递归地把 batch 中的 tensor 移到 device。"""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    return obj

# ======================== BaseModel ========================
class BaseModel(object):
    """Base class for all models"""
    def __init__(self):
        self.smin_calculator = None
        self.params = None
        self.loss = nn.MSELoss()  # task_loss
        self.optimizer_task = None
        self.optimizer_cl = None
        self.best_fmax = -1
        self.best_smin = -1
        self.best_aupr = -1
        self.best_threshold = -1
        self.net = None
        log.do_print(f'Model: {self.__class__.__name__}')

    @logging_params
    def init_model(self, **kwargs):
        raise NotImplementedError

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    @torch.no_grad()
    def predict_y(self, batch, device):
        y_hat, _ = self.get_y_score(batch, device)
        return torch.sigmoid(y_hat)

    def get_y_true(self, batch, device):
        raise NotImplementedError

    def get_y_score(self, batch, device):
        raise NotImplementedError

    # ----------------------- helper: setup optimizers -----------------------
    def _setup_optimizers(self, lr_task, lr_cl, cl_param_selector=None):
        """
        创建两个 optimizer：
          - optimizer_task: 更新除 cl_params 之外的所有参数
          - optimizer_cl: 更新 cl_params（例如 struct2seq），如果没有 cl_params，则为 None

        cl_param_selector: 可选的字符串列表或单个字符串，表示参数名中包含这些子串的参数将被分配到 cl optimizer。
                           如果为 None，则会尝试寻找 self.net.struct2seq（module）并将其参数分配给 cl optimizer。
        """
        # 处理 DataParallel
        net_core = self.net.module if isinstance(self.net, nn.DataParallel) else self.net

        # 收集 cl 参数
        cl_params = []
        if cl_param_selector is None:
            # 约定：如果 net 有 struct2seq 属性，把它交给 cl optimizer
            if hasattr(net_core, 'struct2seq'):
                cl_params = list(net_core.struct2seq.parameters())
        else:
            selectors = [cl_param_selector] if isinstance(cl_param_selector, str) else list(cl_param_selector)
            for name, p in net_core.named_parameters():
                for sel in selectors:
                    if sel in name:
                        cl_params.append(p)
                        break

        # 去重
        cl_param_ids = {id(p) for p in cl_params}
        task_params = [p for p in net_core.parameters() if id(p) not in cl_param_ids]

        # 创建 optimizers
        self.optimizer_task = torch.optim.AdamW(task_params, lr=lr_task)
        if len(cl_params) > 0:
            self.optimizer_cl = torch.optim.AdamW(cl_params, lr=lr_cl)
        else:
            self.optimizer_cl = None

    # ======================== 训练 ========================
    def train_step(self, train_iter, device, clip_grad_norm=None, debug_batches=1):
        """
        单 epoch 的训练：对每个 batch
         - 先做 task_loss 的 forward/backward/step（使用 optimizer_task）
         - 再重新 forward 得到 cl_loss（新的计算图），对 cl_loss 做 backward/step（使用 optimizer_cl）
        这样 cl_loss 的训练是完全独立的，不会影响 task_loss 的合并。
        """
        if self.net is None:
            raise RuntimeError("Network not initialized.")

        if self.optimizer_task is None:
            raise RuntimeError("optimizer_task is not set. Call _setup_optimizers first.")

        task_loss_sum = 0.0
        cl_loss_sum = 0.0
        data_count = 0
        self.net.train()

        net_core = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
        # prepare lists for gradient clipping
        task_param_list = list(self.optimizer_task.param_groups[0]['params'])
        cl_param_list = list(self.optimizer_cl.param_groups[0]['params']) if self.optimizer_cl is not None else []

        for batch_idx, batch in enumerate(tqdm(train_iter, desc='train')):
            batch = move_to_device(batch, device)

            # ---------- forward for task ----------
            y_hat, _ = self.get_y_score(batch, device)
            y_true = self.get_y_true(batch, device)

            # ensure tensors
            if not torch.is_tensor(y_hat) or not torch.is_tensor(y_true):
                raise RuntimeError("y_hat and y_true must be tensors.")

            # task backward & step (task optimizer)
            self.optimizer_task.zero_grad()
            task_loss = self.loss(y_hat, y_true)
            task_loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(task_param_list, clip_grad_norm)
            self.optimizer_task.step()

            # ---------- recompute forward for cl and update cl optimizer separately ----------
            cl_loss_val = None
            if self.optimizer_cl is not None:
                # recompute the forward so cl_loss has fresh computation graph after task update
                _, cl_loss = self.get_y_score(batch, device)
                if cl_loss is not None:
                    self.optimizer_cl.zero_grad()
                    cl_loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(cl_param_list, clip_grad_norm)
                    self.optimizer_cl.step()
                    cl_loss_val = float(cl_loss.item())
                else:
                    cl_loss_val = None

            batch_size = y_true.shape[0] if y_true.dim() > 0 else 1
            task_loss_sum += float(task_loss.item()) * batch_size
            if cl_loss_val is not None:
                cl_loss_sum += float(cl_loss_val) * batch_size
            data_count += int(batch_size)

            if debug_batches and batch_idx < debug_batches:
                print(f"[DEBUG train] batch={batch_idx} task_loss={task_loss.item():.6f} cl_loss={cl_loss_val if cl_loss_val is not None else 'None'}")

        avg_task_loss = task_loss_sum / data_count if data_count > 0 else 0.0
        avg_cl_loss = cl_loss_sum / data_count if data_count > 0 else 0.0
        return avg_task_loss, avg_cl_loss

    # ======================== 预测 ========================
    def predict_step(self, data_iter, device):
        self.net = self.net.to(device)
        self.net.eval()
        y_true_list, y_hat_list = [], []
        with torch.no_grad():
            for batch in tqdm(data_iter, desc='validate'):
                batch = move_to_device(batch, device)
                y_hat, _ = self.get_y_score(batch, device)
                y_true = self.get_y_true(batch, device)
                y_true_list.append(y_true)
                y_hat_list.append(torch.sigmoid(y_hat))
        if len(y_hat_list) == 0:
            return torch.empty((0,)), torch.empty((0,))
        return torch.cat(y_hat_list, dim=0), torch.cat(y_true_list, dim=0)

    # ======================== Smin ========================
    def init_smin_calculator(self, go_graph_path, train_data, test_data):
        train_annotations = train_data.annotation[train_data.annotation_type]
        test_annotations = test_data.annotation[test_data.annotation_type]
        annotations = list(train_annotations) + list(test_annotations)
        annotations = [a for anno in annotations for a in anno.split(',')]
        go_graph = obonet.read_obo(open(go_graph_path, 'r'))
        self.smin_calculator = SminCalculatorPytorch(go_graph, annotations, train_data.terms_embed)

    # ======================== 评估 ========================
    def evaluate_result(self, y_hat, y_true, device):
        fmax_, threshold_ = fmax_pytorch(y_hat, y_true, device)
        aupr_ = pair_aupr(y_hat, y_true)
        auc_ = auc_pytorch(y_hat, y_true)
        if self.smin_calculator is not None:
            smin_ = self.smin_calculator.smin_score(y_hat, y_true)
        else:
            log.do_print(f'smin_calculator is not initialized.')
            smin_ = 0.0
        return fmax_, threshold_, aupr_, auc_, smin_

    # ======================== Checkpoint ========================
    def save_checkpoint_model(self, checkpoint_dir, epoch, fmax, threshold_, is_best=False, max_ckpt_to_keep=1, tag='best'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint = {
            "net": self.net.state_dict(),
            "optimizer_task": self.optimizer_task.state_dict() if self.optimizer_task else None,
            "optimizer_cl": self.optimizer_cl.state_dict() if self.optimizer_cl else None,
            "epoch": epoch,
            "fmax": fmax,
            "threshold": threshold_
        }
        if is_best:
            old_best = glob.glob(os.path.join(checkpoint_dir, f'{tag}*'))
            for f in old_best: os.remove(f)
            save_path = os.path.join(checkpoint_dir, f'{tag}_model_{epoch}.pth')
        else:
            save_path = os.path.join(checkpoint_dir, f'ckpt_{epoch}.pth')
        torch.save(checkpoint, save_path)
        # 清理低 fmax ckpt
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, 'ckpt_*.pth'))
        if len(ckpt_files) > max_ckpt_to_keep:
            ckpt_files.sort(key=lambda f: torch.load(f, map_location='cpu').get("fmax", 0.0), reverse=True)
            for file_to_delete in ckpt_files[max_ckpt_to_keep:]:
                try: os.remove(file_to_delete)
                except: pass

    # ======================== 训练 & 测试 ========================
    @logging_params
    def train_and_test(self, train_data, validation_data, test_data, batch_size, num_epochs, num_workers=0, lr=1e-3,
                       device='cpu', resume=False, start_epoch=-1, checkpoint_dir='./models/checkpoint',
                       clip_grad_norm=None, cl_lr=None, cl_param_selector=None):
        """
        cl_lr: 如果不指定，默认为 lr（同 learning rate）
        cl_param_selector: 可选字符串或字符串列表，选择需要交给 cl optimizer 的参数名子串。
        """
        if cl_lr is None:
            cl_lr = lr

        # create base optimizer placeholders; actual optimizers will be created after net is moved to device
        # move model to device (handles DataParallel)
        device = self.net_to_devices(device)

        # initialize weights if starting fresh
        if resume and start_epoch != -1:
            start_epoch = self.load_checkpoint_model(checkpoint_dir, start_epoch, device)
        else:
            log.do_print('train a new model')
            self.net.apply(self.init_weights)
            start_epoch = 0

        # set up optimizers now that net is on device (and DataParallel wraps if any)
        self._setup_optimizers(lr_task=lr, lr_cl=cl_lr, cl_param_selector=cl_param_selector)

        train_iter = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        validation_iter = DataLoader(validation_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        test_iter = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

        best_test_fmax = -1
        best_test_result = {}
        best_mse_loss = float('inf')  # 用于记录 task (MSE) 最小的轮次
        best_cl_loss = float('inf')   # 用于记录 cl_loss 最小的轮次

        for epoch in range(start_epoch, num_epochs):
            avg_task_loss, avg_cl_loss = self.train_step(train_iter, device, clip_grad_norm, debug_batches=1)
            log.do_print(f'epoch({epoch}): average task MSE loss: {avg_task_loss:.6f}, average CL loss: {avg_cl_loss:.6f}')

            # 保存 MSE 最小的模型
            if avg_task_loss < best_mse_loss:
                best_mse_loss = avg_task_loss
                self.save_checkpoint_model(checkpoint_dir, epoch, fmax=0.0, threshold_=0.0, is_best=True, tag='mseloss_best')
                log.do_print(f'[MSE Best] epoch={epoch}, avg_mse={best_mse_loss:.6f}')

            # 保存 CL 最小的模型（如果有 cl optimizer）
            if self.optimizer_cl is not None and avg_cl_loss < best_cl_loss:
                best_cl_loss = avg_cl_loss
                self.save_checkpoint_model(checkpoint_dir, epoch, fmax=0.0, threshold_=0.0, is_best=True, tag='clbest')
                log.do_print(f'[CL Best] epoch={epoch}, avg_cl={best_cl_loss:.6f}')

            # 验证集评估
            y_hat_val, y_true_val = self.predict_step(validation_iter, device)
            val_fmax, val_threshold, val_aupr, val_auc, val_smin = self.evaluate_result(y_hat_val, y_true_val, device)
            log.do_print(f'epoch({epoch}): Validation fmax:{val_fmax:.4f}, threshold:{val_threshold}, '
                         f'aupr:{val_aupr:.4f}, auc:{val_auc:.4f}, smin:{val_smin:.4f}')

            # 保存 validation fmax 最佳模型
            if val_fmax > self.best_fmax:
                self.best_fmax = val_fmax
                self.best_aupr = val_aupr
                self.best_smin = val_smin
                self.best_threshold = val_threshold
                self.save_checkpoint_model(checkpoint_dir, epoch, val_fmax, val_threshold, is_best=True, tag='best')

            # 测试集评估
            y_hat_test, y_true_test = self.predict_step(test_iter, device)
            test_fmax, test_threshold, test_aupr, test_auc, test_smin = self.evaluate_result(y_hat_test, y_true_test, device)
            log.do_print(f'epoch({epoch}): Test fmax:{test_fmax:.4f}, threshold:{test_threshold}, '
                         f'aupr:{test_aupr:.4f}, auc:{test_auc:.4f}, smin:{test_smin:.4f}')

            # 保存测试集 fmax 最佳模型
            if test_fmax > best_test_fmax:
                best_test_fmax = test_fmax
                best_test_result = {
                    'epoch': epoch,
                    'fmax': test_fmax,
                    'threshold': test_threshold,
                    'aupr': test_aupr,
                    'auc': test_auc,
                    'smin': test_smin,
                }
                self.save_checkpoint_model(checkpoint_dir, epoch, test_fmax, test_threshold, is_best=True, tag='testbest')

            # 保存普通 ckpt
            self.save_checkpoint_model(checkpoint_dir, epoch, val_fmax, val_threshold, is_best=False)

        log.do_print(f'[FINAL TEST BEST] epoch={best_test_result.get("epoch", "-")}, '
                     f'fmax={best_test_result.get("fmax", 0):.4f}, threshold={best_test_result.get("threshold", 0):.4f}, '
                     f'aupr={best_test_result.get("aupr", 0):.4f}, auc={best_test_result.get("auc", 0):.4f}, '
                     f'smin={best_test_result.get("smin", 0):.4f}, best_mse_loss={best_mse_loss:.6f}, best_cl_loss={best_cl_loss:.6f}')
        return best_test_result

    def net_to_devices(self, device):
        devices = device.split(';')
        print('training on', devices)
        if len(devices) > 1:
            gpus = [int(d) if d.isdigit() else d for d in devices]
            self.net = nn.DataParallel(self.net, device_ids=gpus, output_device=gpus[0])
            device = torch.device(gpus[0])
        self.net = self.net.to(device)
        return device

# ======================== PredGOModel ========================
class PredGOModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_y_score(self, batch, device):
        """
        调用 self.net(...) 并返回 (y_hat, cl_loss)
        兼容以下情况的返回值：
          - net 返回 y_hat (Tensor) -> 返回 (y_hat, None)
          - net 返回 (y_hat, ..., cl_loss) (list/tuple) -> 返回 (y_hat, cl_loss)
          - net 返回 (y_hat, cl_loss) -> 返回 (y_hat, cl_loss)
        cl_loss 可能为 None 或 Tensor（标量或按 batch 平均）。
        """
        # 确保 net 在正确 device 上
        self.net = self.net.to(device)
        batch = batch.to(device)
        esm_features = batch.esm_pre
        ca_coords = batch.aa_ca_coords
        edge_index = batch.aa_edge_index
        ppi_tensor = batch.ppi_data
        num_ppi = batch.num_ppi
        net_ret = self.net(sequence_embedding=esm_features,
                           aa_ca_coords=ca_coords,
                           aa_edge_index=edge_index,
                           num_aa=batch.num_aa,
                           ppi_tensor=ppi_tensor,
                           num_ppi=num_ppi)
        # 如果 net_ret 是 tuple/list，取第一个作为 y_hat，最后一个作为 cl_loss（若 last == y_hat 则认为没有 cl_loss）
        if isinstance(net_ret, (tuple, list)):
            if len(net_ret) == 0:
                raise RuntimeError("Network returned empty tuple/list.")
            y_hat = net_ret[0]
            if len(net_ret) >= 2:
                cl_loss_candidate = net_ret[-1]
                if cl_loss_candidate is y_hat:
                    cl_loss = None
                else:
                    cl_loss = cl_loss_candidate if torch.is_tensor(cl_loss_candidate) else None
            else:
                cl_loss = None
        else:
            y_hat = net_ret
            cl_loss = None

        # ensure devices
        if torch.is_tensor(y_hat):
            y_hat = y_hat.to(device)
        if torch.is_tensor(cl_loss):
            cl_loss = cl_loss.to(device)
        return y_hat, cl_loss

    def get_y_true(self, batch, device):
        return batch.y.to(device)

    @logging_params
    def init_model(self, num_class, aa_node_in_dim, aa_ca_coords, aa_edge_index, egnn_out_dim,
                     ppn_num_heads, num_ppn_layers, hidden_dim, num_layers, **kwargs):
        self.net = PredGONet(num_class=num_class, aa_node_in_dim=aa_node_in_dim,
                             aa_edge_index=aa_edge_index,
                             ppn_num_heads=ppn_num_heads, num_ppn_layers=num_ppn_layers,
                             hidden_dim=hidden_dim, num_layers=num_layers)


    def predict_from_predgo_data(self, data, device='cpu'):
        loader = DataLoader([data])
        score = None
        for batch in loader:
            score, _ = self.get_y_score(batch, device)
        return score

    def predict_from_binary_file(self, target_file, device='cpu'):
        target = torch.load(target_file, map_location=device)
        return self.predict_from_predgo_data(target, device)

    def predict_from_binary_dir(self, binary_dir, device='cpu'):
        files = [(file, os.path.join(binary_dir, file)) for file in os.listdir(binary_dir) if '.pt' in file]
        binary_files = [torch.load(file[1], map_location=device) for file in files]
        loader = DataLoader(binary_files)
        score_list = []
        for batch in loader:
            score, _ = self.get_y_score(batch, device)
            score_list.append(score)
        return score_list, files
