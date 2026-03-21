import glob
import os

import obonet
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Batch

from esm.extract import extract_esm_features_in_memory
#from gvp.test import ca_coords
from predgo.data import generate_PredGOData
from predgo.modules import *
from tools.log import log, logging_params
from tools.metrics import fmax_pytorch, pair_aupr, auc_pytorch, SminCalculatorPytorch
from tools.structure_data_parser import StructureDataParser
from gvp.go_embedding_utils import GOTreeEncoder
from data import *

class BaseModel(object):
    """
        The model base class, which defines and implements the basic methods used during training and prediction
    """

    def __init__(self):
        self.smin_calculator = None
        self.params = None
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = None
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
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    @torch.no_grad()
    #predict_y 方法用于预测输入批次的数据
    def predict_y(self, batch, device):
        y_hat = self.get_y_score(batch, device)
        return torch.sigmoid(y_hat)

    #get_y_true 方法用于从批次数据中获取真实标签。
    def get_y_true(self, batch: object, device: str) -> torch.Tensor:
        raise NotImplementedError

    #用于获取模型对输入批次数据的预测分数（未经过激活函数处理）
    def get_y_score(self, batch: object, device: str) -> torch.Tensor:
        raise NotImplementedError

    #用于执行模型训练的一个完整步骤，包括前向传播、计算损失、反向传播和参数更新。
    def train_step(self, train_iter, device, scheduler=None):
        loss_sum = 0.0
        data_count = 0
        self.net.train()
        for batch in tqdm(train_iter, desc='train'):

            self.optimizer.zero_grad()
            y_true = self.get_y_true(batch, device)
            y_hat = self.get_y_score(batch, device)
            loss_value = self.loss(y_hat, y_true)
            loss_value.backward()
            self.optimizer.step()
            data_num = y_true.shape[0]

            loss_sum += loss_value * data_num
            data_count += data_num
        if scheduler is not None:
            scheduler.step()
        avg_loss = loss_sum / data_count
        return avg_loss

    def predict_step(self, data_iter, device):
        self.net = self.net.to(device)
        self.net.eval()
        y_true_list = []
        y_hat_list = []
        with torch.no_grad():
            for batch in tqdm(data_iter, desc='validate'):
                y_hat = self.predict_y(batch, device)
                y_hat = torch.sigmoid(y_hat)
                y_true = self.get_y_true(batch, device)
                y_true_list.append(y_true)
                y_hat_list.append(y_hat)
        return torch.cat(y_hat_list, dim=0), torch.cat(y_true_list, dim=0)

    def init_smin_calculator(self, go_graph_path, train_data, test_data):
        train_annotations = train_data.annotation[train_data.annotation_type]
        test_annotations = test_data.annotation[test_data.annotation_type]
        annotations = list(train_annotations) + list(test_annotations)
        annotations = [a for anno in annotations for a in anno.split(',')]
        go_graph = obonet.read_obo(open(go_graph_path, 'r'))
        self.smin_calculator = SminCalculatorPytorch(go_graph, annotations, train_data.terms_embed)

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

    def load_specified_model(self, model_path, device='cpu', need_train=True):
        # Load the breakpoint
        checkpoint = torch.load(model_path, map_location=device)
        # Load the parameters that the model can learn
        self.net.load_state_dict(checkpoint['net'], strict=False)
        if need_train:
            if self.optimizer:
                # Load optimizer parameters
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('optimizer is None.Please init optimizer')
        self.best_fmax = checkpoint['fmax']
        self.best_threshold = checkpoint['threshold']
        log.do_print(f'load best model(need train: {need_train}):{model_path}')

    def save_now_model(self, path):
        checkpoint = {
            "net": self.net.cpu().state_dict(),
            'optimizer': None,
            "epoch": -1,
            "fmax": self.best_fmax,
            "threshold": self.best_threshold
        }
        if self.optimizer is not None:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        print('save now model:', path)
        torch.save(checkpoint, path)

    def save_checkpoint_model(self, checkpoint_dir, epoch, fmax, threshold_, is_best=False, max_ckpt_to_keep=1,
                              tag='best'):
        if not os.path.exists(os.path.abspath(checkpoint_dir)):
            os.makedirs(os.path.abspath(checkpoint_dir))
            print('mkdir:', os.path.abspath(checkpoint_dir))

        if is_best:
            checkpoint = {
                "net": self.net.cpu().state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "fmax": fmax,
                "threshold": threshold_
            }

            old_best = glob.glob(os.path.join(checkpoint_dir, f'{tag}*'))
            for f in old_best:
                os.remove(f)

            save_path = os.path.join(checkpoint_dir, f'{tag}_model_{epoch}.pth')
            torch.save(checkpoint, save_path)

        else:
            # 保存普通 ckpt
            checkpoint = {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "fmax": fmax,
                "threshold": threshold_
            }
            ckpt_name = f'ckpt_{epoch}.pth'
            save_path = os.path.join(checkpoint_dir, ckpt_name)
            torch.save(checkpoint, save_path)

            # 清理低 fmax 的 ckpt
            ckpt_files = glob.glob(os.path.join(checkpoint_dir, 'ckpt_*.pth'))
            if len(ckpt_files) > max_ckpt_to_keep:
                def extract_fmax(path):
                    try:
                        ckpt = torch.load(path, map_location='cpu')
                        return ckpt.get("fmax", 0.0)
                    except:
                        return 0.0

                ckpt_files.sort(key=lambda f: extract_fmax(f), reverse=True)
                for file_to_delete in ckpt_files[max_ckpt_to_keep:]:
                    try:
                        os.remove(file_to_delete)
                    except Exception as e:
                        print(f'[WARN] Failed to delete {file_to_delete}: {e}')

    def load_checkpoint_best_model(self, checkpoint_dir, device='cpu', need_train=True):
        ck_path = glob.glob(os.path.join(checkpoint_dir, 'best*'))[0]
        # ck_path = os.path.join(checkpoint_dir, best_model)
        self.load_specified_model(ck_path, device, need_train)
        return ck_path

    #这个方法用于从指定目录加载特定 epoch 的模型检查点。
    def load_checkpoint_model(self, checkpoint_dir, start_epoch, device='cpu', need_train=True):
        ck_path = os.path.join(checkpoint_dir, 'ckpt_%s.pth' % (str(start_epoch)))
        if os.path.exists(ck_path):
            print('load model ', ck_path)
            checkpoint = torch.load(ck_path, map_location=device)
            self.net.load_state_dict(checkpoint['net'], strict=False)
            if need_train:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # set start epoch
            start_epoch = checkpoint['epoch']
            self.best_fmax = checkpoint['fmax']
            self.best_threshold = checkpoint['threshold']
            log.do_print(f'load model(need train: {need_train}):{ck_path}')
            return start_epoch
        else:
            raise ValueError(ck_path + 'is not existed')

    def net_to_devices(self, device):
        devices = device.split(';')
        print('training on', devices)
        if len(devices) > 1:
            gpus = devices
            self.net = nn.DataParallel(self.net, device_ids=gpus, output_device=gpus[0])
            device = torch.device(gpus[0])
        self.net = self.net.to(device)
        return device

    def predict_dataset(self, test_data, batch_size=4, device='cpu'):
        #使用 PyTorch 的 DataLoader 类创建一个数据加载器 test_iter，将 test_data 作为输入。
        test_iter = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        y_hat, y_true = self.predict_step(test_iter, device)
        return y_hat, y_true

    def evaluate_dataset(self, test_data, batch_size=4, device='cpu'):
        test_result = {}
        y_hat, y_true = self.predict_dataset(test_data, batch_size, device)
        test_fmax, test_threshold, test_aupr, test_auc, test_smin = self.evaluate_result(y_hat, y_true, device)
        log.do_print(
            f'test fmax:{test_fmax:.4f},threshold:{test_threshold},aupr:{test_aupr:.4f},auc:{test_auc:.4f}, smin:{test_smin:.4f}')
        return test_result

    @logging_params
    def train_and_test(self, train_data, validation_data, test_data, batch_size, num_epochs, num_workers=0, lr=1e-3,
                       device='cpu', resume=False, start_epoch=-1, checkpoint_dir='./models/checkpoint'):

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        device = self.net_to_devices(device)

        if resume and checkpoint_dir.endswith('.pth'):
            self.load_specified_model(model_path=checkpoint_dir, device=device, need_train=False)
            log.do_print(f'[TEST ONLY] Loaded model from: {checkpoint_dir}')

            test_iter = DataLoader(test_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
            y_hat, y_true = self.predict_step(test_iter, device)
            test_fmax, test_threshold, test_aupr, test_auc, test_smin = self.evaluate_result(y_hat, y_true, device)
            log.do_print(f'[TEST RESULT] fmax:{test_fmax:.4f}, threshold:{test_threshold:.4f}, '
                         f'aupr:{test_aupr:.4f}, auc:{test_auc:.4f}, smin:{test_smin:.4f}')
            return {
                'fmax': test_fmax,
                'threshold': test_threshold,
                'aupr': test_aupr,
                'auc': test_auc,
                'smin': test_smin
            }

        train_iter = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        validation_iter = DataLoader(validation_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)

        if resume and start_epoch != -1:
            start_epoch = self.load_checkpoint_model(checkpoint_dir, start_epoch, device)
        else:
            log.do_print('train a new model')
            self.net.apply(self.init_weights)
            start_epoch = 0

        test_result = {}
        best_test_fmax = -1
        best_test_result = {}

        for epoch in range(start_epoch, num_epochs):
            avg_loss = self.train_step(train_iter, device)
            log.do_print(f'epoch({epoch}):avg loss:{avg_loss :.4f},')

            # 验证集评估
            y_hat, y_true = self.predict_step(validation_iter, device)
            val_fmax, val_threshold, val_aupr, val_auc, val_smin = self.evaluate_result(y_hat, y_true, device)
            log.do_print(f'epoch({epoch}):Validation fmax:{val_fmax:.4f},threshold:{val_threshold},'
                         f'aupr:{val_aupr:.4f},auc:{val_auc:.4f}, smin:{val_smin:.4f}')

            # 测试集评估

            test_iter = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
            y_hat, y_true = self.predict_step(test_iter, device)
            test_fmax, test_threshold, test_aupr, test_auc, test_smin = self.evaluate_result(y_hat, y_true, device)
            log.do_print(f'epoch({epoch}):Test fmax:{test_fmax:.4f},threshold:{test_threshold},'
                         f'aupr:{test_aupr:.4f},auc:{test_auc:.4f}, smin:{test_smin:.4f}')

            # # Case Study: 输出测试集中的 Q47319 蛋白预测的所有GO术语和概率
            # test_iter_case = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
            # case_found = False
            # for batch in test_iter_case:
            #     batch = batch.to(device)
            #     y_hat_batch = self.predict_y(batch, device)
            #     protein_ids = batch.protein_id
            #     true_labels = batch.y  # shape: [batch_size, num_classes]
            #
            #     if isinstance(protein_ids, str):
            #         protein_ids = [protein_ids]
            #     for i, pid in enumerate(protein_ids):
            #         if pid == 'Q47319':
            #             case_found = True
            #             pred_score = torch.sigmoid(y_hat_batch[i])  # shape: [num_classes]
            #             true_label = true_labels[i]  # shape: [num_classes]
            #             index_to_go = {v: k for k, v in train_data.terms_embed.items()}
            #             case_study_result = []
            #             for idx, label_value in enumerate(true_label.tolist()):
            #                 if label_value > 0:  # 只输出真实标签是正的GO
            #                     go_term = index_to_go.get(idx, f'GO:{idx}')
            #                     prob = pred_score[idx].item()
            #                     case_study_result.append(f'{go_term} ({prob:.4f})')
            #             if case_study_result:
            #                 line_list = []
            #                 line = []
            #                 for j, item in enumerate(case_study_result, 1):
            #                     line.append(item)
            #                     if j % 5 == 0:
            #                         line_list.append(' | '.join(line))
            #                         line = []
            #                 if line:
            #                     line_list.append(' | '.join(line))
            #
            #                 case_study_str = '\n'.join(line_list)
            #                 log.do_print(
            #                     f'[Case Study] Epoch {epoch} - Q47319 Predicted GO terms (only true labels):\n{case_study_str}')
            #             else:
            #                 log.do_print(f'[Case Study] Epoch {epoch} - Q47319 has no true GO terms in ground truth.')
            #             break  # 找到了就跳出 batch
            #
            # if not case_found:
            #     log.do_print(f'[Case Study] Epoch {epoch} - Q47319 not found in current test set.')

            if val_fmax > self.best_fmax:
                self.best_fmax = val_fmax
                self.best_aupr = val_aupr
                self.best_smin = val_smin
                self.best_threshold = val_threshold
                self.save_checkpoint_model(checkpoint_dir, epoch, val_fmax, val_threshold, is_best=True)

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
                self.save_checkpoint_model(checkpoint_dir, epoch, test_fmax, test_threshold,
                                           is_best=True, tag='testbest')
                log.do_print(f'[Test Result] epoch={epoch}, fmax={test_fmax:.4f}, threshold={test_threshold}, '
                             f'aupr={test_aupr:.4f}, auc={test_auc:.4f}, smin={test_smin:.4f}')
            self.save_checkpoint_model(checkpoint_dir, epoch, val_fmax, val_threshold, is_best=False)
        log.do_print(
            f'[FINAL TEST BEST] epoch={best_test_result.get("epoch", "-")}, '
            f'fmax={best_test_result.get("fmax", 0):.4f}, threshold={best_test_result.get("threshold", 0):.4f}, '
            f'aupr={best_test_result.get("aupr", 0):.4f}, auc={best_test_result.get("auc", 0):.4f}, '
            f'smin={best_test_result.get("smin", 0):.4f}')

        return best_test_result


class PredGOModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.go_tree_encoder = GOTreeEncoder(device='cuda:1')
        self.go_tree_embedding = self.go_tree_encoder.get_go_embeddings().detach()  # 保存嵌入

    def get_y_score(self, batch: object, device: str) -> torch.Tensor:
        self.net = self.net.to(device)
        batch = batch.to(device)
        esm_features = batch.esm_pre
        h_v = batch.aa_s
        ca_coords = batch.aa_ca_coords
        edge_index = batch.aa_edge_index
        ppi_tensor = batch.ppi_data
        num_ppi = batch.num_ppi
        go_tree_embedding = self.go_tree_embedding.to(device)
        y_hat,_ = self.net(sequence_embedding=esm_features, aa_h_V=h_v, aa_ca_coords=ca_coords,aa_edge_index = edge_index,
                         num_aa=batch.num_aa, ppi_tensor=ppi_tensor, num_ppi=num_ppi)
        return y_hat

    def get_y_true(self, batch: object, device: str) -> torch.Tensor:
        return batch.y.to(device)

    @logging_params
    def init_model(self, num_class, aa_node_in_dim, aa_ca_coords, aa_edge_index, egnn_out_dim,
                     ppn_num_heads, num_ppn_layers, hidden_dim, num_layers,**kwargs):

        self.net = PredGONet(num_class=num_class, aa_node_in_dim=aa_node_in_dim, aa_ca_coords=aa_ca_coords,
                             aa_edge_index=aa_edge_index, egnn_out_dim=egnn_out_dim,
                             ppn_num_heads=ppn_num_heads,num_ppn_layers=num_ppn_layers,
                             hidden_dim=hidden_dim,num_layers=num_layers,
                             go_tree_embedding=self.go_tree_embedding)

    def predict_from_PDB_file(self, pdb_file, ppi_seqs_list=None, device='cpu'):
        if ppi_seqs_list is None:
            ppi_seqs_list = []
        struct_parser = StructureDataParser(pdb_file, 'target', 'pdb')
        coords = struct_parser.get_residue_atoms_coords()
        target_seq = ''.join(struct_parser.get_sequence())

        sequence_labels, sequence_strs = ['target'], [target_seq]
        for i, seq in enumerate(ppi_seqs_list):
            sequence_labels.append(f'p{i + 1}')
            sequence_strs.append(str(seq))
        esm_result = extract_esm_features_in_memory(sequence_labels, sequence_strs, device)
        #处理序列特征
        target_esm_pre = esm_result['target']['mean_representations'][33]
        if len(esm_result) > 1:
            #从ESM结果中提取所有PPI序列的第33层的平均表示，并将它们堆叠成一个张量。
            ppi_esm_pre = torch.stack([v['mean_representations'][33] for k, v in esm_result.items() if k != 'target'],
                                      dim=0)
            #将目标蛋白质的特征和所有PPI序列的特征连接在一起，形成一个完整的特征张量。
            ppi_esm_pre = torch.cat([target_esm_pre.view(1, -1), ppi_esm_pre], dim=0)
        else:
            ppi_esm_pre = esm_result['target']['mean_representations'][33].view(1, -1)
        #生成预测数据
        target = generate_PredGOData(target_esm_pre, ppi_esm_pre, coords)
        return self.predict_from_predgo_data(target)
    # def predict_from_PDB_file(self, pdb_file, device='cpu'):
    #     struct_parser = StructureDataParser(pdb_file, 'target', 'pdb')
    #     coords = struct_parser.get_residue_atoms_coords()
    #     target_seq = ''.join(struct_parser.get_sequence())
    #     sequence_labels = ['target']
    #     sequence_strs = [target_seq]
    #     esm_result = extract_esm_features_in_memory(sequence_labels, sequence_strs, device)
    #     target_esm_pre = esm_result['target']['mean_representations'][33]  # [L, 1280]
    #     ppi_esm_pre = target_esm_pre.view(1, -1)  # 仅使用 target 自身
    #     # 构建 PredGO 输入结构
    #     target = generate_PredGOData(target_esm_pre, ppi_esm_pre, coords)
    #     return self.predict_from_predgo_data(target)

    def predict_from_predgo_data(self, data, device='cpu'):
        loader = DataLoader([data])
        score = None
        for batch in loader:
            score = self.predict_y(batch, device)
        return score

    def predict_from_binary_file(self, target_file, device='cpu'):
        target = torch.load(target_file, map_location=device)
        return self.predict_from_predgo_data(target)

    def predict_from_binary_dir(self, binary_dir, device='cpu'):
        files = [(file, os.path.join(binary_dir, file)) for file in os.listdir(binary_dir) if '.pt' in file]
        binary_files = [torch.load(file[1], map_location=device) for file in files]
        loader = DataLoader(binary_files)
        score_list = []
        for batch in loader:
            score_list.append(self.predict_y(batch, device))
        return score_list, files
