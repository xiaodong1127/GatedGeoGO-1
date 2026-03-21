import copy
import math
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_cluster
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Batch
from tqdm import tqdm
from torch_geometric.data import Data

from esm.extract import extract_esm_mean
from tools.structure_data_parser import StructureDataParser
from tools.tools import read_fasta_file, map_annotation, read_ppi_info

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                 'N': 2, 'Y': 18, 'M': 12}
num_to_letter = {v: k for k, v in letter_to_num.items()}


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def transform_annotation_type(annotation_type):
    if annotation_type == 'all':
        return 'annotation_all'
    elif annotation_type == 'mf':
        return 'annotation_mf'
    elif annotation_type == 'bp':
        return 'annotation_bp'
    elif annotation_type == 'cc':
        return 'annotation_cc'
    else:
        raise ValueError('annotation_type can only by all;mf;bp;cc')


class PredGOData(GeometricData):
    def __init__(self, esm_pre, num_aa, ppi_data, num_ppi, aa_s, aa_ca_coords, aa_edge_index,seq=None):
        super().__init__()
        self.esm_pre = esm_pre
        self.num_aa = num_aa
        self.aa_s = aa_s
        self.aa_ca_coords = aa_ca_coords
        self.aa_edge_index = aa_edge_index
        self.ppi_data = ppi_data
        self.num_ppi = num_ppi
        self.seq = seq

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'aa_edge_index':
            return self.aa_s.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def generate_PredGOData(target_esm_pre, ppi_esm_pre, coords):
    features_generator = ResidueGraphFeaturesGenerator()
    residue_features = features_generator.generate_graph_features(coords)
    data = PredGOData(esm_pre=target_esm_pre.clone(), num_aa=len(coords['CA']), ppi_data=ppi_esm_pre,
                      num_ppi=ppi_esm_pre.shape[0], aa_s=residue_features['node_s'],
                      aa_ca_coords=residue_features['ca_coords'],
                      aa_edge_index=residue_features['edge_index'])
    return data

class ResidueGraphFeaturesGenerator(object):
    #主要功能是从蛋白质的三维坐标中提取和生成图形特征，以便用于后续的模型训练或分析。

    def __init__(self, device='cpu'):
        self.top_k = 20#用于确定每个Cα原子的邻居数量
        self.device = device

    def generate_graph_features(self, coords, device='cpu'):
        # coords={k:v[:100] for k,v in coords.items()}
        CA_coords = torch.as_tensor(coords['CA'], dtype=torch.float32, device=device)
        C_coords = torch.as_tensor(coords['C'], dtype=torch.float32, device=device)
        N_coords = torch.as_tensor(coords['N'], dtype=torch.float32, device=device)
        # O_coords = torch.as_tensor(coords['O'], dtype=torch.float32, device=device)

        edge_index = torch_cluster.knn_graph(CA_coords, k=self.top_k)
        # # Dihedral Angle N_coords,CA_coords,C_coords
        dihedrals = self._dihedrals(N_coords, CA_coords, C_coords)
        # 添加键长特征
        bond_lengths_features = self.calculate_bond_lengths(N_coords, CA_coords, C_coords)
        # 添加键角特征
        bond_angles_features = self.calculate_bond_angles(N_coords, CA_coords, C_coords)
        node_s = torch.cat([dihedrals, bond_lengths_features, bond_angles_features], dim=-1)
        node_s = torch.nan_to_num(node_s)
        ca_coords = CA_coords

        return {'node_s': node_s,  'ca_coords': ca_coords , 'edge_index': edge_index }

    def calculate_bond_lengths(self, N_coords, CA_coords, C_coords):
        """
        计算N-Ca, Ca-C, 和 N-C 键长。
        """
        bond_lengths_NC = torch.norm(CA_coords - N_coords, dim=1)
        bond_lengths_CaC = torch.norm(C_coords - CA_coords, dim=1)
        bond_lengths_NC_term = torch.norm(C_coords - N_coords, dim=1)

        # 将所有键长组合成一个特征向量
        bond_lengths_features = torch.stack([bond_lengths_NC, bond_lengths_CaC, bond_lengths_NC_term], dim=1)

        return bond_lengths_features

    def calculate_bond_angles(self, N_coords, CA_coords, C_coords):
        """
        计算N-Ca-C 键角。
        """
        vec_1 = N_coords - CA_coords
        vec_2 = C_coords - CA_coords

        cos_angle = F.cosine_similarity(vec_1, vec_2, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)  # 防止acos超出范围
        angles = torch.acos(cos_angle)

        # 将角度转换为特征表示形式，例如使用cos和sin
        angle_features = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        return angle_features

    def _dihedrals(self, N_coords, CA_coords, C_coords, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        # First 3 coordinates are N, CA, C
        X = torch.stack((N_coords, CA_coords, C_coords), dim=1)
        X = torch.reshape(X, [3 * X.shape[0], 3])
        # X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        # Shifted slices of unit vectors
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals torch.cross
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features



class PredGODataset(GeometricDataset):

    def __init__(self, root, annotation_path, sequences_path=None, ppi_sequences_path=None, terms_file_path=None,
                 struct_dir_path=None,
                 esm_dir_path=None,
                 ppi_path=None,
                 annotation_type='all',
                 seq_max_len=1000, seq_min_len=16, max_ppi_data_num=200, data_overwrite=False, skip_load=True,
                 device='cpu'):
        self.name = 'PredGO'

        self.ppi_path = ppi_path

        self.seq_min_len = seq_min_len
        self.seq_max_len = seq_max_len
        self.max_ppi_data_num = max_ppi_data_num
        self.esm_dir_path = esm_dir_path
        self.struct_dir_path = struct_dir_path
        self.data_overwrite = data_overwrite
        self.skip_load = skip_load
        self.device = device

        self.terms_embed = map_annotation(terms_file_path, annotation_type)
        self.terms_embed_num = len(self.terms_embed)
        #读取注释数据文件（以制表符分隔的CSV格式），并将其索引设为蛋白质ID。
        self.annotation = pd.read_csv(annotation_path, sep='\t', index_col=0)
        self.annotation = self.annotation.set_index('protein_id')
        #通过调用 transform_annotation_type 函数转换注释类型。
        self.annotation_type = transform_annotation_type(annotation_type)

        if 'orgs' in self.annotation.columns:
            self.annotation = self.annotation[[self.annotation_type, 'orgs']]
        else:
            self.annotation = self.annotation[[self.annotation_type]]
        if not self.skip_load:
            self.struct_file_dict = {file.split('.')[0]: os.path.join(self.struct_dir_path, file)
                                     for file in os.listdir(self.struct_dir_path)}
            self.esm_dict = {file.split('.pt')[0]: os.path.join(self.esm_dir_path, file) for file in
                             os.listdir(self.esm_dir_path)}
            self.sequence_dict = {rec.id: str(rec.seq) for rec in read_fasta_file(sequences_path) if
                                  seq_min_len <= len(rec.seq) <= seq_max_len}
            self.ppi_sequence_dict = {rec.id: str(rec.seq) for rec in read_fasta_file(ppi_sequences_path)}
            self.ppi_dict = read_ppi_info(self.ppi_path)
        self.ckeck_and_filter_data()
        self.need_processed_files = None
        self.init_need_processed_files()

        super().__init__(root, None, None, None)
        if not self.skip_load:
            del self.struct_file_dict
            del self.sequence_dict
            del self.esm_dict
            del self.ppi_dict
            del self.ppi_sequence_dict

    def init_need_processed_files(self):
        pid_set = set(self.annotation.index)
        self.need_processed_files = [f'{self.name}_{file}.pt' for file in pid_set]

    def ckeck_and_filter_data(self):

        self.annotation = self.annotation[self.annotation[self.annotation_type].notna()]
        annotation_list = [line.split(',') for line in self.annotation[self.annotation_type]]
        for i, annotation in enumerate(annotation_list):
            annotation_list[i] = ','.join([a for a in annotation if a in self.terms_embed])
        self.annotation[self.annotation_type] = annotation_list
        self.annotation = self.annotation[self.annotation[self.annotation_type] != '']
        if not self.skip_load:
            """
            Filter out no sequence data 
            Because the length is filtered when the sequence is read, 
            annotations of proteins that do not have the desired length are also filtered out
            """
            self.annotation = self.annotation[self.annotation.index.isin(self.sequence_dict.keys())]

            ids = set(self.annotation.index)
            missing_struct = ids - self.struct_file_dict.keys()
            if len(missing_struct) > 0:
                print(
                    f'{len(missing_struct)} struct files are missing, '
                    f'please put the structure file into the folder{self.struct_dir_path}')
                raise FileNotFoundError(f'{missing_struct} not found')
            missing_esm = ids - self.esm_dict.keys()
            if len(missing_esm) > 0:
                print(
                    f'{len(missing_esm)} target proteins esm files are missing, '
                    f'try using ESM to generate')
                self.run_esm(missing_esm, self.sequence_dict, self.device)
                # Update the esm dictionary mapping
                self.esm_dict = {file.split('.pt')[0]: os.path.join(self.esm_dir_path, file) for file in
                                 os.listdir(self.esm_dir_path)}
            ppi_list = []
            for i in ids:
                if i in self.ppi_dict:
                    ppi_list.extend(self.ppi_dict[i])
            missing_ppi_esm = set(ppi_list) - self.esm_dict.keys()
            if len(missing_ppi_esm) > 0:
                print(
                    f'{len(missing_ppi_esm)} interacting proteins esm files are missing, '
                    f'try using ESM to generate')
                self.run_esm(missing_ppi_esm, self.ppi_sequence_dict, self.device)
                # Update the esm dictionary mapping
                self.esm_dict = {file.split('.pt')[0]: os.path.join(self.esm_dir_path, file) for file in
                                 os.listdir(self.esm_dir_path)}

    def run_esm(self, ids, sequence_dict, device):
        temp_file = os.path.join(self.esm_dir_path, 'tep.fasta')
        temp_recs = [SeqRecord(Seq(sequence_dict[idx]),
                               id=idx,
                               description="") for idx in ids]
        SeqIO.write(temp_recs, temp_file, "fasta")
        extract_esm_mean(temp_file, self.esm_dir_path, device)
        os.remove(temp_file)

    def get_sub_dataset_by_orgs(self, orgs):
        if 'orgs' not in self.annotation.columns:
            raise ValueError('this dataset has not orgs column.')
        new_dataset = copy.deepcopy(self)

        if isinstance(orgs, int):
            new_dataset.annotation = new_dataset.annotation[new_dataset.annotation['orgs'] == orgs]
        elif isinstance(orgs, list) or isinstance(orgs, set):
            orgs = set(orgs)
            new_dataset.annotation = new_dataset.annotation[new_dataset.annotation['orgs'].isin(orgs)]
        else:
            raise TypeError('orgs can only be integers, lists, or sets.')
        new_dataset.init_need_processed_files()
        return new_dataset

    def get_sub_dataset_in_set(self, id_set):
        new_dataset = copy.deepcopy(self)
        new_dataset.annotation = new_dataset.annotation[new_dataset.annotation.index.isin(id_set)]
        new_dataset.init_need_processed_files()
        return new_dataset

    def get_sub_dataset_not_in_set(self, id_set):
        new_dataset = copy.deepcopy(self)
        new_dataset.annotation = new_dataset.annotation[~new_dataset.annotation.index.isin(id_set)]
        new_dataset.init_need_processed_files()
        return new_dataset

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.need_processed_files

    def download(self):
        pass

    def process(self):
        if self.skip_load:
            raise FileNotFoundError('raw data has not processed ,can not skip load')
        processed_files = {file for file in os.listdir(self.processed_dir)}
        cache_ppn_node_features = {}

        def build_ppi_feature(protein_id_):
            ppi_set = set()
            if protein_id_ in self.ppi_dict:
                ppi_set = set(self.ppi_dict[protein_id_])
            result = [torch.load(self.esm_dict[protein_id_])['mean_representations'][33]]

            if protein_id_ in ppi_set:
                ppi_set.remove(protein_id_)
            for p in ppi_set:
                if p not in cache_ppn_node_features:
                    cache_ppn_node_features[p] = torch.load(self.esm_dict[p])['mean_representations'][33]
                result.append(cache_ppn_node_features[p])

            return torch.stack(result, dim=0)

        #主要用于从注释信息中提取和处理蛋白质数据，生成用于训练的特征数据
        for i, row in tqdm(self.annotation.iterrows(), total=len(self.annotation)):
            protein_id = row.name
            if self.data_overwrite is False and f'{self.name}_{protein_id}.pt' in processed_files:
                continue
            #加载蛋白质的ESM特征：
            if protein_id not in cache_ppn_node_features:
                esm_pre = torch.load(self.esm_dict[protein_id])['mean_representations'][33]
            else:
                esm_pre = cache_ppn_node_features[protein_id]
            #解析蛋白质结构文件，提取蛋白质的PPI特征：
            struct_parser = StructureDataParser(self.struct_file_dict[protein_id], protein_id, 'pdb')
            coords = struct_parser.get_residue_atoms_coords()

            ppi_pre = build_ppi_feature(protein_id)
            predgo_data = generate_PredGOData(esm_pre, ppi_pre, coords)

            torch.save(predgo_data, osp.join(self.processed_dir, f'{self.name}_{protein_id}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        protein_id = self.annotation.iloc[idx].name
        data = torch.load(osp.join(self.processed_dir, f'{self.name}_{protein_id}.pt'))
        # Add Y label
        data.y = self.annotations2y(self.annotation.iloc[idx][self.annotation_type].split(',')).unsqueeze(0)
        data.protein_id = protein_id
        return data

    def annotations2y(self, annotations):
        #annotations2y方法则负责根据注释生成对应的标签，方便后续机器学习模型的训练和预测。
        """
        Convert the input to the y-tag pytorch version
        :param annotations: An annotation that needs to be transformed
        :return:
        """
        y = torch.zeros(self.terms_embed_num, dtype=torch.float32)
        for anno in annotations:
            if anno in self.terms_embed:
                y[self.terms_embed[anno]] = 1
        return y
