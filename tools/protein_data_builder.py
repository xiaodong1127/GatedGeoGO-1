import csv
import gzip
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from Bio import SwissProt
from tqdm import tqdm

from tools.tools import gafiterator, load_go_graph

CAFA3_TARGETS = {'7227', '208963', '237561', '3702', '7955', '8355', '9606', '10090', '10116', '44689', '83333',
                 '85962', '99287', '160488', '170187', '223283', '224308', '243232', '243273', '273057', '284812',
                 '321314', '559292'}

CAFA4_TARGETS = {'287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
                 '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'}
ORGS = {
    '9606': "Homo sapiens",
    '10090': "Mus musculus",
    '10116': "Rattus norvegicus",
    '3702': "Arabidopsis thaliana",
    '83333': "Escherichia coli K-12",
    '7227': "Drosophila melanogaster",
    '287': "Pseudomonas aeruginosa",
    '559292': "Saccharomyces cerevisiae ATCC 204508",
    '284812': "Schizosaccharomyces pombe ATCC 24843",
    '7955': "Danio rerio",
    '44689': "Dictyostelium discoideum",
    '243273': "Mycoplasma genitalium ATCC 33530",
    '6239': "Caenorhabditis elegans",
    '226900': "Bacillus cereus ATCC 14579",
    '4577': "Zea Mays [All names]",
    '9823': "Sus scrofa",
    '99287': "Salmonella typhymurium ATCC 700720"
}
BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
ROOT_TERMS_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS
}
NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}
NAMESPACES1 = {
    'cellular_component': 'cc',
    'molecular_function': 'mf',
    'biological_process': 'bp'
}
EXP_CODES = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
             'HTP', 'HDA', 'HMP', 'HGI', 'HEP'}


class AnnotationDatasetBuilder:
    def __init__(self, goa_path, go_graph_path, sprot_path, t0, t1, t2, orgs=ORGS.keys()):
        self.sprot_df = None
        self.goa_path = goa_path
        self.go_graph_path = go_graph_path
        self.sprot_path = sprot_path
        self.go_graph = self._load_go_graph()
        self.orgs = orgs
        self.goa = self._load_goa()
        self.train_dataset, self.validation_dataset, self.test_dataset = self.build_dataset(t0, t1, t2)
        print('filter orgs (validation')
        self.validation_dataset = self.filter_by_orgs(self.validation_dataset)
        print('filter orgs (test')
        self.test_dataset = self.filter_by_orgs(self.test_dataset)

        self.all_count = {}
        self.cc_count = {}
        self.mf_count = {}
        self.bp_count = {}

    def filter_by_struct(self, struct_dir):
        #根据结构文件目录中的文件来过滤训练数据集、验证数据集和测试数据集中的蛋白质条目。具体来说，这个方法会检查每个数据集中蛋白质ID是否在结构文件目录中存在对应文件，如果不存在则将该蛋白质条目从数据集中移除。
        struct_ids = {file.split('_')[0] for file in os.listdir(struct_dir)}

        struct_files = {file for file in os.listdir(struct_dir)}
        missing_struct = set()
        print('train dataset filter...')
        print('size:', self.train_dataset.shape)
        for i, row in tqdm(self.train_dataset.iterrows()):
            # file_name = get_struct_file_name(row['protein_id'], 'pdb')
            if row['protein_id'] not in struct_ids:
                missing_struct.add(row['protein_id'])
        print('missing_struct:', len(missing_struct))
        self.train_dataset = self.train_dataset[~self.train_dataset['protein_id'].isin(missing_struct)]

        print('val dataset filter...')
        print('size:', self.validation_dataset.shape)
        missing_struct = set()
        for i, row in tqdm(self.validation_dataset.iterrows()):
            # file_name = get_struct_file_name(row['protein_id'], 'pdb')
            if row['protein_id'] not in struct_ids:
                missing_struct.add(row['protein_id'])
        print('missing_struct:', len(missing_struct))
        self.validation_dataset = self.validation_dataset[~self.validation_dataset['protein_id'].isin(missing_struct)]

        print('test dataset filter...')
        print('size:', self.test_dataset.shape)
        missing_struct = set()
        for i, row in tqdm(self.test_dataset.iterrows()):
            # file_name = get_struct_file_name(row['protein_id'], 'pdb')
            if row['protein_id'] not in struct_ids:
                missing_struct.add(row['protein_id'])
        print('missing_struct:', len(missing_struct))
        #使用~操作符和isin()方法从self.test_dataset中过滤掉那些在missing_struct集合中的蛋白质ID，只保留有效蛋白质条目。
        self.test_dataset = self.test_dataset[~self.test_dataset['protein_id'].isin(missing_struct)]

    def load_coords(self, struct_file_path):
        if struct_file_path.endswith('.json'):
            coords_file = open(struct_file_path, 'r')
            coords = json.load(coords_file)
            coords_file.close()
            return coords
        elif struct_file_path.endswith('.npy'):
            coords = np.load(struct_file_path, allow_pickle=True).item()
            return coords
        elif struct_file_path.endswith('.npz'):
            coords = np.load(struct_file_path, allow_pickle=True)['coords'].item()
            coords = {k: v[0] for k, v in coords.items()}
            return coords
        else:
            raise NotImplementedError('Only the .json .npy .npz formats are supported')

    def filter_annotation_by_count(self, min_num=10):
        self.count_annotation_from_trainset(min_num)
        self.filter_dataset_annotation_by_count(self.train_dataset)
        self.filter_dataset_annotation_by_count(self.validation_dataset)
        self.filter_dataset_annotation_by_count(self.test_dataset)

    def save(self, save_dir):
        self.save_dataset_to_tsv(save_dir)
        self.save_annotation_tsv(os.path.join(save_dir, './annotation.tsv'))

    def save_dataset_to_tsv(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.train_dataset.to_csv(os.path.join(dir_path, './train.tsv'), sep='\t')
        self.validation_dataset.to_csv(os.path.join(dir_path, './validation.tsv'), sep='\t')
        self.test_dataset.to_csv(os.path.join(dir_path, './test.tsv'), sep='\t')

    def save_annotation_tsv(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["##GO-TERMS ALL " + str(len(self.all_count.keys()))])
            writer.writerow(list(self.all_count.keys()))
            writer.writerow(["##GO-TERMS mf " + str(len(self.mf_count.keys()))])
            writer.writerow(list(self.mf_count.keys()))
            writer.writerow(["##GO-TERMS bp " + str(len(self.bp_count.keys()))])
            writer.writerow(list(self.bp_count.keys()))
            writer.writerow(["##GO-TERMS cc " + str(len(self.cc_count.keys()))])
            writer.writerow(list(self.cc_count.keys()))

    def combine_sequences(self, spot_path=None):
        if spot_path is not None:
            self.sprot_path = spot_path
        self.train_dataset['sequence'] = ''
        self.validation_dataset['sequence'] = ''
        self.test_dataset['sequence'] = ''
        print('load sprot..')
        self.sprot_df = self._load_sprot()
        no_sequence_id = []
        print('train dataset combine...')
        for i, row in tqdm(self.train_dataset.iterrows()):
            try:
                temp_df = self.sprot_df.loc[row['protein_id']]
                row['sequence'] = temp_df['sequences']
            except KeyError:
                no_sequence_id.append(row['protein_id'])
                print(row['protein_id'], 'No sequence was matched(train')
        print('validation dataset combine...')
        for i, row in tqdm(self.validation_dataset.iterrows()):
            try:
                temp_df = self.sprot_df.loc[row['protein_id']]
                row['sequence'] = temp_df['sequences']
            except KeyError:
                no_sequence_id.append(row['protein_id'])
                print(row['protein_id'], 'No sequence was matched(val')
        print('test dataset combine...')
        for i, row in tqdm(self.test_dataset.iterrows()):
            try:
                temp_df = self.sprot_df.loc[row['protein_id']]
                row['sequence'] = temp_df['sequences']
            except KeyError:
                no_sequence_id.append(row['protein_id'])
                print(row['protein_id'], 'No sequence was matched(test')
        print('no sequence_id:', no_sequence_id)

    def filter_by_orgs(self, dataframe):
        return dataframe[
            [True if d['orgs'] in self.orgs else False for i, d in tqdm(dataframe.iterrows(), desc='filter orgs')]]

    def filter_dataset_annotation_by_count(self, dataset):
        for i, row in tqdm(dataset.iterrows(), desc='filter annotation by count'):
            all_ = row['annotation_all'].split(',')
            new_all = [anno for anno in all_ if anno in self.all_count]
            row['annotation_all'] = ','.join(new_all)
            cc_ = row['annotation_cc'].split(',')
            new_cc = [anno for anno in cc_ if anno in self.cc_count]
            row['annotation_cc'] = ','.join(new_cc)
            mf_ = row['annotation_mf'].split(',')
            new_mf = [anno for anno in mf_ if anno in self.mf_count]
            row['annotation_mf'] = ','.join(new_mf)
            bp_ = row['annotation_bp'].split(',')
            new_bp = [anno for anno in bp_ if anno in self.bp_count]
            row['annotation_bp'] = ','.join(new_bp)

    def count_annotation_from_trainset(self, min_num):
        # count
        for i, row in tqdm(self.train_dataset.iterrows(), desc='count annotation from train dataset'):
            #对于每个蛋白质条目，获取其所有注释项，并按逗号分割。然后统计每个注释项在训练数据集中的出现次数，存储在self.all_count字典中。
            all_ = row['annotation_all'].split(',')
            for anno in all_:
                if anno in self.all_count:
                    self.all_count[anno] += 1
                else:
                    self.all_count[anno] = 1
            cc_ = row['annotation_cc'].split(',')
            for anno in cc_:
                if anno in self.cc_count:
                    self.cc_count[anno] += 1
                else:
                    self.cc_count[anno] = 1
            mf_ = row['annotation_mf'].split(',')
            for anno in mf_:
                if anno in self.mf_count:
                    self.mf_count[anno] += 1
                else:
                    self.mf_count[anno] = 1
            bp_ = row['annotation_bp'].split(',')
            for anno in bp_:
                if anno in self.bp_count:
                    self.bp_count[anno] += 1
                else:
                    self.bp_count[anno] = 1
        # 删除''
        if '' in self.all_count.keys():
            self.all_count.pop('')
        if '' in self.cc_count.keys():
            self.cc_count.pop('')
        if '' in self.mf_count.keys():
            self.mf_count.pop('')
        if '' in self.bp_count.keys():
            self.bp_count.pop('')

        self.all_count = {k: v for k, v in self.all_count.items() if v > min_num}
        self.cc_count = {k: v for k, v in self.cc_count.items() if v > min_num}
        self.mf_count = {k: v for k, v in self.mf_count.items() if v > min_num}
        self.bp_count = {k: v for k, v in self.bp_count.items() if v > min_num}

    def _combine_annotation(self, data_df):
        '''
        GO is propagated in the GO tree and divided into three categories: mf cc bp
        :param data_df:
        :return:
        '''
        protein_id_list = []
        annotation_all = []
        cc = []
        mf = []
        bp = []
        orgs = []
        #对输入的data_df数据框按照DB_Object_ID（即蛋白质ID）进行分组。这样做的目的是为了对每个蛋白质的所有注释信息进行合并和处理。
        for group in tqdm(data_df.groupby('DB_Object_ID'), desc='combine annotation'):
            protein_id_list.append(group[0])
            orgs.append(group[1].iloc[0]['orgs'])
            go_ids = set()
            for i, row in group[1].iterrows():
                go_ids.add(row['GO_ID'])
                # propagated in GO tree
                temp_go_ids = nx.descendants(self.go_graph, row['GO_ID'])
                go_ids = go_ids.union(temp_go_ids)
            go_ids = go_ids.difference(set(''))
            annotation_all.append(','.join(go_ids))
            go_ids = go_ids.difference(ROOT_TERMS_DICT.values())
            #difference表示去除

            # Grouping annotations
            go_cc = set()
            go_mf = set()
            go_bp = set()
            for go in go_ids:
                if self.go_graph.nodes[go]['namespace'] == 'molecular_function':
                    go_mf.add(go)
                elif self.go_graph.nodes[go]['namespace'] == 'biological_process':
                    go_bp.add(go)
                elif self.go_graph.nodes[go]['namespace'] == 'cellular_component':
                    go_cc.add(go)
                else:
                    print('Does not belong to mf, bp and cc:', self.go_graph.nodes[go])
            go_cc = go_cc.difference(set(''))
            go_mf = go_mf.difference(set(''))
            go_bp = go_bp.difference(set(''))
            cc.append(','.join(go_cc))
            mf.append(','.join(go_mf))
            bp.append(','.join(go_bp))
        return pd.DataFrame({
            'protein_id': protein_id_list,
            'annotation_all': annotation_all,
            'orgs': orgs,
            'annotation_mf': mf,
            'annotation_bp': bp,
            'annotation_cc': cc
        })

    def build_dataset(self, t0, t1, t2):
        print('build train dataset')
        train = self.get_annotation_by_time(start=None, end=t0)
        before_t0_protein = set([row['DB_Object_ID'] for i, row in train.iterrows()])

        print('build validation dataset')
        t1_dataset = self.get_annotation_by_time(start=t0, end=t1)
        before_t1_protein = set([row['DB_Object_ID'] for i, row in t1_dataset.iterrows()])
        before_t1_protein = before_t1_protein.union(before_t0_protein)
        validation = t1_dataset[
            [True if d['DB_Object_ID'] not in before_t0_protein else False for i, d in t1_dataset.iterrows()]]

        print('build test dataset')
        t2_dataset = self.get_annotation_by_time(start=t1, end=t2)
        test = t2_dataset[
            [True if d['DB_Object_ID'] not in before_t1_protein else False for i, d in t2_dataset.iterrows()]]

        train_dataset = self._combine_annotation(train)
        validation_dataset = self._combine_annotation(validation)
        test_dataset = self._combine_annotation(test)
        return train_dataset, validation_dataset, test_dataset

    def get_annotation_by_time(self, start, end):
        if start is not None and end is not None:
            return self.goa[lambda x: (x['Date'] >= start) & (x['Date'] < end)]
        elif start is None:
            return self.goa[lambda x: (x['Date'] < end)]
        elif end is None:
            return self.goa[lambda x: (x['Date'] >= start)]
        else:
            return self.goa

    def _load_sprot(self):
        # Protein information was extracted from Swiss Prot and transformed into pandas
        if self.sprot_path.endswith('gz'):
            handle = gzip.open(self.sprot_path, "rt")
        else:
            handle = open(self.sprot_path, "rt")

        proteins = []
        protein_id = []
        sequences = []
        sport_annotations = []
        orgs = []
        for record in tqdm(SwissProt.parse(handle), desc='load sprot'):
            proteins.append(record.entry_name)
            protein_id.append(record.accessions[0])
            sequences.append(record.sequence)
            orgs.append(record.taxonomy_id[0])
            if len(record.taxonomy_id) != 1:
                print(record)
            annotations_temp = []
            for reference in record.cross_references:
                if reference[0] == 'GO':
                    annotations_temp.append(reference)
            sport_annotations.append(annotations_temp)
        df = pd.DataFrame({
            'proteins': proteins,
            # 'protein_id': protein_id,
            'sequences': sequences,
            'sport_annotations': sport_annotations,
            'orgs': orgs
        }, index=protein_id)
        return df

    def _load_go_graph(self):
        return load_go_graph(self.go_graph_path)

    def _load_goa(self):
        # Transform the GOA data into pandas
        #该方法的主要目的是加载和处理来自Gene Ontology Annotation (GOA)文件的数据，将其转换为Pandas数据框，并进行特定的过滤操作
        if self.goa_path.endswith('gz'):
            handle = gzip.open(self.goa_path, "rt")
        else:
            handle = open(self.goa_path, "rt")
        print('load goa...')
        # annotations = [annotation for annotation in tqdm(gafiterator(handle), desc='load goa')]
        annotations = []
        for annotation in tqdm(gafiterator(handle), desc='load and filter evidence'):
            if annotation['Evidence'] in EXP_CODES and annotation['GO_ID'] in self.go_graph.nodes:
                annotations.append(annotation)
        df = pd.DataFrame(annotations)
        print('into dataframe...')
        del annotations
        print('del cache...')
        df['orgs'] = ''
        for i, row in tqdm(df.iterrows(), desc='extract orgs'):
            df.loc[i, 'orgs'] = row['Taxon_ID'][0].split(':')[-1]
        return df

    def filter_annotation(self, df):
        # Filter non-experimental comments
        df = df[[True if d['Evidence'] in EXP_CODES else False for i, d in tqdm(df.iterrows(), desc='filter evidence')]]
        # Filter terms that are not in the GO tree
        df = df[[True if d['GO_ID'] in self.go_graph.nodes else False for i, d in
                 tqdm(df.iterrows(), desc='filter go tree')]]
        return df
