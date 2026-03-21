import argparse
import os.path
import time
from predgo.data import PredGODataset, PredGOData
from predgo.model import PredGOModel
from tools.log import log
from tools.tools import print_args_params
from torch_geometric.data import Data
import random
import numpy as np
import torch
import os

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str,
                        default=r'data/CAFA3/PredGODataset')

    parser.add_argument('--train_annotation_path', type=str,
                        default=r'data/CAFA3/train_data_train_standardAA_1000_afdb.tsv')
    parser.add_argument('--train_seq_dir', type=str,
                        default=r'data/CAFA3/train_data_train_standardAA_1000_afdb.fasta')

    parser.add_argument('--validation_annotation_path', type=str,
                        default=r'data/CAFA3/train_data_valid_standardAA_1000_afdb.tsv')
    parser.add_argument('--validation_seq_dir', type=str,
                        default=r'data/CAFA3/train_data_valid_standardAA_1000_afdb.fasta')

    parser.add_argument('--test_annotation_path', type=str,
                        default=r'data/CAFA3/test_data_standardAA_1000.tsv')
    parser.add_argument('--test_seq_dir', type=str,
                        default=r'data/CAFA3/test_data_standardAA_1000.fasta')

    parser.add_argument('--term_file_path', type=str,
                        default="data/CAFA3/terms.tsv",
                        help="annotation_terms_file_path.")
    parser.add_argument('--esm_dir_path', type=str,
                        default="data/CAFA3/esm_dir", help="esm_dir_path.")
    parser.add_argument('--struct_dir_path', type=str,
                        default="data/CAFA3/afdb_dir", help="esm_dir_path.")
    parser.add_argument('--ppi_path', type=str,
                        default="data/CAFA3/ppi_score.tsv")
    parser.add_argument('--ppi_seqs', type=str,
                        default="data/CAFA3/ppi_seqs.fasta")

    parser.add_argument('--go_graph_path', type=str, default="data/CAFA3/go.obo",
                        help="go_graph_path.")

    parser.add_argument('--ont', type=str,
                        default="bp", help="mf bp cc type.")
    parser.add_argument('--device', type=str,
                        default="cuda:1", help="device.")
    parser.add_argument('--batch_size', type=int,
                        default="84", help="batch_size.")
    parser.add_argument('--num_epochs', type=int,
                        default="15", help="num_epochs.")

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    print_args_params(args)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('device:', args.device)

    train_dataset = PredGODataset(root=args.root_dir, annotation_path=args.train_annotation_path,
                                  sequences_path=args.train_seq_dir,
                                  terms_file_path=args.term_file_path,
                                  struct_dir_path=args.struct_dir_path,
                                  esm_dir_path=args.esm_dir_path,
                                  ppi_path=args.ppi_path,
                                  ppi_sequences_path=args.ppi_seqs,
                                  annotation_type=args.ont,
                                  seq_max_len=1000, seq_min_len=16, max_ppi_data_num=200,
                                  data_overwrite=False, skip_load=False, device=args.device)

    val_dataset = PredGODataset(root=args.root_dir, annotation_path=args.validation_annotation_path,
                                sequences_path=args.validation_seq_dir,
                                terms_file_path=args.term_file_path,
                                struct_dir_path=args.struct_dir_path,
                                esm_dir_path=args.esm_dir_path,
                                ppi_path=args.ppi_path,
                                ppi_sequences_path=args.ppi_seqs,
                                annotation_type=args.ont,
                                seq_max_len=1000, seq_min_len=16, max_ppi_data_num=200,
                                data_overwrite=False, skip_load=False, device=args.device)

    test_dataset = PredGODataset(root=args.root_dir, annotation_path=args.test_annotation_path,
                                 sequences_path=args.test_seq_dir,
                                 terms_file_path=args.term_file_path,
                                 struct_dir_path=args.struct_dir_path,
                                 esm_dir_path=args.esm_dir_path,
                                 ppi_path=args.ppi_path,
                                 ppi_sequences_path=args.ppi_seqs,
                                 annotation_type=args.ont,
                                 seq_max_len=1000, seq_min_len=16, max_ppi_data_num=200,
                                 data_overwrite=False, skip_load=False, device=args.device)

    log.do_print(f'train_dataset len: {len(train_dataset)}')
    log.do_print(f'val_dataset len: {len(val_dataset)}')
    log.do_print(f'test_dataset len: {len(test_dataset)}')

    num_class = test_dataset.terms_embed_num

    model = PredGOModel()

    model.init_smin_calculator(go_graph_path=args.go_graph_path, train_data=train_dataset, test_data=test_dataset)
    model.init_model(num_class=num_class, aa_node_in_dim=11, aa_ca_coords=(1000, 3), aa_edge_index=(2,10000),
                     egnn_out_dim=64,ppn_num_heads=10, num_ppn_layers=4, hidden_dim=256, num_layers=5,
                     go_tree_embedding=256)

    checkpoint_dir = os.path.join('./checkpoint_dir/', model.__class__.__name__ + '_cafa3',
                                  f'{args.ont}_checkpoint{log.name}')
    test_result = model.train_and_test(train_data=train_dataset, validation_data=val_dataset, test_data=test_dataset,
                                       batch_size=args.batch_size,
                                       num_epochs=args.num_epochs, num_workers=0, lr=1e-3,
                                       device=args.device,
                                       resume=True,
                                       start_epoch=-1,
                                       checkpoint_dir=checkpoint_dir)

