import csv

import numpy as np
import obonet
import pandas as pd
from Bio import SeqIO
from Bio.UniProt.GOA import gafiterator as bio_gafiterator
from tqdm import tqdm
from tools.log import log

GAF22FIELDS = [
    "DB",
    "DB_Object_ID",
    "DB_Object_Symbol",
    "Qualifier",
    "GO_ID",
    "DB:Reference",
    "Evidence",
    "With",
    "Aspect",
    "DB_Object_Name",
    "Synonym",
    "DB_Object_Type",
    "Taxon_ID",
    "Date",
    "Assigned_By",
    "Annotation_Extension",
    "Gene_Product_Form_ID",
]


def read_ppi_info(ppi_path):
    ppi_info = pd.read_csv(ppi_path, sep='\t', header=None, names=['query', 'target', 'combine_score'])
    ppi_dict = {k: list(v['target']) for k, v in ppi_info.groupby('query')}
    return ppi_dict


def read_fasta_file(path):
    return [seq_record for seq_record in SeqIO.parse(path, "fasta")]


def load_go_graph(go_graph_path):
    # read *.obo file
    go_graph = obonet.read_obo(open(go_graph_path, 'r'))
    return go_graph


def write_fasta_file(records, path):
    SeqIO.write(records, path, "fasta")


def load_data_coords(struct_file_path):
    if struct_file_path.endswith('.npz'):
        coords = np.load(struct_file_path, allow_pickle=True)['coords'].item()
        coords = {k: v for k, v in coords.items()}
        return coords
    else:
        raise NotImplementedError('Only the .npz formats are supported')


def print_args_params(params):
    print('------args params--------')
    log.do_info('------args params--------')
    for k, v in vars(params).items():
        print(f'{k}: {v}')
        log.do_info(f'{k}: {v}')
    print('--------------------------')
    log.do_info('--------------------------')


def filter_annotation(annotation_df, ont_, term_):
    annotation_df = annotation_df.copy()
    annotation_df = annotation_df[annotation_df[ont_].notna()]
    annotation_list = [line.split(',') for line in annotation_df[ont_]]
    for i, annotation in enumerate(annotation_list):
        annotation_list[i] = ','.join([a for a in annotation if a in term_])
    annotation_df[ont_] = annotation_list
    annotation_df = annotation_df[annotation_df[ont_] != '']
    return annotation_df


def pd_read_tsv(path):
    return pd.read_csv(path, sep='\t', index_col=0)


def map_annotation(path, type_='all'):
    all_embed, mf_embed, bp_embed, cc_embed = create_annotation_embed(path)
    if type_ == 'all':
        return all_embed
    elif type_ == 'mf':
        return mf_embed
    elif type_ == 'bp':
        return bp_embed
    elif type_ == 'cc':
        return cc_embed
    else:
        raise ValueError('annotation_type can only by all;mf;bp;cc')


def gafiterator(handle):
    """Iterate over a GAF 1.0 or 2.0 or 2.2 file.

    This function should be called to read a
    gene_association.goa_uniprot file. Reads the first record and
    returns a gaf 2.0 or a gaf 1.0 iterator as needed

    Example: open, read, interat and filter results.

    Original data file has been trimed to ~600 rows.

    Original source ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/YEAST/goa_yeast.gaf.gz
    Putative uncharacterized protein YAL019W-A
    ND
    ['YA19A_YEAST', 'YAL019W-A']
    ['taxon:559292']
    Putative uncharacterized protein YAL019W-A
    ND
    ['YA19A_YEAST', 'YAL019W-A']
    ['taxon:559292']
    Putative uncharacterized protein YAL019W-A
    ND
    ['YA19A_YEAST', 'YAL019W-A']
    ['taxon:559292']

    """

    inline = handle.readline()
    if inline.strip() == "!gaf-version: 2.2":
        def _gaf22iterator(handle):
            for inline1 in handle:
                if inline1[0] == "!":
                    continue
                inrec = inline1.rstrip("\n").split("\t")
                if len(inrec) == 1:
                    continue
                inrec[3] = inrec[3].split("|")  # Qualifier
                inrec[5] = inrec[5].split("|")  # DB:reference(s)
                inrec[7] = inrec[7].split("|")  # With || From
                inrec[10] = inrec[10].split("|")  # Synonym
                inrec[12] = inrec[12].split("|")  # Taxon
                yield dict(zip(GAF22FIELDS, inrec))

        return _gaf22iterator(handle)
    else:
        handle.seek(0, 0)
        return bio_gafiterator(handle)


def create_annotation_embed(path):
    with open(path, 'r', newline='\n') as f:
        reader = csv.reader(f)
        # read all annotation
        next(reader)
        go_all = next(reader)
        # read mf annotation
        next(reader)
        go_mf = next(reader)
        # read bp annotation
        next(reader)
        go_bp = next(reader)
        # read cc annotation
        next(reader)
        go_cc = next(reader)
        go_all_embed = dict(zip(go_all, range(len(go_all))))
        go_mf_embed = dict(zip(go_mf, range(len(go_mf))))
        go_bp_embed = dict(zip(go_bp, range(len(go_bp))))
        go_cc_embed = dict(zip(go_cc, range(len(go_cc))))
        return go_all_embed, go_mf_embed, go_bp_embed, go_cc_embed


def extract_ppi_text(ppi_path, sources):
    """
    Read the information from ppi, find the line corresponding to sources, and find the target corresponding to source    :param ppi_path:
    :param sources:
    :return:
    """
    ppi_texts_ = []
    source_set = set()
    target_set = set()
    for line in tqdm(open(ppi_path, 'r'), desc="extract_ppi_text"):
        # line: 394.NGR_c00010 394.NGR_c00020 522 0 0 0 0 0 0 522
        temp = line.split(" ")
        source_ = temp[0]
        target_ = temp[1]
        if source_ in sources:
            ppi_texts_.append(line)
            target_set.add(target_)
    target_set = target_set.difference(source_set)
    return ppi_texts_, target_set


def ppi_text_to_df(ppi_texts):
    protein1_list = []
    protein2_list = []
    combined_score_list = []
    bar = tqdm(total=len(ppi_texts))
    while ppi_texts:
        line = ppi_texts.pop()
        temp = line.split(" ")
        protein1_list.append(temp[0])
        protein2_list.append(temp[1])
        combined_score_list.append(temp[2].strip())
        bar.update(1)
    df = pd.DataFrame({
        "protein1": protein1_list,
        "protein2": protein2_list,
        "combined_score": combined_score_list,
    })
    return df
