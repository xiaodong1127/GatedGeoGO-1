import os
import pathlib

import torch

from esm import FastaBatchedDataset, pretrained, MSATransformer


class EsmArgs:
    def __init__(self, fasta_file, output_dir, device):
        self.model_location = 'esm1b_t33_650M_UR50S'
        self.fasta_file = pathlib.Path(fasta_file)
        self.output_dir = pathlib.Path(output_dir)
        self.toks_per_batch = 4096
        self.repr_layers = [33]
        self.include = {'mean'}
        self.truncate = True
        self.skip = True
        self.nogpu = False
        self.device = device

    def __contains__(self, key):
        return str(key) in {'model_location', 'fasta_file', 'output_dir', 'toks_per_batch', 'repr_layers', 'include',
                            'truncate',
                            'skip', 'nogpu', 'device'}


def extract_esm_mean(fasta_file, output_dir, device):
    esm_args = EsmArgs(fasta_file, output_dir, device)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    extract_esm(esm_args)


def extract_esm_features_in_memory(sequence_labels, sequence_strs,
                                   device='cpu',
                                   model_path='esm1b_t33_650M_UR50S',
                                   ):
    include = {'mean'}
    repr_layers = [33]
    truncate = True
    model, alphabet = pretrained.load_model_and_alphabet(model_path)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available() and device != 'cpu':
        model = model.to(device)
        print("Transferred model to GPU")

    assert len(set(sequence_labels)) == len(
        sequence_labels
    ), "Found duplicate sequence labels"

    dataset = FastaBatchedDataset(sequence_labels, sequence_strs)
    print(f"Calculate the ESM features of {len(dataset)} sequences")

    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )

    return_contacts = "contacts" in include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    r = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and device != 'cpu':
                toks = toks.to(device=device, non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if truncate:
                toks = toks[:, :1022]

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                result = {"label": label}
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in include:
                    result["representations"] = {
                        layer: t[i, 1: len(strs[i]) + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in include:
                    result["mean_representations"] = {
                        layer: t[i, 1: len(strs[i]) + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone()
                r[label] = result
    return r


def extract_esm(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available() and not args.nogpu:
        if 'device' not in args:
            model = model.cuda()
        else:
            model = model.to(args.device)
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    if args.skip:
        existed_files = {file.split('.pt')[0] for file in os.listdir(args.output_dir)}
        sequence_labels = []
        sequence_strs = []
        for i in range(len(dataset)):
            if dataset.sequence_labels[i] not in existed_files:
                sequence_labels.append(dataset.sequence_labels[i])
                sequence_strs.append(dataset.sequence_strs[i])
        dataset.sequence_labels = sequence_labels
        dataset.sequence_strs = sequence_strs
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                if 'device' not in args:
                    toks = toks.to(device="cuda", non_blocking=True)
                else:
                    toks = toks.to(device=args.device, non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if args.truncate:
                toks = toks[:, :1022]

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1: len(strs[i]) + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = { 
                        layer: t[i, 1: len(strs[i]) + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone()

                torch.save(
                    result,
                    args.output_file,
                )
