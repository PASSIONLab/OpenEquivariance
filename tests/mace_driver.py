import sys, json, time, pathlib

import argparse
import logging
from pathlib import Path

from ase import Atoms
import ase.io
import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric
from torch.utils.benchmark import Timer
from mace.calculators import mace_mp
from torch.profiler import profile, record_function, ProfilerActivity

import warnings
warnings.filterwarnings("ignore")

try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

def analyze_trace(trace_file):
    trace = None
    with open(trace_file, "r") as f:
        trace = json.load(f)

    total = 0
    cgtp_fwd_bwd = 0
    reduce_by_key = 0
    other_kernels = 0

    for event in trace["traceEvents"]:
        if "args" in event and "stream" in event["args"]:
            total += event["dur"]

            if "forward" in event["name"] \
                or "backward" in event["name"] \
                or "TensorProductUniform1dKernel" in event["name"]:
                cgtp_fwd_bwd += event["dur"]

            elif "_scatter_gather_elementwise_kernel" in event["name"]:
                reduce_by_key += event["dur"]
            else:
                other_kernels += event["dur"]

    return { 
        "total_cuda_ms": total / 1000.,
        "cgtp_fwd_bwd_ms": cgtp_fwd_bwd / 1000.,
        "reduce_by_key_ms": reduce_by_key / 1000.,
        "other_kernels_ms": other_kernels / 1000.
    }


def benchmark_model(model, batch, num_iterations=100, warmup=100, label=None, output_folder=None):
    def run_inference():
        out = model(batch,training=True)
        torch.cuda.synchronize()
        return out

    # Warmup
    for _ in range(warmup):
        run_inference()

    # Benchmark
    timer = Timer(
        stmt="run_inference()",
        globals={
            "run_inference": run_inference,
        },
    )
    warm_up_measurement = timer.timeit(num_iterations)
    measurement = timer.timeit(num_iterations)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            run_inference() 

    trace_file = str(output_folder / f"traces/{label}_trace.json")
    prof.export_chrome_trace(trace_file)

    with open(output_folder / f"{label}.json", "w") as f:
        json.dump({
            "time_ms_mean": measurement.mean * 1000, 
            "label": label,
            "cuda_time_profile": analyze_trace(trace_file)
        }, f, indent=4) 

    return measurement

def get_e3nn_model(source_model, device):
    from mace.tools.scripts_utils import extract_config_mace_model
    config = extract_config_mace_model(source_model)
    config["oeq_config"] = {"enabled": False}
    target_model = source_model.__class__(**config).to(device)

    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    # To migrate openequivariance, we should transfer all keys
    for key in target_dict:
        if key in source_dict:
            target_dict[key] = source_dict[key]

    target_model.load_state_dict(target_dict)

    return target_model.to(device)

def get_oeq_model(source_model, device):
    from mace.tools.scripts_utils import extract_config_mace_model
    config = extract_config_mace_model(source_model)
    config["oeq_config"] = {"enabled": True, "conv_fusion": "deterministic"}
    target_model = source_model.__class__(**config).to(device)

    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    # To migrate openequivariance, we should transfer all keys
    for key in target_dict:
        if key in source_dict:
            target_dict[key] = source_dict[key]

    target_model.load_state_dict(target_dict)
    return target_model.to(device)


def get_cueq_model(source_model, device):
    model_cueq = run_e3nn_to_cueq(source_model)
    return model_cueq.to(device)

def get_implementation_model(source_model, implementation, device):
    if implementation == 'oeq':
        return get_oeq_model(source_model, device)
    if implementation == 'cue':
        return get_cueq_model(source_model, device)
    if implementation == 'e3nn':
        return get_e3nn_model(source_model, device)
    else: 
        raise ValueError("not a valid model implementation")

def main():
    print("WARNING: You need a modified version of MACE to run this driver.")
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_file", type=str, help="Path to xyz file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--max_ell", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_folder", '-o', type=str, required=True)
    parser.add_argument("--implementations", "-i", type=str, nargs='+', 
            default=['e3nn', 'cue', 'oeq'], help="Implementations to benchmark",
            choices=['e3nn', 'cue', 'oeq'])

    args = parser.parse_args()

    output_folder = args.output_folder
    output_folder = pathlib.Path(output_folder)
    
    for dtype_str, dtype, mace_dtype_str in [   
            ("f32", torch.float32, "float32"),
            ("f64", torch.float64, "float64"),
                            ]:
        torch.set_default_dtype(dtype)
        device = torch.device(args.device)

        for model_str in ['small', 'medium', 'large']:
            source_model = mace_mp(model=model_str, default_dtype=mace_dtype_str, return_raw_model=True)
            source_model.to(device=device, dtype=dtype)

            # # Create dataset
            atoms_list = ase.io.read(args.xyz_file, index=":")
            assert isinstance(atoms_list, list)

            z_table = tools.utils.AtomicNumberTable([int(z) for z in source_model.atomic_numbers])

            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=[data.AtomicData.from_config(
                    data.config_from_atoms(atoms),
                    z_table=z_table,
                    cutoff=6.0
                ) for atoms in atoms_list],
                batch_size=min(len(atoms_list), args.batch_size),
                shuffle=False,
                drop_last=False,
            )
            batch = next(iter(data_loader)).to(device)
            batch_dict = batch.to_dict()
            for k,v in batch_dict.items():
                if k not in ["index", "head", "node_attrs", "batch", "edge_index"]:
                    batch_dict[k] = v.to(dtype=dtype)
            
            output_folder.mkdir(parents=True, exist_ok=True)

            traces_folder = output_folder / "traces"
            traces_folder.mkdir(parents=True, exist_ok=True) 

            # Compile is still not working for MACE and cueq; turned off for now
            print("\nBenchmarking Configuration:")
            print(f"Number of atoms: {len(atoms_list[0])}")
            print(f"Model Size: {model_str}")
            print(f"Number of edges: {batch['edge_index'].shape[1]}")
            print(f"Batch size: {min(len(atoms_list), args.batch_size)}")
            print(f"Device: {args.device}")
            print(f"Number of iterations: {args.num_iters}\n")
        
            for implementation in args.implementations:
                model = get_implementation_model(source_model, implementation, device)
                measurement = benchmark_model(model, batch_dict, args.num_iters, label=f"{implementation}_{model_str}_{dtype_str}", output_folder=output_folder)
                print(f"{implementation} Measurement:\n{measurement}")


if __name__ == "__main__":
    main()