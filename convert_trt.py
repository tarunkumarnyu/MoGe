#!/usr/bin/env python3
"""
Convert MoGe ONNX model to TensorRT engine with FP16 optimization.

Usage:
    python convert_trt.py                                          # Default
    python convert_trt.py --onnx moge_v2.onnx --output moge_v2.trt --fp16
    python convert_trt.py --workspace 8                            # 8 GB workspace
"""

import argparse
import os
import time

import numpy as np
import tensorrt as trt


def build_engine(onnx_path, output_path, fp16=True, workspace_gb=6):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Set workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Enable FP16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")
    elif fp16:
        print("WARNING: FP16 not supported on this platform, using FP32")

    # Parse ONNX
    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    # Print network info
    print(f"Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: {inp.shape} ({inp.dtype})")
    print(f"Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: {out.shape} ({out.dtype})")

    # Build engine
    print(f"Building TensorRT engine (this may take several minutes)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    t1 = time.time()

    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    print(f"Engine built in {t1-t0:.1f}s")

    # Save engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    return serialized_engine


def benchmark_engine(engine_path, num_warmup=5, num_runs=50):
    """Benchmark TRT engine inference speed."""
    import torch

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate I/O buffers
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        tensor = torch.empty(list(shape), dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype, device="cuda")
        buffers[name] = tensor
        context.set_tensor_address(name, tensor.data_ptr())

    # Fill input with random data
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            buffers[name].normal_()

    stream = torch.cuda.current_stream().cuda_stream

    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000)

    avg = np.mean(times)
    std = np.std(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    fps = 1000.0 / avg

    print(f"\nResults:")
    print(f"  Mean: {avg:.1f} ms")
    print(f"  Std:  {std:.1f} ms")
    print(f"  P50:  {p50:.1f} ms")
    print(f"  P95:  {p95:.1f} ms")
    print(f"  FPS:  {fps:.1f}")

    # Print output shapes
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            print(f"  {name}: {list(buffers[name].shape)}")


def main():
    parser = argparse.ArgumentParser(description="Convert MoGe ONNX to TensorRT")
    parser.add_argument("--onnx", type=str, default="moge_v2.onnx", help="Input ONNX file")
    parser.add_argument("--output", type=str, default="moge_v2_fp16.trt", help="Output TRT engine")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 (default)")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument("--workspace", type=int, default=6, help="Workspace size in GB")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Run benchmark after build")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmark")
    args = parser.parse_args()

    use_fp16 = not args.fp32

    build_engine(args.onnx, args.output, fp16=use_fp16, workspace_gb=args.workspace)

    if args.benchmark and not args.no_benchmark:
        print("\n--- Benchmark ---")
        benchmark_engine(args.output)


if __name__ == "__main__":
    main()
