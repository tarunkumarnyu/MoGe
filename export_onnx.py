#!/usr/bin/env python3
"""
Export MoGe v2 model to ONNX format.

Usage:
    python export_onnx.py                           # Default export (640x480, fp16)
    python export_onnx.py --height 480 --width 640  # Custom resolution
    python export_onnx.py --opset 17                # Custom opset version
"""

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")


class MoGeONNXWrapper(nn.Module):
    """Wrapper that makes MoGe's forward pass ONNX-exportable.

    Fixes the dynamic num_tokens to a static value and returns
    (points, mask) tensors directly instead of a dict.
    """

    def __init__(self, model, num_tokens):
        super().__init__()
        self.model = model
        self.num_tokens = num_tokens

    def forward(self, image):
        output = self.model(image, num_tokens=self.num_tokens)
        points = output["points"]  # (B, H, W, 3)
        mask = output.get("mask", torch.ones(image.shape[0], image.shape[2], image.shape[3], device=image.device))
        normal = output.get("normal", torch.zeros_like(points))
        metric_scale = output.get("metric_scale", torch.ones(image.shape[0], device=image.device))
        return points, mask, normal, metric_scale


def main():
    parser = argparse.ArgumentParser(description="Export MoGe to ONNX")
    parser.add_argument("--height", type=int, default=480, help="Input height")
    parser.add_argument("--width", type=int, default=640, help="Input width")
    parser.add_argument("--num-tokens", type=int, default=1800, help="Number of ViT tokens (1200-3600)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--output", type=str, default="moge_v2.onnx", help="Output ONNX file")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")
    args = parser.parse_args()

    device = "cuda"

    # Load model
    from moge.model.v2 import MoGeModel

    print("Loading MoGe v2 model...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl")
    model = model.to(device).eval()
    model.onnx_compatible_mode = True
    model.enable_pytorch_native_sdpa()

    # Create wrapper with fixed num_tokens
    wrapper = MoGeONNXWrapper(model, num_tokens=args.num_tokens).eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, args.height, args.width, device=device)
    if args.fp16:
        wrapper = wrapper.half()
        dummy_input = dummy_input.half()

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        points, mask, normal, metric_scale = wrapper(dummy_input)
        print(f"  points: {points.shape}, mask: {mask.shape}, normal: {normal.shape}, metric_scale: {metric_scale.shape} ({metric_scale.item():.4f})")

    # Export to ONNX
    print(f"Exporting to ONNX (opset {args.opset})...")
    t0 = time.time()

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        args.output,
        input_names=["image"],
        output_names=["points", "mask", "normal", "metric_scale"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,  # Fixed shape for TRT optimization
        dynamo=False,  # Use legacy exporter to capture all weights
    )

    t1 = time.time()
    print(f"ONNX export completed in {t1-t0:.1f}s -> {args.output}")

    # Verify ONNX model
    import onnx

    print("Verifying ONNX model...")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified!")

    # Print model info
    import os

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    # Simplify if requested
    if args.simplify:
        try:
            import onnxsim

            print("Simplifying ONNX graph...")
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, args.output)
                new_size = os.path.getsize(args.output) / (1024 * 1024)
                print(f"Simplified: {size_mb:.1f} MB -> {new_size:.1f} MB")
            else:
                print("Simplification failed, keeping original")
        except ImportError:
            print("onnxsim not installed, skipping simplification")

    print("Done!")


if __name__ == "__main__":
    main()
