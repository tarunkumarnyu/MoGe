#!/usr/bin/env python3
"""
MoGe + Intel RealSense D455 live depth estimation pipeline (optimized).
Threaded capture + async TRT inference for maximum FPS.

Usage:
    python d455_moge_live.py                              # D455 camera
    python d455_moge_live.py --webcam                     # Fallback to webcam
    python d455_moge_live.py --trt moge_v2_fp16.trt       # TensorRT mode
    python d455_moge_live.py --trt moge_v2_fp16.trt --compare  # Side-by-side with D455 depth
"""

import argparse
import sys
import time
import threading

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, ".")


class ThreadedD455:
    """Non-blocking D455 capture in a background thread."""

    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        self.bgr = None
        self.d455_depth = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())
            d455_depth = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale if depth else None
            with self.lock:
                self.bgr = bgr
                self.d455_depth = d455_depth

    def read(self):
        with self.lock:
            return self.bgr, self.d455_depth

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.pipeline.stop()


class ThreadedWebcam:
    """Non-blocking webcam capture in a background thread."""

    def __init__(self, width=640, height=480):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.bgr = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.bgr = cv2.resize(frame, (640, 480))

    def read(self):
        with self.lock:
            return self.bgr, None

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


def load_moge_model(device="cuda"):
    """Load MoGe v2 model."""
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl")
    model = model.to(device).eval()
    model.enable_pytorch_native_sdpa()
    return model


class TRTInferencer:
    """Pre-allocated TensorRT inference context with proper MoGe post-processing."""

    def __init__(self, engine_path, fov_x=None):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.fov_x = fov_x  # D455 horizontal FOV ~87 degrees

        # Pre-allocate buffers
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.array([], dtype=dtype)).dtype
            self.buffers[name] = torch.empty(list(shape), dtype=torch_dtype, device="cuda")
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())

        # Import post-processing utilities
        from moge.utils.geometry_torch import recover_focal_shift

        self._recover_focal_shift = recover_focal_shift

        print(f"TRT buffers allocated: {list(self.buffers.keys())}")

    def __call__(self, bgr_np):
        """Run inference on a BGR numpy image (H, W, 3). Returns (depth, mask) on CPU."""
        h, w = bgr_np.shape[:2]
        aspect_ratio = w / h

        # Preprocess: BGR->RGB, float32, NCHW
        rgb = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)[np.newaxis]).cuda()

        # Copy and execute
        self.buffers["image"].copy_(input_tensor)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # Get raw outputs from TRT
        points = self.buffers["points"].float()        # (1, H, W, 3)
        mask_raw = self.buffers["mask"].float()         # (1, H, W)
        metric_scale = self.buffers["metric_scale"].float()  # (1,)

        # --- Post-processing (same as MoGeModel.infer) ---
        mask_binary = mask_raw > 0.5

        # Recover focal length and depth shift from affine point map
        if self.fov_x is not None:
            focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(
                torch.deg2rad(torch.tensor(self.fov_x, device="cuda", dtype=torch.float32) / 2)
            )
            focal = focal[None]
            _, shift = self._recover_focal_shift(points, mask_binary, focal=focal)
        else:
            focal, shift = self._recover_focal_shift(points, mask_binary)

        # Apply shift to get proper depth
        depth = points[..., 2] + shift[..., None, None]

        # Mask out invalid pixels
        mask_binary = mask_binary & (depth > 0)

        # Apply metric scale
        depth = depth * metric_scale[:, None, None]

        depth = torch.where(mask_binary, depth, torch.tensor(torch.inf, device="cuda"))

        return depth[0].cpu().numpy(), mask_binary[0].cpu().numpy()


def colorize_depth(depth, mask=None, vmin=None, vmax=None):
    """Convert depth map to colorized visualization."""
    if mask is not None:
        valid = depth[mask]
    else:
        valid = depth[depth < np.inf]

    if len(valid) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if vmin is None:
        vmin = np.percentile(valid, 2)
    if vmax is None:
        vmax = np.percentile(valid, 98)

    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

    if mask is not None:
        colored[~mask] = 0

    return colored


def run_moge_pytorch(model, rgb_tensor, resolution_level):
    """Run MoGe inference with PyTorch."""
    output = model.infer(rgb_tensor, resolution_level=resolution_level)
    depth = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()
    return depth, mask


def main():
    parser = argparse.ArgumentParser(description="MoGe + D455 Live Depth (Optimized)")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of D455")
    parser.add_argument("--resolution", type=int, default=7, help="MoGe resolution level 0-9 (default: 7)")
    parser.add_argument("--trt", type=str, default=None, help="Path to TensorRT engine file")
    parser.add_argument("--compare", action="store_true", help="Show D455 depth alongside MoGe depth")
    parser.add_argument("--fov", type=float, default=None, help="Camera horizontal FOV in degrees (D455=87)")
    parser.add_argument("--save-video", type=str, default=None, help="Save output to video file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if args.trt:
        print(f"Loading TensorRT engine: {args.trt}")
        trt_infer = TRTInferencer(args.trt, fov_x=args.fov)
        use_trt = True
    else:
        print("Loading MoGe v2 model...")
        model = load_moge_model(device)
        use_trt = False

    # Init threaded camera
    use_d455 = not args.webcam
    if use_d455:
        try:
            cam = ThreadedD455()
            print(f"D455 connected (depth scale: {cam.depth_scale:.6f})")
        except Exception as e:
            print(f"D455 not available ({e}), falling back to webcam")
            use_d455 = False

    if not use_d455:
        cam = ThreadedWebcam()
        print("Using webcam (threaded)")

    # Wait for first frame
    while cam.read()[0] is None:
        time.sleep(0.01)

    # Video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = 640 * 3 if (use_d455 and args.compare) else 640 * 2
        writer = cv2.VideoWriter(args.save_video, fourcc, 30, (out_w, 480))

    print("Running... Press 'q' to quit, 's' to save frame")
    frame_times = []

    try:
        while True:
            # Grab latest frame (non-blocking)
            bgr, d455_depth = cam.read()
            if bgr is None:
                continue

            # Run MoGe
            t0 = time.time()
            if use_trt:
                moge_depth, moge_mask = trt_infer(bgr)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_tensor = TF.to_tensor(rgb).to(device)
                moge_depth, moge_mask = run_moge_pytorch(model, rgb_tensor, args.resolution)

            dt = time.time() - t0
            frame_times.append(dt)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))

            # Visualize
            moge_vis = colorize_depth(moge_depth, moge_mask)
            moge_vis = cv2.resize(moge_vis, (640, 480))

            cv2.putText(moge_vis, f"MoGe {avg_fps:.1f} FPS ({dt*1000:.0f}ms)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            panels = [bgr, moge_vis]

            if d455_depth is not None and args.compare:
                d455_vis = colorize_depth(d455_depth, d455_depth > 0)
                d455_vis = cv2.resize(d455_vis, (640, 480))
                cv2.putText(d455_vis, "D455 Stereo Depth",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(d455_vis)

            display = np.hstack(panels)
            cv2.imshow("MoGe + D455", display)

            if writer:
                writer.write(display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"frame_{int(time.time())}.png"
                cv2.imwrite(fname, display)
                print(f"Saved {fname}")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if writer:
            writer.release()
        if frame_times:
            print(f"Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    main()
