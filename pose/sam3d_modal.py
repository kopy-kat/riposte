"""Video -> per-frame SAM 3D Body (MHR) reconstruction on Modal.

    modal run pose/sam3d_modal.py --video data/raw/stock/1zDKDT-_p9U.mp4
writes data/processed/<parent dir>_<stem>.sam3d.npz.
"""

import io
from pathlib import Path

import modal

SAM3D_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "libegl1", "build-essential")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "pytorch-lightning", "opencv-python-headless", "yacs", "scikit-image", "einops", "timm", "dill",
        "pandas", "rich", "hydra-core", "pyrootutils", "webdataset", "chump", "networkx==3.2.1", "roma",
        "joblib", "appdirs", "cython", "jsonlines", "xtcocotools", "loguru", "optree", "fvcore",
        "pycocotools", "huggingface_hub", "pyrender", "matplotlib", "omegaconf", "iopath",
    )
    # Modal's Python was built with clang, so extension builds look for clang++; the builder has no GPU,
    # so name the architectures of the GPUs we accept for detectron2's CUDA kernels.
    .env({"CC": "gcc", "CXX": "g++", "FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9"})
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps",
        "pip install git+https://github.com/microsoft/MoGe.git",
        f"git clone https://github.com/facebookresearch/sam-3d-body.git /sam3d && cd /sam3d && git checkout {SAM3D_COMMIT}",
    )
    # setuptools 81 dropped pkg_resources, which detectron2's import chain still uses.
    .pip_install("setuptools<81")
    .env({"HF_HOME": "/cache/hf", "PYOPENGL_PLATFORM": "egl"})
)

app = modal.App("riposte-sam3d", image=image)
cache = modal.Volume.from_name("riposte-cache", create_if_missing=True)


@app.cls(gpu=["L4", "A10G", "A100-40GB"], volumes={"/cache": cache}, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
class Sam3D:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/sam3d")
        from notebook.utils import setup_sam_3d_body

        self.est = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", detector_name="vitdet", fov_name="moge2")

    @modal.method()
    def run(self, video: bytes, stride: int = 1) -> bytes:
        import tempfile

        import cv2
        import numpy as np

        keys = ["bbox", "focal_length", "pred_keypoints_3d", "pred_keypoints_2d", "pred_vertices", "pred_cam_t",
                "global_rot", "body_pose_params", "hand_pose_params", "scale_params", "shape_params",
                "pred_joint_coords", "pred_global_rots", "mhr_model_params"]
        per_frame, cam_int = [], None
        # Frames are processed as they are decoded: a few minutes of 1080p video doesn't fit in memory.
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(video)
            f.flush()
            cap = cv2.VideoCapture(f.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            i = -1
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                i += 1
                if i % stride:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if cam_int is None:
                    # One intrinsics estimate for the whole take: the camera is static, and per-frame
                    # FOV estimates would make the body's depth wobble.
                    cam_int = self.est.fov_estimator.get_cam_intrinsics(rgb)
                outs = self.est.process_one_image(rgb, cam_int=cam_int)
                if not outs:
                    per_frame.append(None)
                    continue
                # The fencer is the largest person in frame; good enough for solo takes.
                o = max(outs, key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
                per_frame.append({k: np.asarray(o[k], dtype=np.float16 if k == "pred_vertices" else np.float32)
                                  for k in keys})

        template = next(p for p in per_frame if p is not None)
        arrays = {
            k: np.stack([p[k] if p is not None else np.full_like(template[k], np.nan, dtype=np.float32)
                         for p in per_frame]).astype(np.float16 if k == "pred_vertices" else np.float32)
            for k in keys
        }
        buf = io.BytesIO()
        np.savez_compressed(buf, fps=fps / stride, cam_int=cam_int.cpu().numpy(), faces=self.est.faces,
                            found=np.array([p is not None for p in per_frame]), **arrays)
        return buf.getvalue()


@app.local_entrypoint()
def main(video: str, stride: int = 1):
    src = Path(video)
    out = Path(__file__).resolve().parents[1] / "data/processed" / f"{src.parent.name}_{src.stem}.sam3d.npz"
    out.write_bytes(Sam3D().run.remote(src.read_bytes(), stride))
    print(out)
