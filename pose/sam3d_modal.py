"""Video -> per-frame SAM 3D Body (MHR) reconstruction on Modal.

    modal run pose/sam3d_modal.py --video data/raw/stock/1zDKDT-_p9U.mp4 [--stride 2]
writes data/processed/<parent dir>_<stem>.sam3d.npz. The video is split into chunks run on parallel GPUs.
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


KEYS = ["bbox", "focal_length", "pred_keypoints_3d", "pred_keypoints_2d", "pred_vertices", "pred_cam_t",
        "global_rot", "body_pose_params", "hand_pose_params", "scale_params", "shape_params",
        "pred_joint_coords", "pred_global_rots", "mhr_model_params"]
CHUNK = 240  # processed frames per GPU call; about 4 minutes each


def frames(video: bytes, start: int, stop: int, stride: int):
    import tempfile

    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(video)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        i = -1
        while True:
            ok, bgr = cap.read()
            if not ok or i + 1 >= stop:
                break
            i += 1
            if i >= start and i % stride == 0:
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.cls(gpu=["L4", "A10G", "A100-40GB"], volumes={"/cache": cache}, secrets=[modal.Secret.from_name("huggingface")],
         timeout=1800, max_containers=10)
class Sam3D:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/sam3d")
        from notebook.utils import setup_sam_3d_body

        self.est = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", detector_name="vitdet", fov_name="moge2")

    @modal.method()
    def intrinsics(self, video: bytes):
        # One intrinsics estimate for the whole take: the camera is static, and per-frame or per-chunk
        # FOV estimates would make the body's depth wobble.
        import tempfile

        import cv2

        rgb = next(frames(video, 0, 1, 1))
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(video)
            f.flush()
            cap = cv2.VideoCapture(f.name)
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        return self.est.fov_estimator.get_cam_intrinsics(rgb).cpu().numpy(), n, fps, self.est.faces

    @modal.method()
    def chunk(self, video: bytes, start: int, stop: int, stride: int, cam_int) -> list:
        import numpy as np
        import torch

        K = torch.as_tensor(cam_int).cuda()
        out = []
        # Detect every frame: reusing a box between detections shifts the crop, and the crop sets depth.
        for rgb in frames(video, start, stop, stride):
            outs = self.est.process_one_image(rgb, cam_int=K)
            if not outs:
                out.append(None)
                continue
            # The fencer is the largest person in frame; good enough for solo takes.
            o = max(outs, key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
            out.append({k: np.asarray(o[k], dtype=np.float16 if k == "pred_vertices" else np.float32) for k in KEYS})
        return out


@app.function(timeout=1800, memory=16384)
def merge(chunks: list, cam_int, faces, fps: float, stride: int) -> bytes:
    import numpy as np

    per_frame = [p for c in chunks for p in c]
    template = next(p for p in per_frame if p is not None)
    arrays = {k: np.stack([p[k] if p is not None else np.full_like(template[k], np.nan) for p in per_frame])
              for k in KEYS}
    buf = io.BytesIO()
    np.savez_compressed(buf, fps=fps / stride, cam_int=cam_int, faces=faces,
                        found=np.array([p is not None for p in per_frame]), **arrays)
    return buf.getvalue()


@app.local_entrypoint()
def main(video: str, stride: int = 1):
    src = Path(video)
    out = Path(__file__).resolve().parents[1] / "data/processed" / f"{src.parent.name}_{src.stem}.sam3d.npz"
    data = src.read_bytes()
    cam_int, n, fps, faces = Sam3D().intrinsics.remote(data)
    span = CHUNK * stride
    bounds = [(a, min(a + span, n)) for a in range(0, n, span)]
    print(f"{n} frames, {len(bounds)} chunks")
    chunks = list(Sam3D().chunk.starmap([(data, a, b, stride, cam_int) for a, b in bounds]))
    out.write_bytes(merge.remote(chunks, cam_int, faces, fps, stride))
    print(out)
