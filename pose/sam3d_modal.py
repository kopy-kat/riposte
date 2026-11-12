"""Video -> per-frame SAM 3D Body (MHR) reconstruction on Modal.

    modal run pose/sam3d_modal.py --video data/raw/stock/1zDKDT-_p9U.mp4 [--stride 2]
writes data/processed/<parent dir>_<stem>.sam3d.npz. The video is split into chunks run on parallel GPUs.
"""

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
    .env({"CC": "gcc", "CXX": "g++", "FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0"})
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


def frames(video: str, start: int, stop: int, stride: int):
    import cv2

    # Decode from the start: CAP_PROP_POS_FRAMES seeks are not frame-accurate on H.264.
    cap = cv2.VideoCapture(video)
    for i in range(stop):
        ok, bgr = cap.read()
        if not ok:
            break
        if i >= start and i % stride == 0:
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.cls(gpu=["L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "H100"], volumes={"/cache": cache}, secrets=[modal.Secret.from_name("huggingface")],
         timeout=1800, max_containers=10)
class Sam3D:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, "/sam3d")
        from notebook.utils import setup_sam_3d_body

        self.est = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", detector_name="vitdet", fov_name="moge2")

    @modal.method()
    def intrinsics(self, video: str):
        # One intrinsics estimate for the whole take: the camera is static, and per-frame or per-chunk
        # FOV estimates would make the body's depth wobble.
        import cv2

        cache.reload()
        rgb = next(frames(video, 0, 1, 1))
        cap = cv2.VideoCapture(video)
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        return self.est.fov_estimator.get_cam_intrinsics(rgb).cpu().numpy(), n, fps, self.est.faces

    @modal.method()
    def chunk(self, video: str, start: int, stop: int, stride: int, cam_int, part: str):
        import pickle

        import numpy as np
        import torch

        cache.reload()
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
        Path(part).parent.mkdir(parents=True, exist_ok=True)
        Path(part).write_bytes(pickle.dumps(out))
        cache.commit()


@app.function(timeout=1800, memory=16384, volumes={"/cache": cache})
def merge(parts: list, cam_int, faces, fps: float, stride: int, dest: str):
    import pickle

    import numpy as np

    cache.reload()
    per_frame = [p for part in parts for p in pickle.loads(Path(part).read_bytes())]
    template = next(p for p in per_frame if p is not None)
    arrays = {k: np.stack([p[k] if p is not None else np.full_like(template[k], np.nan) for p in per_frame])
              for k in KEYS}
    np.savez_compressed(dest, fps=fps / stride, cam_int=cam_int, faces=faces,
                        found=np.array([p is not None for p in per_frame]), **arrays)
    cache.commit()


@app.local_entrypoint()
def main(video: str, stride: int = 1):
    src = Path(video)
    out = Path(__file__).resolve().parents[1] / "data/processed" / f"{src.parent.name}_{src.stem}.sam3d.npz"
    # Upload once to the volume; passing the bytes to every chunk call re-uploads the whole video per chunk.
    remote = f"/videos/{src.parent.name}_{src.name}"
    with cache.batch_upload(force=True) as up:
        up.put_file(str(src), remote)
    data = "/cache" + remote
    cam_int, n, fps, faces = Sam3D().intrinsics.remote(data)
    span = CHUNK * stride
    bounds = [(a, min(a + span, n)) for a in range(0, n, span)]
    print(f"{n} frames, {len(bounds)} chunks")
    # Chunk results stay on the volume: sending them through this client and back up to merge stalls on
    # a home connection.
    stem = f"/out/{src.parent.name}_{src.stem}"
    parts = [f"/cache{stem}/{i:04d}.pkl" for i in range(len(bounds))]
    list(Sam3D().chunk.starmap([(data, a, b, stride, cam_int, part) for (a, b), part in zip(bounds, parts)]))
    merge.remote(parts, cam_int, faces, fps, stride, f"/cache{stem}.npz")
    with out.open("wb") as f:
        for b in cache.read_file(f"{stem}.npz"):
            f.write(b)
    print(out)
