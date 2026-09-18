
from dataclasses import dataclass
from pathlib import Path
import re
import time

import cv2
import faiss
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform as crs_transform

from src.loftr import LoFTR
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config


@dataclass
class AVLConfig:
    parent_cache: Path
    sub_cache: Path
    sub_tiles_dir: Path

    mode: str = "two"
    topk_stage1: int = 5
    topk_stage2: int = 5

    img_size: tuple = (320, 320)
    target_size: int = 320

    litesam_cfg: str = r"configs/loftr/eloftr_full.py"
    litesam_weights: str = r"weights/mloftr.ckpt"
    loftr_thr: float = 0.2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def coarse_match(uav_img_bgr, index, tile_names, extract_global_desc_fn, topk=5):
    q = extract_global_desc_fn(uav_img_bgr)[None, :]
    k = min(topk, len(tile_names))
    scores, idxs = index.search(q, k=k)
    return [(tile_names[idx], float(score)) for idx, score in zip(idxs[0], scores[0])]


def coarse_match_subtiles(
    uav_img_bgr,
    sub_index,
    sub_names,
    selected_parent_ids,
    extract_global_desc_fn,
    topk=5,
):
    candidate_indices = []
    for i, sub_name in enumerate(sub_names):
        for pid in selected_parent_ids:
            if f"_{pid}_" in str(sub_name):
                candidate_indices.append(i)
                break

    if not candidate_indices:
        raise RuntimeError("No sub-tiles matched selected parent IDs.")

    sub_descs = np.zeros((len(candidate_indices), 2048), dtype=np.float32)
    for j, idx in enumerate(candidate_indices):
        sub_descs[j] = sub_index.reconstruct(idx)

    temp_index = faiss.IndexFlatIP(2048)
    temp_index.add(sub_descs)

    q = extract_global_desc_fn(uav_img_bgr)[None, :]
    k = min(topk, len(candidate_indices))
    scores, idxs = temp_index.search(q, k=k)

    results = []
    for local_idx, score in zip(idxs[0], scores[0]):
        global_idx = candidate_indices[int(local_idx)]
        results.append((sub_names[global_idx], float(score)))
    return results


def fine_match_litesam(
    model,
    uav_tensor,
    candidate_tile_names,
    sub_tiles_dir,
    load_gray_tensor_fn,
    device="cpu",
    verbose=False,
):
    fine_results = []

    for tile_name in candidate_tile_names:
        tile_path = Path(sub_tiles_dir) / str(tile_name)
        tile_tensor = load_gray_tensor_fn(tile_path).to(device)

        data = {"image0": tile_tensor, "image1": uav_tensor}
        t0 = time.time()
        with torch.no_grad():
            model(data)
        if verbose:
            print(f"[LiteSAM] {Path(tile_name).name}  t={time.time() - t0:.3f}s")

        if "mconf" in data and data["mconf"].numel() > 0:
            num_matches = int(data["mconf"].numel())
            mean_conf = float(data["mconf"].mean().item())
        else:
            num_matches, mean_conf = 0, 0.0

        fine_results.append((str(tile_name), num_matches, mean_conf))

    fine_results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_tile = fine_results[0][0] if fine_results else None
    return fine_results, best_tile


def estimate_geolocation(
    k0,
    k1,
    best_tile_path,
    target_size,
    conf=None,
    conf_thr=0.3,
    max_points=300,
    ransac_thresh=5.0,
    verbose=False,
):
    """Project the UAV-image center into a georeferenced map tile."""
    if len(k0) < 4:
        return None

    k0_f, k1_f = k0, k1
    if conf is not None and len(conf) == len(k0):
        keep = conf >= conf_thr
        k0_f = k0[keep]
        k1_f = k1[keep]
        conf_f = conf[keep]

        if len(conf_f) > max_points:
            idx = np.argsort(-conf_f)[:max_points]
            k0_f = k0_f[idx]
            k1_f = k1_f[idx]

    if len(k0_f) < 4:
        return None

    # k1 is the UAV image and k0 is the map tile, so H maps UAV -> tile.
    H, mask = cv2.findHomography(k1_f, k0_f, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None

    inliers = int(mask.ravel().sum())
    total = int(mask.size)
    inlier_ratio = inliers / total if total > 0 else 0.0

    cx = target_size / 2
    cy = target_size / 2
    uav_center = np.array([[[cx, cy]]], dtype=np.float32)
    tile_point = cv2.perspectiveTransform(uav_center, H)
    tx_resized, ty_resized = tile_point[0][0]

    with rasterio.open(best_tile_path) as src:
        scale_x = src.width / float(target_size)
        scale_y = src.height / float(target_size)
        tx = tx_resized * scale_x
        ty = ty_resized * scale_y

        x_crs, y_crs = xy(src.transform, ty, tx, offset="center")
        lon, lat = crs_transform(src.crs, "EPSG:4326", [x_crs], [y_crs])
        lat, lon = float(lat[0]), float(lon[0])

        if verbose:
            print(
                f"[GEO] CRS={src.crs} x={x_crs:.2f} y={y_crs:.2f} "
                f"lat={lat:.7f} lon={lon:.7f} inlier={inlier_ratio:.2f}"
            )

    return {
        "lat": lat,
        "lon": lon,
        "x_crs": float(x_crs),
        "y_crs": float(y_crs),
        "inlier_ratio": float(inlier_ratio),
        "H": H,
    }


class AVLLocalizer:
    """Hierarchical satellite-map localizer used by the final project."""

    def __init__(self, cfg: AVLConfig):
        self.cfg = cfg
        self.device = cfg.device

        self.parent_index = faiss.read_index(str(cfg.parent_cache / "tiles.index"))
        self.parent_names = np.load(
            str(cfg.parent_cache / "tile_names.npy"), allow_pickle=True
        )

        self.sub_index = faiss.read_index(str(cfg.sub_cache / "tiles.index"))
        self.sub_names = np.load(
            str(cfg.sub_cache / "tile_names.npy"), allow_pickle=True
        )

        self._base = models.resnet50(weights="IMAGENET1K_V2")
        self._base = torch.nn.Sequential(*list(self._base.children())[:-2]).to(
            self.device
        ).eval()

        class GeM(torch.nn.Module):
            def __init__(self, p=3.0):
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(1) * p)

            def forward(self, x):
                return torch.pow(
                    torch.mean(torch.pow(x, self.p), dim=(2, 3)),
                    1.0 / self.p,
                )

        self._gem = GeM().to(self.device)
        self._transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(cfg.img_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.litesam = self._build_litesam()

    @torch.no_grad()
    def extract_global_desc(self, img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self._transform(img_rgb).unsqueeze(0).to(self.device)
        feat = self._base(x)
        feat = self._gem(feat)
        vector = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-8
        return vector

    def load_gray_tensor(self, path: Path) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.resize(img, (self.cfg.target_size, self.cfg.target_size))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)[None, None]

    def _build_litesam(self):
        cfg = get_cfg_defaults()
        cfg.merge_from_file(self.cfg.litesam_cfg)
        cfg.LOFTR.COARSE.NPE = [832, 832, 1184, 1184]
        cfg.LOFTR.MATCH_COARSE.THR = self.cfg.loftr_thr

        model_config = lower_config(cfg)
        model = LoFTR(config=model_config["loftr"])

        state = torch.load(
            self.cfg.litesam_weights,
            map_location=self.device,
            weights_only=False,
        )["state_dict"]
        model.load_state_dict(state, strict=False)
        return model.to(self.device).eval()

    def localize(self, uav_image_path: str, verbose=False):
        """Return latitude/longitude and matching metadata, or ``None``."""
        uav_img = cv2.imread(str(uav_image_path), cv2.IMREAD_COLOR)
        if uav_img is None:
            raise FileNotFoundError(uav_image_path)

        stage1 = coarse_match(
            uav_img,
            self.parent_index,
            self.parent_names,
            self.extract_global_desc,
            topk=self.cfg.topk_stage1,
        )

        if self.cfg.mode == "single":
            candidates = [name for name, _ in stage1]
        elif self.cfg.mode == "two":
            parent_ids = []
            for name, _ in stage1:
                match = re.search(r"_(\d{6})_", str(name))
                if match:
                    parent_ids.append(match.group(1))

            stage2 = coarse_match_subtiles(
                uav_img,
                self.sub_index,
                self.sub_names,
                parent_ids,
                self.extract_global_desc,
                topk=self.cfg.topk_stage2,
            )
            candidates = [name for name, _ in stage2]
        else:
            raise ValueError("cfg.mode must be 'single' or 'two'")

        uav_tensor = self.load_gray_tensor(Path(uav_image_path)).to(self.device)
        fine_results, best_tile = fine_match_litesam(
            self.litesam,
            uav_tensor,
            candidates,
            self.cfg.sub_tiles_dir,
            self.load_gray_tensor,
            device=self.device,
            verbose=verbose,
        )
        if best_tile is None:
            return None

        best_tile_path = self.cfg.sub_tiles_dir / best_tile

        tile_tensor = self.load_gray_tensor(best_tile_path).to(self.device)
        data = {"image0": tile_tensor, "image1": uav_tensor}
        with torch.no_grad():
            self.litesam(data)

        k0 = data["mkpts0_f"].detach().cpu().numpy()
        k1 = data["mkpts1_f"].detach().cpu().numpy()
        conf = (
            data["mconf"].detach().cpu().numpy()
            if "mconf" in data
            else None
        )

        geo = estimate_geolocation(
            k0,
            k1,
            best_tile_path,
            self.cfg.target_size,
            conf=conf,
            verbose=verbose,
        )
        if geo is None:
            return None

        geo["best_tile"] = str(best_tile)
        geo["fine_results"] = fine_results[:10]
        return geo
