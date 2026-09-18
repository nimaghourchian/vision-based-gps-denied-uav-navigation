
import faiss
import numpy as np
from pathlib import Path
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
import re
from src.loftr import LoFTR
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
import time
import rasterio
from rasterio.warp import transform

PARENT_CACHE = Path("faiss_cache/parent")
SUB_CACHE = Path("faiss_cache/subtiles")
SUB_TILES_DIR = Path("tiles_out/subtiles")
UAV_IMAGE = "data/example_uav_image.jpg"

TOPK_STAGE1 = 5
TOPK_STAGE2 = 5
IMG_SIZE = (320, 320)

LITESAM_CFG = r"configs/loftr/eloftr_full.py"
LITESAM_WEIGHTS = r"weights/mloftr.ckpt"
TARGET_SIZE = 320
THR = 0.2


def run_litesam(model, tile_path, uav_path, target_size, device):
    tile_gray = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)
    uav_gray = cv2.imread(str(uav_path), cv2.IMREAD_GRAYSCALE)

    if tile_gray is None:
        raise FileNotFoundError(tile_path)
    if uav_gray is None:
        raise FileNotFoundError(uav_path)

    tile_gray = cv2.resize(tile_gray, (target_size, target_size))
    uav_gray = cv2.resize(uav_gray, (target_size, target_size))

    tile_tensor = torch.from_numpy(tile_gray.astype(np.float32) / 255.0)[None, None].to(device)
    uav_tensor = torch.from_numpy(uav_gray.astype(np.float32) / 255.0)[None, None].to(device)

    data = {"image0": tile_tensor, "image1": uav_tensor}
    with torch.no_grad():
        model(data)

    k0 = data["mkpts0_f"].cpu().numpy()
    k1 = data["mkpts1_f"].cpu().numpy()
    conf = data["mconf"].cpu().numpy()

    tile_bgr = cv2.resize(cv2.imread(str(tile_path)), (target_size, target_size))
    uav_bgr = cv2.resize(cv2.imread(str(uav_path)), (target_size, target_size))
    return tile_bgr, uav_bgr, k0, k1, conf


def draw_matches_side_by_side(
    img_left_bgr,
    img_right_bgr,
    kpts_left,
    kpts_right,
    conf,
    max_draw=50,
    conf_thr=0.0,
):
    h, w = img_left_bgr.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = img_left_bgr
    canvas[:, w:] = img_right_bgr

    if len(kpts_left) == 0:
        return canvas

    mask = conf >= conf_thr
    kpts_left = kpts_left[mask]
    kpts_right = kpts_right[mask]
    conf = conf[mask]

    idx = np.argsort(-conf)
    kpts_left = kpts_left[idx]
    kpts_right = kpts_right[idx]
    num_draw = min(max_draw, len(kpts_left))

    for i in range(num_draw):
        x0, y0 = kpts_left[i]
        x1, y1 = kpts_right[i]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt1 = (int(x0), int(y0))
        pt2 = (int(x1) + w, int(y1))
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)

    return canvas


def build_litesam():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(LITESAM_CFG)
    cfg.LOFTR.COARSE.NPE = [832, 832, 1184, 1184]
    cfg.LOFTR.MATCH_COARSE.THR = THR

    model_config = lower_config(cfg)
    model = LoFTR(config=model_config["loftr"])
    state = torch.load(
        LITESAM_WEIGHTS,
        map_location=DEVICE,
        weights_only=False,
    )["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


def load_gray_tensor(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img)[None, None].to(DEVICE)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_base = models.resnet50(weights="IMAGENET1K_V2")
_base = torch.nn.Sequential(*list(_base.children())[:-2]).to(DEVICE).eval()


class GeM(torch.nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)

    def forward(self, x):
        return torch.pow(torch.mean(torch.pow(x, self.p), dim=(2, 3)), 1.0 / self.p)


_gem = GeM().to(DEVICE)
_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@torch.no_grad()
def extract_global_desc(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = _transform(img_rgb).unsqueeze(0).to(DEVICE)
    feat = _base(x)
    feat = _gem(feat)
    vector = feat.squeeze(0).cpu().numpy().astype(np.float32)
    vector /= np.linalg.norm(vector) + 1e-8
    return vector


print("Runtime starts ...")
t5 = time.time()
parent_index = faiss.read_index(str(PARENT_CACHE / "tiles.index"))
parent_names = np.load(str(PARENT_CACHE / "tile_names.npy"), allow_pickle=True)
sub_index = faiss.read_index(str(SUB_CACHE / "tiles.index"))
sub_names = np.load(str(SUB_CACHE / "tile_names.npy"), allow_pickle=True)

uav_img = cv2.imread(UAV_IMAGE, cv2.IMREAD_COLOR)
if uav_img is None:
    raise FileNotFoundError(UAV_IMAGE)
q = extract_global_desc(uav_img)[None, :]

scores1, idxs1 = parent_index.search(q, k=TOPK_STAGE1)
print("\nStage 1 — Top Parent Tiles:")
selected_parent_ids = []
for rank, (idx, score) in enumerate(zip(idxs1[0], scores1[0]), 1):
    parent_name = parent_names[idx]
    print(f"{rank}. {parent_name} score={score:.4f}")
    match = re.search(r"_(\d{6})_", parent_name)
    if match:
        selected_parent_ids.append(match.group(1))

candidate_indices = []
for i, sub_name in enumerate(sub_names):
    for pid in selected_parent_ids:
        if f"_{pid}_" in sub_name:
            candidate_indices.append(i)
            break
if not candidate_indices:
    raise RuntimeError("No sub-tiles matched selected parent IDs.")

sub_descs = np.zeros((len(candidate_indices), 2048), dtype=np.float32)
for j, idx in enumerate(candidate_indices):
    sub_descs[j] = sub_index.reconstruct(idx)

temp_index = faiss.IndexFlatIP(2048)
temp_index.add(sub_descs)
scores2, idxs2 = temp_index.search(q, k=min(TOPK_STAGE2, len(candidate_indices)))

print("\nStage 2 — Best Sub-Tiles:")
for rank, (local_idx, score) in enumerate(zip(idxs2[0], scores2[0]), 1):
    global_idx = candidate_indices[local_idx]
    sub_name = sub_names[global_idx]
    print(f"{rank}. {sub_name} score={score:.4f}")

print("\nStage 3 — LiteSAM Fine Matching")
model = build_litesam()
uav_tensor = load_gray_tensor(UAV_IMAGE)
fine_results = []

for local_idx in idxs2[0]:
    global_idx = candidate_indices[local_idx]
    sub_name = sub_names[global_idx]
    sub_path = SUB_TILES_DIR / sub_name
    sub_tensor = load_gray_tensor(sub_path)

    data = {"image0": sub_tensor, "image1": uav_tensor}
    t0 = time.time()
    with torch.no_grad():
        model(data)
    print(f"LiteSAM inference time: {time.time() - t0}")

    if "mconf" in data and data["mconf"].numel() > 0:
        num_matches = int(data["mconf"].numel())
        mean_conf = float(data["mconf"].mean().item())
    else:
        num_matches, mean_conf = 0, 0.0
    fine_results.append((sub_name, num_matches, mean_conf))

fine_results.sort(key=lambda x: (x[1], x[2]), reverse=True)
print("\nFinal Ranking After LiteSAM:")
for rank, (name, nm, mc) in enumerate(fine_results, 1):
    print(f"{rank}. {name} matches={nm} mean_conf={mc:.4f}")

best_tile = fine_results[0][0]
best_tile_path = SUB_TILES_DIR / best_tile
tile_img, uav_match_img, k0, k1, conf = run_litesam(
    model=model,
    tile_path=best_tile_path,
    uav_path=UAV_IMAGE,
    target_size=TARGET_SIZE,
    device=DEVICE,
)
canvas = draw_matches_side_by_side(
    img_left_bgr=tile_img,
    img_right_bgr=uav_match_img,
    kpts_left=k0,
    kpts_right=k1,
    conf=conf,
    max_draw=50,
    conf_thr=0.4,
)
print(f"One inference session: {time.time() - t5}")
cv2.imshow("Best tile (left) vs UAV (right) - LiteSAM matches", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

CONF_THR = 0.3
N_TOP = 300
k0_f, k1_f = k0, k1
if conf is not None and len(conf) == len(k0):
    keep = conf >= CONF_THR
    k0_f = k0[keep]
    k1_f = k1[keep]
    conf_f = conf[keep]
    if len(conf_f) > N_TOP:
        idx = np.argsort(-conf_f)[:N_TOP]
        k0_f = k0_f[idx]
        k1_f = k1_f[idx]

if len(k0_f) < 4:
    print("Not enough matches after filtering.")
else:
    H, mask = cv2.findHomography(k1_f, k0_f, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        print("Homography estimation failed.")
    else:
        inliers = int(mask.ravel().sum())
        total = len(mask)
        print(f"Inliers: {inliers}/{total} ratio={inliers / total if total else 0:.2f}")

        uav_center = np.array(
            [[[TARGET_SIZE / 2, TARGET_SIZE / 2]]], dtype=np.float32
        )
        tile_point = cv2.perspectiveTransform(uav_center, H)
        tx_resized, ty_resized = tile_point[0][0]

        from rasterio.transform import xy

        with rasterio.open(best_tile_path) as src:
            tx = tx_resized * (src.width / float(TARGET_SIZE))
            ty = ty_resized * (src.height / float(TARGET_SIZE))
            x_crs, y_crs = xy(src.transform, ty, tx, offset="center")
            lon, lat = transform(src.crs, "EPSG:4326", [x_crs], [y_crs])

        print("\nLocation (EPSG:4326 — WGS84):")
        print(f"Latitude : {lat[0]:.8f}")
        print(f"Longitude: {lon[0]:.8f}")
