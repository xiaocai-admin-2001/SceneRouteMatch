import os
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms


MODEL_PATH = os.getenv(
    "FEATURE_MODEL_PATH",
    "/opt/RoadSegmentTestService/models/vit_base_patch16_dinov3.lvd1689m/model.safetensors",
)
LOMA_ROOT = Path(os.getenv("LOMA_ROOT", "/opt/RoadSegmentTestService/LoMa"))
SCENE_ROOT = Path(os.getenv("SCENE_ROOT", "/opt/RoadSegmentTestService/scene"))
DEVICE = os.getenv("FEATURE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
PORT = int(os.getenv("FEATURE_SERVICE_PORT", "19001"))
MATCHER_MODE = os.getenv("FEATURE_MATCHER", "hybrid").lower()
LOMA_TOP_K = max(1, int(os.getenv("LOMA_TOP_K", "2")))
LOMA_KEYPOINTS = max(256, int(os.getenv("LOMA_KEYPOINTS", "2048")))
LOMA_FILTER_THRESHOLD = float(os.getenv("LOMA_FILTER_THRESHOLD", "0.1"))
LOMA_RANSAC_THRESHOLD = float(os.getenv("LOMA_RANSAC_THRESHOLD", "1.5"))

if MATCHER_MODE not in {"dino", "loma", "hybrid"}:
    raise ValueError(f"unsupported FEATURE_MATCHER: {MATCHER_MODE}")
if MATCHER_MODE in {"loma", "hybrid"}:
    sys.path.insert(0, str(LOMA_ROOT / "src"))
    from loma import LoMa
    from loma.loma import LoMaB, filter_matches, to_pixel_coords

app = Flask(__name__)
dino_model = timm.create_model(
    "vit_base_patch16_dinov3",
    pretrained=False,
    num_classes=0,
    checkpoint_path=MODEL_PATH,
).to(DEVICE).eval()
dino_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
loma_model = LoMa(LoMaB()) if MATCHER_MODE in {"loma", "hybrid"} else None

cache_lock = threading.Lock()
dino_cache = {}
loma_cache = {}
inference_lock = threading.Lock()


def scene_metadata(camera_dir):
    path = camera_dir / "distance_map.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def label_from_name(path, metadata=None):
    if metadata:
        item = metadata.get(path.name)
        if isinstance(item, dict):
            result = str(item.get("result", ""))
            if result in ("0", "1", "2"):
                return result
    first = path.name[:1]
    return first if first in ("1", "2") else "0"


def dino_feature(image):
    tensor = dino_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        value = dino_model(tensor).float().cpu().numpy().squeeze(0)
    return value / (np.linalg.norm(value) + 1e-8)


def cached_dino(path):
    stat = path.stat()
    key = str(path)
    with cache_lock:
        cached = dino_cache.get(key)
        if cached and cached[0:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
    with Image.open(path) as image:
        value = dino_feature(image)
    with cache_lock:
        dino_cache[key] = (stat.st_mtime_ns, stat.st_size, value)
    return value


def loma_features(path):
    stat = path.stat()
    key = str(path)
    with cache_lock:
        cached = loma_cache.get(key)
        if cached and cached[0:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2:]
    kpts, desc, height, width = loma_model.detect_and_describe(
        str(path), num_keypoints=LOMA_KEYPOINTS
    )
    value = (kpts, desc, height, width)
    with cache_lock:
        loma_cache[key] = (stat.st_mtime_ns, stat.st_size, *value)
    return value


def geometry_metrics(kpts_a, kpts_b, desc_a, desc_b, h1, w1, h2, w2):
    with torch.inference_mode():
        scores = loma_model(kpts_a, kpts_b, desc_a, desc_b)["scores"]
    m0, _, match_scores, _ = filter_matches(scores, LOMA_FILTER_THRESHOLD)
    valid = m0[0] > -1
    match_count = int(valid.sum().item())
    if match_count == 0:
        return {
            "match_count": 0, "average_match_score": 0.0,
            "inlier_count": 0, "inlier_ratio": 0.0, "loma_score": 0.0,
        }
    ids_a = torch.where(valid)[0]
    ids_b = m0[0][valid]
    pts_a = to_pixel_coords(kpts_a[0][ids_a], h1, w1).cpu().numpy()
    pts_b = to_pixel_coords(kpts_b[0][ids_b], h2, w2).cpu().numpy()
    average_score = float(match_scores[0][valid].mean().item())
    inlier_count = 0
    if match_count >= 8:
        _, mask = cv2.findFundamentalMat(
            pts_a, pts_b,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=LOMA_RANSAC_THRESHOLD,
            confidence=0.999,
            maxIters=10000,
        )
        if mask is not None:
            inlier_count = int(mask.ravel().astype(bool).sum())
    inlier_ratio = inlier_count / max(1, match_count)
    coverage = min(match_count / 300.0, 1.0)
    loma_score = 0.50 * inlier_ratio + 0.30 * coverage + 0.20 * average_score
    return {
        "match_count": match_count,
        "average_match_score": average_score,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "loma_score": loma_score,
    }


def candidate_paths(camera_id):
    camera_dir = SCENE_ROOT / camera_id
    if not camera_dir.is_dir():
        return [], {}
    metadata = scene_metadata(camera_dir)
    return sorted(
        p for p in camera_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        and label_from_name(p, metadata) in {"0", "1", "2"}
    ), metadata


@app.get("/ping")
def ping():
    return jsonify({
        "status_code": 200,
        "msg": None,
        "data": {
            "device": DEVICE,
            "matcher": MATCHER_MODE,
            "loma_top_k": LOMA_TOP_K if MATCHER_MODE == "hybrid" else None,
        },
    })


@app.post("/match")
def match():
    started = time.perf_counter()
    camera_id = request.form.get("camera_id", "")
    if not camera_id or "/" in camera_id or "\\" in camera_id or ".." in camera_id:
        return jsonify({"status_code": 400, "data": None, "msg": "invalid camera_id"}), 400
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify({"status_code": 400, "data": None, "msg": "missing image"}), 400
    candidates, metadata = candidate_paths(camera_id)
    if not candidates:
        return jsonify({"status_code": 404, "data": None, "msg": "no scene images"}), 404

    suffix = Path(uploaded.filename or "query.jpg").suffix or ".jpg"
    temp_path = None
    try:
        raw = uploaded.read()
        with Image.open(__import__("io").BytesIO(raw)) as image:
            image = image.convert("RGB")
            query_dino = dino_feature(image)
        dino_rows = []
        for path in candidates:
            dino_rows.append({
                "path": path,
                "image": path.name,
                "label": label_from_name(path, metadata),
                "dino_score": float(np.dot(query_dino, cached_dino(path))),
            })
        dino_rows.sort(key=lambda item: item["dino_score"], reverse=True)

        if MATCHER_MODE == "dino":
            rows = [{**r, "score": r["dino_score"]} for r in dino_rows]
        else:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(raw)
                temp_path = Path(temp.name)
            with inference_lock:
                qk, qd, qh, qw = loma_model.detect_and_describe(
                    str(temp_path), num_keypoints=LOMA_KEYPOINTS
                )
                selected = (
                    dino_rows[:min(LOMA_TOP_K, len(dino_rows))]
                    if MATCHER_MODE == "hybrid" else dino_rows
                )
                rows = []
                for row in selected:
                    rk, rd, rh, rw = loma_features(row["path"])
                    metrics = geometry_metrics(qk, rk, qd, rd, qh, qw, rh, rw)
                    combined = (
                        0.15 * row["dino_score"] + 0.85 * metrics["loma_score"]
                        if MATCHER_MODE == "hybrid" else metrics["loma_score"]
                    )
                    rows.append({**row, **metrics, "score": combined})
            rows.sort(key=lambda item: item["score"], reverse=True)

        best = rows[0]
        public_rows = [{k: v for k, v in row.items() if k != "path"} for row in rows]
        return jsonify({
            "status_code": 200,
            "msg": None,
            "data": {
                "result": best["label"],
                "best_image": best["image"],
                "best_score": best["score"],
                "matcher": MATCHER_MODE,
                "compared_count": len(rows),
                "candidate_count": len(candidates),
                "runtime_ms": (time.perf_counter() - started) * 1000.0,
                "scores": public_rows,
            },
        })
    except Exception as exc:
        app.logger.exception("hybrid feature match failed")
        return jsonify({"status_code": 500, "data": None, "msg": str(exc)}), 500
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, threaded=True)
