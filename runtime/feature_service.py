import os
import json
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
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
CALIBRATION_API_URL = os.getenv(
    "CALIBRATION_API_URL",
    "http://10.88.4.37:9091/api/oauth/v3/marker/calibration/highway_scene_images",
)
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


def label_from_name(path, metadata=None):
    if metadata:
        item = metadata.get(str(path))
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
    url = CALIBRATION_API_URL + "?" + urllib.parse.urlencode({"camera_id": camera_id})
    last_error = None
    payload = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Connection": "close"})
            with urllib.request.urlopen(req, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(0.5 * (attempt + 1))
    if payload is None:
        raise RuntimeError(f"calibration api request failed after 5 attempts: {last_error}")
    if payload.get("code") != 200 or not isinstance(payload.get("data"), dict):
        raise RuntimeError(f"calibration api error: {payload}")

    paths = []
    metadata = {}
    for item in payload["data"].get("images", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("imagePath", "")))
        result = str(item.get("result", ""))
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            or result not in {"0", "1", "2"}
        ):
            continue
        metadata[str(path)] = {
            "result": result,
            "direction": item.get("direction"),
            "distances": item.get("distances") if isinstance(item.get("distances"), list) else [],
        }
        paths.append(path)
    return sorted(paths), metadata


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
    suffix = Path(uploaded.filename or "query.jpg").suffix or ".jpg"
    temp_path = None
    try:
        candidates, metadata = candidate_paths(camera_id)
        if not candidates:
            return jsonify({"status_code": 404, "data": None, "msg": "no scene images"}), 404
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
        best_metadata = metadata.get(str(best["path"]), {})
        public_rows = [{k: v for k, v in row.items() if k != "path"} for row in rows]
        return jsonify({
            "status_code": 200,
            "msg": None,
            "data": {
                "result": best["label"],
                "best_image": best["image"],
                "best_score": best["score"],
                "direction": best_metadata.get("direction"),
                "distances": best_metadata.get("distances", []),
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
