from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flask import Flask, jsonify, request, send_from_directory
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image  # noqa: E402
from sharp.models import PredictorParams, create_predictor  # noqa: E402
from sharp.utils import io  # noqa: E402
from sharp.utils.gaussians import save_ply  # noqa: E402

HOST = os.getenv("SHARP_HOST", "127.0.0.1")
PORT = int(os.getenv("SHARP_PORT", "7860"))

INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output"
VIEWER_DIR = REPO_ROOT / "viewer"
PAGE_DIR = REPO_ROOT / "page"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "src" / "models" / "sharp_2572gikvuh.pt"

MODEL: torch.nn.Module | None = None
MODEL_DEVICE: torch.device | None = None
MODEL_ERROR: str | None = None
MODEL_READY = threading.Event()
MODEL_LOAD_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()
INIT_LOCK = threading.Lock()
INIT_DONE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("sharp-web")

ALLOWED_EXTS = {ext.lower() for ext in io.get_supported_image_extensions()}

app = Flask(__name__, static_folder=None)


def _resolve_device() -> str:
    requested = os.getenv("SHARP_DEVICE", "").strip().lower()
    if requested in {"cpu", "cuda", "mps"}:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_checkpoint() -> Path | None:
    requested = os.getenv("SHARP_CHECKPOINT", "").strip()
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.is_file():
            return candidate
        LOGGER.warning("Requested checkpoint not found: %s", candidate)
    LOGGER.info("Checking local checkpoint at %s", DEFAULT_CHECKPOINT_PATH)
    if DEFAULT_CHECKPOINT_PATH.is_file():
        return DEFAULT_CHECKPOINT_PATH
    return None


def _load_model() -> None:
    global MODEL, MODEL_DEVICE, MODEL_ERROR
    with MODEL_LOAD_LOCK:
        if MODEL is not None or MODEL_READY.is_set():
            return
        try:
            device_name = _resolve_device()
            MODEL_DEVICE = torch.device(device_name)
            LOGGER.info("Loading model on %s...", device_name)

            checkpoint_path = _resolve_checkpoint()
            if checkpoint_path is None:
                allow_download = os.getenv("SHARP_ALLOW_DOWNLOAD", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                }
                if not allow_download:
                    raise FileNotFoundError(
                        "Checkpoint not found. Place the model at "
                        f"{DEFAULT_CHECKPOINT_PATH} or set SHARP_CHECKPOINT, "
                        "or set SHARP_ALLOW_DOWNLOAD=1 to download automatically."
                    )
                LOGGER.info("Downloading checkpoint from %s", DEFAULT_MODEL_URL)
                state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
            else:
                LOGGER.info("Loading checkpoint from %s", checkpoint_path)
                state_dict = torch.load(checkpoint_path, weights_only=True)

            predictor = create_predictor(PredictorParams())
            predictor.load_state_dict(state_dict)
            predictor.eval()
            predictor.to(MODEL_DEVICE)

            MODEL = predictor
            LOGGER.info("Model ready.")
        except Exception as exc:  # noqa: BLE001
            MODEL_ERROR = str(exc)
            LOGGER.exception("Model failed to load.")
        finally:
            MODEL_READY.set()


def _ensure_model_ready() -> None:
    if not MODEL_READY.is_set():
        LOGGER.info("Waiting for model load to finish...")
        MODEL_READY.wait()
    if MODEL_ERROR:
        raise RuntimeError(MODEL_ERROR)
    if MODEL is None or MODEL_DEVICE is None:
        raise RuntimeError("Model not initialized.")


def _start_background_model_load() -> None:
    if MODEL_READY.is_set() or MODEL is not None:
        return
    thread = threading.Thread(target=_load_model, daemon=True)
    thread.start()


def _make_job_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{stamp}_{suffix}"


def _normalize_output_name(name: str) -> str:
    if not name:
        raise ValueError("Empty filename.")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Empty filename.")
    if Path(cleaned).name != cleaned:
        raise ValueError("Invalid filename.")
    if not cleaned.lower().endswith(".ply"):
        cleaned = f"{cleaned}.ply"
    return cleaned


def _resolve_output_path(name: str) -> Path:
    cleaned = _normalize_output_name(name)
    output_root = OUTPUT_DIR.resolve()
    candidate = (output_root / cleaned).resolve()
    if output_root not in candidate.parents and candidate != output_root:
        raise ValueError("Invalid filename.")
    return candidate


def _list_outputs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not OUTPUT_DIR.exists():
        return items
    for path in OUTPUT_DIR.glob("*.ply"):
        if not path.is_file():
            continue
        stat = path.stat()
        ply_url = f"/outputs/{quote(path.name)}"
        viewer_url = f"/viewer/index.html?load={quote(ply_url, safe='/%')}"
        items.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "ply_url": ply_url,
                "viewer_url": viewer_url,
            }
        )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items


def _ensure_initialized() -> None:
    global INIT_DONE
    if INIT_DONE:
        return
    with INIT_LOCK:
        if INIT_DONE:
            return
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _start_background_model_load()
        INIT_DONE = True


@app.before_request
def _before_request() -> None:
    _ensure_initialized()


@app.route("/")
def index() -> Any:
    return send_from_directory(PAGE_DIR, "index.html")


@app.route("/page/<path:filename>")
def page_assets(filename: str) -> Any:
    return send_from_directory(PAGE_DIR, filename)


@app.route("/viewer/")
def viewer_index() -> Any:
    return send_from_directory(VIEWER_DIR, "index.html")


@app.route("/viewer/<path:filename>")
def viewer_assets(filename: str) -> Any:
    return send_from_directory(VIEWER_DIR, filename)


@app.route("/outputs/<path:filename>")
def outputs(filename: str) -> Any:
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/health")
def api_health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "model_ready": MODEL is not None and MODEL_DEVICE is not None,
            "device": str(MODEL_DEVICE) if MODEL_DEVICE is not None else None,
            "model_error": MODEL_ERROR,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict() -> Any:
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image'."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type '{ext}'."}), 400

    try:
        _ensure_model_ready()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    job_id = _make_job_id()
    input_path = INPUT_DIR / f"{job_id}{ext}"
    output_path = OUTPUT_DIR / f"{job_id}.ply"

    file.save(input_path)
    LOGGER.info("Request %s saved to %s", job_id, input_path.name)

    with INFER_LOCK:
        LOGGER.info("Running inference for %s", input_path.name)
        image, _, f_px = io.load_rgb(input_path)
        height, width = image.shape[:2]
        gaussians = predict_image(MODEL, image, f_px, MODEL_DEVICE)  # type: ignore[arg-type]
        LOGGER.info("Writing PLY to %s", output_path.name)
        save_ply(gaussians, f_px, (height, width), output_path)

    ply_url = f"/outputs/{quote(output_path.name)}"
    viewer_url = f"/viewer/index.html?load={quote(ply_url, safe='/%')}"
    return jsonify({"job_id": job_id, "ply_url": ply_url, "viewer_url": viewer_url})


@app.route("/api/outputs", methods=["GET"])
def api_outputs() -> Any:
    return jsonify({"items": _list_outputs()})


@app.route("/api/outputs/rename", methods=["POST"])
def api_outputs_rename() -> Any:
    payload = request.get_json(silent=True) or {}
    old_name = payload.get("old_name")
    new_name = payload.get("new_name")
    if not old_name or not new_name:
        return jsonify({"error": "Missing old_name or new_name."}), 400

    try:
        old_path = _resolve_output_path(str(old_name))
        new_path = _resolve_output_path(str(new_name))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not old_path.exists():
        return jsonify({"error": "File not found."}), 404
    if new_path.exists():
        return jsonify({"error": "Target name already exists."}), 409

    old_path.rename(new_path)
    return jsonify({"status": "ok", "old_name": old_path.name, "new_name": new_path.name, "items": _list_outputs()})


@app.route("/api/outputs/delete", methods=["POST"])
def api_outputs_delete() -> Any:
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    if not name:
        return jsonify({"error": "Missing name."}), 400
    try:
        target_path = _resolve_output_path(str(name))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not target_path.exists():
        return jsonify({"error": "File not found."}), 404

    target_path.unlink()
    return jsonify({"status": "ok", "name": target_path.name, "items": _list_outputs()})


@app.route("/api/shutdown", methods=["POST", "GET"])
def api_shutdown() -> Any:
    LOGGER.info("Shutdown requested by client.")
    shutdown = request.environ.get("werkzeug.server.shutdown")
    if shutdown is not None:
        shutdown()
        return jsonify({"status": "shutting down"})
    os._exit(0)


if __name__ == "__main__":
    _ensure_initialized()
    LOGGER.info("Starting server at http://%s:%s", HOST, PORT)
    if os.getenv("SHARP_NO_BROWSER", "").strip().lower() not in {"1", "true", "yes"}:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}/")).start()
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
