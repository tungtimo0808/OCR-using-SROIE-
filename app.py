import os
import json
import re
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# ── Lazy-load models (only once) ─────────────────────────────────────────────
_models = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load all three models. Called once on first request."""
    if _models:
        return _models

    import torch
    import torchvision
    from torchvision.transforms import functional as TF
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        LayoutLMv3Processor,
        LayoutLMv3ForTokenClassification,
    )

    DATA_ROOT = os.path.dirname(__file__)
    DET_PT = os.path.join(DATA_ROOT, "best_det.pt")
    REC_PT = os.path.join(DATA_ROOT, "best_rec.pt")
    IE_DIR  = os.path.join(DATA_ROOT, "layoutlmv3_sroie_out")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DET
    def build_det(num_classes=2):
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_f, num_classes)
        )
        return m

    det_model = build_det().to(device)
    det_model.load_state_dict(torch.load(DET_PT, map_location="cpu"), strict=False)
    det_model.eval()

    # REC
    REC_BASE = "microsoft/trocr-base-printed"
    rec_processor = TrOCRProcessor.from_pretrained(REC_BASE)
    rec_model = VisionEncoderDecoderModel.from_pretrained(REC_BASE).to(device)
    rec_model.load_state_dict(torch.load(REC_PT, map_location="cpu"), strict=False)
    rec_model.eval()

    # IE
    ie_processor = LayoutLMv3Processor.from_pretrained(IE_DIR, apply_ocr=False)
    ie_model = LayoutLMv3ForTokenClassification.from_pretrained(IE_DIR).to(device)
    ie_model.eval()

    _models.update(dict(
        device=device,
        det_model=det_model,
        rec_processor=rec_processor,
        rec_model=rec_model,
        ie_processor=ie_processor,
        ie_model=ie_model,
        id2label=ie_model.config.id2label,
        TF=TF,
        torch=torch,
        torchvision=torchvision,
    ))
    return _models


# ── Pipeline helpers (same logic as notebook) ────────────────────────────────

def clean_spaces(s):
    return re.sub(r"\s+", " ", s).strip()


def nms_xyxy(boxes, scores, iou_thr=0.3):
    if not boxes:
        return []
    import torch, torchvision
    b = torch.tensor(boxes, dtype=torch.float32)
    s = torch.tensor(scores, dtype=torch.float32)
    return torchvision.ops.nms(b, s, iou_thr).cpu().numpy().tolist()


def sort_boxes_reading_order(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    return np.lexsort((cx, cy)).tolist()


def normalize_box(box, W, H):
    x0, y0, x1, y1 = box
    return [
        max(0, min(1000, int(1000 * x0 / W))),
        max(0, min(1000, int(1000 * y0 / H))),
        max(0, min(1000, int(1000 * x1 / W))),
        max(0, min(1000, int(1000 * y1 / H))),
    ]


def split_line_words(text, line_box):
    text = clean_spaces(text)
    if not text:
        return [], []
    words = [w for w in text.split() if w]
    if not words:
        return [], []
    x0, y0, x1, y1 = line_box
    total_w = max(1, x1 - x0)
    total_chars = max(1, len(text))
    cur_x = x0
    boxes = []
    for w in words:
        ww = int(total_w * len(w) / total_chars)
        nx1 = min(x1, cur_x + max(2, ww))
        boxes.append([cur_x, y0, nx1, y1])
        cur_x = min(x1, nx1 + 2)
    boxes[-1][2] = x1
    return words, boxes


def extract_fields(words, labels):
    out = {"company": [], "date": [], "address": [], "total": []}
    cur = None
    for w, lab in zip(words, labels):
        if lab == "O":
            cur = None
            continue
        tag = lab.split("-", 1)[1].lower()
        if lab.startswith("B-"):
            cur = tag
            out.setdefault(cur, []).append(w)
        else:
            if cur != tag:
                cur = tag
            out.setdefault(cur, []).append(w)
    return {k: " ".join(v).strip() for k, v in out.items()}


date_re   = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
money_re  = re.compile(r"^\d{1,6}(\.\d{2})$")


def fallback_date(words):
    for w in words:
        m = date_re.search(w)
        if m:
            return m.group(1)
    return ""


def fallback_total(words):
    cands = []
    for w in words:
        w2 = w.replace(",", "")
        if money_re.match(w2):
            try:
                cands.append(float(w2))
            except Exception:
                pass
    return f"{max(cands):.2f}" if cands else ""


def run_pipeline(pil_img):
    m = load_models()
    device    = m["device"]
    TF        = m["TF"]
    torch     = m["torch"]
    det_model = m["det_model"]
    rec_proc  = m["rec_processor"]
    rec_model = m["rec_model"]
    ie_proc   = m["ie_processor"]
    ie_model  = m["ie_model"]
    id2label  = m["id2label"]

    # --- DET ---
    img_t = TF.to_tensor(pil_img).to(device)
    with torch.no_grad():
        out = det_model([img_t])[0]

    boxes  = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    keep   = np.where(scores >= 0.5)[0].tolist()
    boxes  = boxes[keep].tolist()
    scores = scores[keep].tolist()

    if not boxes:
        return {"error": "No text regions detected"}, {}

    keep2 = nms_xyxy(boxes, scores, 0.3)
    boxes = [boxes[i] for i in keep2][:200]
    boxes = [boxes[i] for i in sort_boxes_reading_order(boxes)]

    W, H = pil_img.size
    words, word_boxes_norm, word_boxes_raw = [], [], []

    with torch.no_grad():
        for b in boxes:
            x0, y0, x1, y1 = [int(v) for v in b]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W - 1, x1), min(H - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = pil_img.crop((x0, y0, x1, y1)).convert("RGB")
            pv   = rec_proc(images=crop, return_tensors="pt").pixel_values.to(device)
            ids  = rec_model.generate(pixel_values=pv, max_new_tokens=64, num_beams=1)
            text = rec_proc.batch_decode(ids, skip_special_tokens=True)[0]
            text = clean_spaces(text)
            if not text:
                continue
            ws, bs = split_line_words(text, [x0, y0, x1, y1])
            for w, bb in zip(ws, bs):
                w = clean_spaces(w)
                if w:
                    words.append(w)
                    word_boxes_raw.append(bb)
                    word_boxes_norm.append(normalize_box(bb, W, H))

    if not words:
        return {"error": "OCR produced no text"}, {}

    # --- IE ---
    enc = ie_proc(
        images=pil_img, text=words, boxes=word_boxes_norm,
        truncation=True, padding="max_length", max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = ie_model(**enc).logits

    pred_ids = logits.argmax(-1).squeeze(0).cpu().numpy().tolist()

    word_ids = ie_proc(
        images=pil_img, text=words, boxes=word_boxes_norm,
        truncation=True, padding="max_length", max_length=512,
        return_tensors=None,
    ).word_ids()

    word_pred = ["O"] * len(words)
    seen = set()
    for tok_i, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        word_pred[wid] = id2label[int(pred_ids[tok_i])]

    fields = extract_fields(words, word_pred)
    if not fields.get("date"):
        fields["date"] = fallback_date(words)
    if not fields.get("total"):
        fields["total"] = fallback_total(words)

    debug = {
        "num_boxes": len(boxes),
        "num_words": len(words),
        "ocr_preview": " ".join(words[:60]),
    }
    return fields, debug


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        pil_img = Image.open(save_path).convert("RGB")
        fields, debug = run_pipeline(pil_img)
        return jsonify({
            "success": True,
            "fields": fields,
            "debug": debug,
            "image_url": f"/uploads/{filename}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
