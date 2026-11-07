import os
import uuid
import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO


# ====== CONFIG ======
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"


# Create folders if not exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024 # 15 MB


MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    str(BASE_DIR.parent / "runs" / "train" / "military_or_not" / "weights" / "best.pt"),
))


app = Flask(__name__)
app.secret_key = "change-me" # set a stronger key in production
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# Load model once at startup
model = YOLO(MODEL_PATH)




def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT




@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("Lütfen bir görsel seçin.")
        return redirect(url_for("index"))


    file = request.files["image"]
    if file.filename == "":
        flash("Dosya seçilmedi.")
        return redirect(url_for("index"))


    if not allowed_file(file.filename):
        flash("Geçersiz dosya türü. (jpg, jpeg, png, bmp, webp)")
        return redirect(url_for("index"))
    
    # Unique job id and safe filename
    job_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename)

    # Save upload
    upload_path = UPLOAD_DIR / f"{job_id}_{filename}"
    file.save(str(upload_path))

    # Run prediction - save annotated image under static/results/<job_id>/
    # Ultralytics will mirror the filename under that folder
    results = model.predict(
        source=str(upload_path),
        conf=0.25,
        imgsz=640,
        save=True,
        project=str(RESULTS_DIR),
        name=job_id,
        exist_ok=True,
        save_txt=False,
        verbose=False,
    )

    # Collect simple metadata (class counts + per-detection list)
    meta = {"job_id": job_id, "filename": filename, "detections": [], "class_counts": {}}

    if results and len(results) > 0:
        r = results[0]
        names = r.names # dict: id->name
        boxes = r.boxes
        if boxes is not None and boxes.cls is not None:
            cls_ids = boxes.cls.tolist()
            confs = boxes.conf.tolist() if boxes.conf is not None else [None] * len(cls_ids)

            # Count per class
            for cid in cls_ids:
                cname = names.get(int(cid), str(cid))
                meta["class_counts"].setdefault(cname, 0)
                meta["class_counts"][cname] += 1
            
            # Store per detection
            for cid, c in zip(cls_ids, confs):
                meta["detections"].append({
                    "class_id": int(cid),
                    "class_name": names.get(int(cid), str(cid)),
                    "conf": float(c) if c is not None else None,
                })

    # Save metadata json next to the annotated image
    meta_path = RESULTS_DIR / job_id / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return redirect(url_for("result", job_id=job_id))

@app.route("/result/<job_id>")
def result(job_id):
    job_dir = RESULTS_DIR / job_id
    if not job_dir.exists():
        flash("Sonuç bulunamadı.")
        return redirect(url_for("index"))
    
    # Find the single annotated image in the job folder
    images = [p.name for p in job_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    annotated_name = images[0] if images else None

    # Load metadata
    meta_path = job_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Build static path for template
    annotated_url = None
    if annotated_name:
        annotated_url = url_for("static", filename=f"results/{job_id}/{annotated_name}")

        return render_template(
        "result.html",
        annotated_image=annotated_url,
        metadata=meta
    )


if __name__ == "__main__":
    # Debug modda çalıştır
    app.run(host="0.0.0.0", port=5000, debug=True)
    