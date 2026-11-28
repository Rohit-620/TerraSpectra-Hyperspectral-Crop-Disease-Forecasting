from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import numpy as np
import joblib
import json
import uuid
from PIL import Image, UnidentifiedImageError
from model import extract_features
from forecasting import estimate_risk, advice_for

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
PLANT_UPLOAD_DIR = BASE / "static" / "plant_uploads"
MODEL_PATH = BASE / "models" / "terraspectra_model.joblib"
PLANT_MODEL_PATH = BASE / "models" / "plant_disease_mobilenet.keras"
CLASS_NAMES_PATH = BASE / "models" / "plant_class_names.json"
UPLOAD_DIR.mkdir(exist_ok=True)
PLANT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CUBE = {".npy"}
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)
app.secret_key = "change-this-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

hyperspectral_model = None
plant_model = None
plant_class_names = []

def load_models():
    global hyperspectral_model, plant_model, plant_class_names
    if MODEL_PATH.exists():
        hyperspectral_model = joblib.load(MODEL_PATH)
    if PLANT_MODEL_PATH.exists() and CLASS_NAMES_PATH.exists():
        try:
            import tensorflow as tf
            plant_model = tf.keras.models.load_model(PLANT_MODEL_PATH)
            plant_class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print("Plant model could not be loaded:", exc)
            plant_model = None
            plant_class_names = []

load_models()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if hyperspectral_model is None:
            flash("Hyperspectral demo model not found. Run: python train.py", "error")
            return redirect(url_for("index"))
        file = request.files.get("cube")
        if not file or not file.filename:
            flash("Please choose a hyperspectral .npy file.", "error")
            return redirect(url_for("index"))
        if Path(file.filename).suffix.lower() not in ALLOWED_CUBE:
            flash("Only .npy hyperspectral cubes are accepted. RGB JPG/PNG is not hyperspectral data.", "error")
            return redirect(url_for("index"))

        safe_name = secure_filename(file.filename)
        path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
        file.save(path)
        try:
            cube = np.load(path, allow_pickle=False)
            if cube.ndim != 3:
                raise ValueError("Expected a 3D array: height × width × spectral bands.")
            if cube.shape[2] < 3:
                raise ValueError("The cube must contain at least 3 spectral bands.")
            if not np.isfinite(cube).all():
                raise ValueError("The cube contains NaN or infinite values.")
            features = extract_features(cube)
            predicted = str(hyperspectral_model.predict([features])[0])
            probabilities = hyperspectral_model.predict_proba([features])[0]
            confidence = float(np.max(probabilities)) * 100
            temp = float(request.form.get("temperature", 25))
            humidity = float(request.form.get("humidity", 60))
            rainfall = float(request.form.get("rainfall", 0))
            history = request.form.get("disease_history", "no") == "yes"
            risk, risk_level = estimate_risk(temp, humidity, rainfall, history)
            return render_template(
                "result.html", prediction=predicted, confidence=round(confidence, 1),
                risk=risk, risk_level=risk_level, advice=advice_for(predicted, risk_level),
                shape=cube.shape, bands=cube.shape[2],
                probabilities=sorted(zip(hyperspectral_model.classes_, probabilities),
                                      key=lambda x: x[1], reverse=True)
            )
        except Exception as exc:
            flash(f"Could not process file: {exc}", "error")
            return redirect(url_for("index"))
        finally:
            path.unlink(missing_ok=True)
    return render_template("index.html", model_ready=(hyperspectral_model is not None))

@app.route("/plant-detect", methods=["GET", "POST"])
def plant_detect():
    if request.method == "POST":
        if plant_model is None or not plant_class_names:
            flash("Plant image model is not trained yet. Follow README: run python train_plant.py.", "error")
            return redirect(url_for("plant_detect"))
        file = request.files.get("plant_image")
        if not file or not file.filename:
            flash("Please select a plant leaf photo.", "error")
            return redirect(url_for("plant_detect"))
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE:
            flash("Please upload JPG, JPEG, PNG or WEBP image.", "error")
            return redirect(url_for("plant_detect"))
        image_name = f"{uuid.uuid4().hex}{ext}"
        image_path = PLANT_UPLOAD_DIR / image_name
        try:
            image = Image.open(file.stream).convert("RGB")
            image.save(image_path, format="JPEG", quality=90)
            import tensorflow as tf
            resized = image.resize((224, 224))
            arr = np.asarray(resized, dtype=np.float32)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
            scores = plant_model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
            best_idx = int(np.argmax(scores))
            predicted_class = plant_class_names[best_idx]
            confidence = float(scores[best_idx]) * 100
            # A low-confidence output should not be presented as a firm diagnosis.
            uncertain = confidence < 60
            return render_template(
                "plant_result.html",
                image_url=url_for("static", filename=f"plant_uploads/{image_name}"),
                prediction=predicted_class,
                confidence=round(confidence, 1),
                uncertain=uncertain,
                probabilities=sorted(zip(plant_class_names, scores),
                                     key=lambda x: x[1], reverse=True)[:5]
            )
        except (UnidentifiedImageError, OSError):
            image_path.unlink(missing_ok=True)
            flash("The uploaded file is not a readable image. Try another photo.", "error")
            return redirect(url_for("plant_detect"))
        except Exception as exc:
            image_path.unlink(missing_ok=True)
            flash(f"Image analysis failed: {exc}", "error")
            return redirect(url_for("plant_detect"))
    return render_template("plant.html", model_ready=(plant_model is not None))

@app.route("/health")
def health():
    return {
        "status": "ok",
        "hyperspectral_model_ready": hyperspectral_model is not None,
        "plant_image_model_ready": plant_model is not None
    }

@app.errorhandler(413)
def too_large(_error):
    flash("Image is too large. Maximum upload size is 8 MB.", "error")
    return redirect(url_for("plant_detect"))

if __name__ == "__main__":
    app.run(debug=True)
