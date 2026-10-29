
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "student_result_model.pkl"

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]


def load_model():
    """Load the trained student result prediction model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    """Return basic service information."""
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200


@app.get("/health")
def health_check():
    """Health endpoint used by Docker and GitHub Actions."""
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction",
    }), 200


@app.post("/predict")
def predict():
    """Predict PASS or FAIL from the four student features."""
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A valid JSON object is required",
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        # Check that all feature values are numeric.
        for feature in FEATURES:
            sample[feature] = pd.to_numeric(
                sample[feature], errors="raise"
            )

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except FileNotFoundError as error:
        return jsonify({
            "error": str(error),
        }), 500

    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numeric",
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "An internal error occurred during prediction",
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
