
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")

# --------------------------------------------------
# FLASK APPLICATION CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "student_result_model.pkl"
)

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# --------------------------------------------------
# REQUEST LOGGING MIDDLEWARE
# --------------------------------------------------

@app.before_request
def start_request_logging():
    """Assign request ID and record request start time."""

    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()


@app.after_request
def finish_request_logging(response):
    """Log HTTP request details and response time."""

    duration_ms = (
        time.perf_counter() - g.start_time
    ) * 1000

    response.headers["X-Request-ID"] = g.request_id

    log_level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        (
            "event=http_request request_id=%s "
            "method=%s path=%s status=%s duration_ms=%.2f"
        ),
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    return response


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_model():
    """Load the trained student result prediction model."""

    if not MODEL_PATH.exists():
        logger.error(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name,
        )

        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    try:
        model = joblib.load(MODEL_PATH)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name,
        )

        return model

    except Exception:
        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name,
        )
        raise


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    """Return basic service information."""

    logger.info("event=home_endpoint_accessed")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200


# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Health endpoint used by Docker and GitHub Actions."""

    logger.info("event=health_check_success")

    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction",
    }), 200


# --------------------------------------------------
# STUDENT RESULT PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict():
    """Predict PASS or FAIL from student features."""

    data = request.get_json(silent=True)

    # Validate JSON body
    if not isinstance(data, dict):
        logger.warning(
            "event=validation_error reason=invalid_json"
        )

        return jsonify({
            "error": "A valid JSON object is required",
        }), 400

    # Check required features
    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error missing_fields=%s",
            missing_fields,
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        # Log feature names only.
        # Do not log actual student input values.
        logger.info(
            "event=prediction_request feature_names=%s",
            FEATURES,
        )

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        # Validate numeric inputs
        for feature in FEATURES:
            sample[feature] = pd.to_numeric(
                sample[feature],
                errors="raise"
            )

        model = load_model()

        prediction_code = int(
            model.predict(sample)[0]
        )

        prediction = (
            "PASS"
            if prediction_code == 1
            else "FAIL"
        )

        logger.info(
            "event=prediction_success prediction=%s",
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except FileNotFoundError:
        logger.exception(
            "event=prediction_failed reason=model_not_found"
        )

        return jsonify({
            "error": "Model file not found",
        }), 500

    except (TypeError, ValueError):
        logger.warning(
            "event=validation_error reason=non_numeric_input"
        )

        return jsonify({
            "error": "All feature values must be numeric",
        }), 400

    except Exception:
        logger.exception(
            "event=prediction_failed"
        )

        return jsonify({
            "error": "An internal error occurred during prediction",
        }), 500


# --------------------------------------------------
# APPLICATION STARTUP
# --------------------------------------------------

if __name__ == "__main__":

    logger.info(
        "event=application_starting service=student-result-prediction"
    )

    port = int(__import__("os").environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
    )
