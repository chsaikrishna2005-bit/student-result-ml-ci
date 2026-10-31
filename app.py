
from pathlib import Path
import logging
import os
import sys
import time
import uuid

import joblib
import pandas as pd

from flask import Flask, jsonify, request, g, Response

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

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
# PROMETHEUS METRICS
# --------------------------------------------------

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded"
)

MODEL_LOADED.set(0)

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
# REQUEST LOGGING AND METRICS MIDDLEWARE
# --------------------------------------------------

@app.before_request
def start_request_logging():
    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def finish_request_logging(response):

    duration = time.perf_counter() - g.start_time
    duration_ms = duration * 1000

    response.headers["X-Request-ID"] = g.request_id

    path = request.path
    method = request.method
    status = response.status_code

    # Do not include Prometheus scrape requests
    # in application traffic metrics.
    if path != "/metrics":

        HTTP_REQUESTS.labels(
            method=method,
            path=path,
            status=str(status)
        ).inc()

        HTTP_LATENCY.labels(
            method=method,
            path=path
        ).observe(duration)

    log_level = (
        logging.WARNING
        if status >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        (
            "event=http_request request_id=%s "
            "method=%s path=%s status=%s duration_ms=%.2f"
        ),
        g.request_id,
        method,
        path,
        status,
        duration_ms,
    )

    return response


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():

        MODEL_LOADED.set(0)

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

        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name,
        )

        return model

    except Exception:

        MODEL_LOADED.set(0)

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

    logger.info("event=home_endpoint_accessed")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    }), 200


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    logger.info("event=health_check_success")

    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction",
    }), 200


# --------------------------------------------------
# PROMETHEUS METRICS ENDPOINT
# --------------------------------------------------

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


# --------------------------------------------------
# STUDENT RESULT PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        logger.warning(
            "event=validation_error reason=invalid_json"
        )

        return jsonify({
            "error": "A valid JSON object is required",
        }), 400

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

        logger.info(
            "event=prediction_request feature_names=%s",
            FEATURES,
        )

        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

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

        PREDICTIONS.labels(
            result="success"
        ).inc()

        logger.info(
            "event=prediction_success prediction=%s",
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        }), 200

    except FileNotFoundError:

        PREDICTIONS.labels(
            result="error"
        ).inc()

        logger.exception(
            "event=prediction_failed reason=model_not_found"
        )

        return jsonify({
            "error": "Model file not found",
        }), 500

    except (TypeError, ValueError):

        PREDICTIONS.labels(
            result="error"
        ).inc()

        logger.warning(
            "event=validation_error reason=non_numeric_input"
        )

        return jsonify({
            "error": "All feature values must be numeric",
        }), 400

    except Exception:

        PREDICTIONS.labels(
            result="error"
        ).inc()

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

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
    )
