from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "anemia_prediction_model.pkl"

try:
    saved_data = joblib.load(MODEL_PATH)

    # Model saved using the previous code
    if isinstance(saved_data, dict):
        model = saved_data["model"]
        label_encoder = saved_data["label_encoder"]
        feature_columns = saved_data["feature_columns"]
    else:
        # If only the model was saved
        model = saved_data
        label_encoder = None
        feature_columns = [
            "age",
            "sex",
            "rbc",
            "pcv",
            "mcv",
            "mch",
            "mchc",
            "rdw",
            "tlc",
            "plt_mm3",
            "hgb"
        ]

    print("Model loaded successfully.")

except Exception as e:
    print("Error loading model:", e)
    model = None
    label_encoder = None
    feature_columns = []


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Anemia Prediction API is running",
        "status": "success"
    })


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    if model is not None:
        return jsonify({
            "status": "healthy",
            "model_loaded": True
        })

    return jsonify({
        "status": "error",
        "model_loaded": False
    }), 500


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model is not loaded"
        }), 500

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data received"
            }), 400

        # --------------------------------------------------
        # Check required fields
        # --------------------------------------------------

        missing_fields = [
            column for column in feature_columns
            if column not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # --------------------------------------------------
        # Create dataframe
        # --------------------------------------------------

        input_data = {}

        for column in feature_columns:
            input_data[column] = [data[column]]

        input_df = pd.DataFrame(input_data)

        # --------------------------------------------------
        # Convert numeric fields
        # --------------------------------------------------

        numeric_columns = [
            "age",
            "rbc",
            "pcv",
            "mcv",
            "mch",
            "mchc",
            "rdw",
            "tlc",
            "plt_mm3",
            "hgb"
        ]

        for column in numeric_columns:

            if column in input_df.columns:

                input_df[column] = pd.to_numeric(
                    input_df[column],
                    errors="coerce"
                )

        # --------------------------------------------------
        # Check invalid numeric values
        # --------------------------------------------------

        if input_df[numeric_columns].isnull().any().any():

            invalid_columns = input_df[numeric_columns].columns[
                input_df[numeric_columns].isnull().any()
            ].tolist()

            return jsonify({
                "error": "Invalid numeric value",
                "columns": invalid_columns
            }), 400

        # --------------------------------------------------
        # Make prediction
        # --------------------------------------------------

        prediction = model.predict(input_df)[0]

        # --------------------------------------------------
        # Convert encoded prediction back to original label
        # --------------------------------------------------

        if label_encoder is not None:

            try:
                prediction_label = label_encoder.inverse_transform(
                    [int(prediction)]
                )[0]

            except Exception:
                prediction_label = str(prediction)

        else:
            prediction_label = str(prediction)

        # --------------------------------------------------
        # Probability if available
        # --------------------------------------------------

        probabilities = None

        if hasattr(model, "predict_proba"):

            try:

                probability_values = model.predict_proba(input_df)[0]

                probabilities = {}

                if label_encoder is not None:

                    classes = label_encoder.classes_

                    for i, class_name in enumerate(classes):
                        probabilities[str(class_name)] = round(
                            float(probability_values[i]) * 100,
                            2
                        )

                else:

                    for i, class_name in enumerate(model.classes_):
                        probabilities[str(class_name)] = round(
                            float(probability_values[i]) * 100,
                            2
                        )

            except Exception:
                probabilities = None

        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        response = {
            "success": True,
            "prediction": prediction_label
        }

        if probabilities is not None:
            response["probabilities"] = probabilities

        return jsonify(response)

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
