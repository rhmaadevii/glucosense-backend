from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import tensorflow as tf
import joblib

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ===========================
# Firebase
# ===========================

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# ===========================
# AI Model
# ===========================

model = tf.keras.models.load_model("model/best_model.keras")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return "GlucoSense AI Backend Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    gender = float(data["gender"])
    age = float(data["age"])
    hypertension = float(data["hypertension"])
    heart_disease = float(data["heartDisease"])
    bmi = float(data["bmi"])
    hba1c = float(data["hba1c"])
    glucose = float(data["glucose"])

    input_data = np.array([[
        gender,
        age,
        hypertension,
        heart_disease,
        bmi,
        hba1c,
        glucose
    ]])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled, verbose=0)

    probability = float(prediction[0][0])

    result = "High Risk" if probability >= 0.5 else "Low Risk"

    probability_percent = round(probability * 100, 2)

    # ===========================
    # Save to Firebase
    # ===========================

    db.collection("predictions").add({

        "gender": "Male" if gender == 1 else "Female",

        "age": age,

        "hypertension": bool(hypertension),

        "heartDisease": bool(heart_disease),

        "bmi": bmi,

        "hba1c": hba1c,

        "glucose": glucose,

        "result": result,

        "probability": probability_percent,

        "createdAt": datetime.now()

    })

    return jsonify({

        "result": result,

        "probability": probability_percent

    })


import os

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )