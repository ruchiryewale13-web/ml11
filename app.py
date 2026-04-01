from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return """
    <h2>ML Predictor</h2>
    <form action="/predict_web" method="post">
        Age: <input name="age"><br><br>
        BMI: <input name="bmi"><br><br>
        <button type="submit">Predict</button>
    </form>
    """

@app.route("/predict_web", methods=["POST"])
def predict_web():
    age = float(request.form["age"])
    bmi = float(request.form["bmi"])

    prediction = model.predict([[age, bmi]])

    return f"<h3>Prediction: {prediction[0]}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
