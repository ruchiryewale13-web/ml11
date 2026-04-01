from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return "ML Model API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        age = data["age"]
        bmi = data["bmi"]

        input_data = np.array([[age, bmi]])
        prediction = model.predict(input_data)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
