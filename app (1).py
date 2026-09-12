"""
app.py
A simple Flask API that loads the trained Iris model
and serves predictions through a /predict endpoint.
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and target names once at startup
model = joblib.load("iris_model.pkl")
target_names = joblib.load("target_names.pkl")

# Simple HTML form for manual testing in a browser
HOME_PAGE = """
<h2>Iris Flower Prediction API</h2>
<p>Send a POST request to /predict with JSON like:</p>
<pre>{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}</pre>

<h3>Or try it here:</h3>
<form action="/predict_form" method="post">
  Sepal Length: <input name="sepal_length" value="5.1"><br>
  Sepal Width: <input name="sepal_width" value="3.5"><br>
  Petal Length: <input name="petal_length" value="1.4"><br>
  Petal Width: <input name="petal_width" value="0.2"><br>
  <input type="submit" value="Predict">
</form>
<p>{{ result }}</p>
"""

@app.route("/")
def home():
    return render_template_string(HOME_PAGE, result="")

@app.route("/predict", methods=["POST"])
def predict():
    """JSON API endpoint for predictions."""
    data = request.get_json(force=True)
    try:
        features = np.array([[
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]])
        prediction = model.predict(features)[0]
        species = target_names[prediction]
        return jsonify({
            "prediction": int(prediction),
            "species": species
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400

@app.route("/predict_form", methods=["POST"])
def predict_form():
    """Simple form-based endpoint for browser testing."""
    try:
        features = np.array([[
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]])
        prediction = model.predict(features)[0]
        species = target_names[prediction]
        result = f"Predicted species: {species}"
    except Exception as e:
        result = f"Error: {e}"
    return render_template_string(HOME_PAGE, result=result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
