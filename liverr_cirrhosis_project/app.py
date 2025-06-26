from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("rf_acc_68.pkl")
scaler = joblib.load("normalizer.pkl")

# List of feature names (8 features)
features = ['age', 'bilirubin', 'alk_phosphate', 'sgpt', 'sgot', 'proteins', 'albumin', 'ag_ratio']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from form
        input_data = [float(request.form[feature]) for feature in features]

        # Convert to numpy array and reshape for scaler/model
        input_array = np.array(input_data).reshape(1, -1)

        # Normalize input
        input_scaled = scaler.transform(input_array)

        # Predict using the model
        prediction = model.predict(input_scaled)[0]

        # Interpret prediction
        if prediction == 1:
            result = "High risk of Liver Cirrhosis."
        else:
            result = "Low risk of Liver Cirrhosis."

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

