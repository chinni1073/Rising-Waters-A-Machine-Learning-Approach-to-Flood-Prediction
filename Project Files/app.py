from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("xg_flood_model.pkl")
xg_scaler = joblib.load("xg_scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = [
        float(request.form['cloud']),
        float(request.form['annual']),
        float(request.form['janfeb']),
        float(request.form['marchmay']),
        float(request.form['junsep'])
    ]

    df = np.array([data])
    scaled = xg_scaler.transform(df)
    pred = model.predict(scaled)[0]

    result = "Flood Expected" if pred == 1 else "No Flood"

    return render_template('predict.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
