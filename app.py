import joblib
import os
from flask import Flask, render_template, request, redirect, url_for

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Wczytaj model
model = joblib.load('diabetes_xgb_model.joblib')

# Strona główna
@app.route('/')
def home():
    return render_template('index.html')

# Obsługuje dane wejściowe
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Wczytaj dane wejściowe
        data = [
            int(request.form['gender']),
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            int(request.form['smoking_history']),
            float(request.form['bmi']),
            float(request.form['HbA1c_level']),
            float(request.form['blood_glucose_level'])
        ]

        # Dokonaj prognozy
        prediction = model.predict([data])

        # Zwróć wynik
        result = 'Cukrzyca' if prediction[0] == 1 else 'Brak cukrzycy'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
