import joblib
import pandas as pd

# Wczytaj model
model = joblib.load('diabetes_xgb_model.joblib')


# Funkcja do przetwarzania danych wejściowych
def preprocess_input(data):
    # Tworzenie DataFrame z wprowadzonych danych
    df = pd.DataFrame([data], columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
    ])
    return df


# Główna pętla konsolowa
if __name__ == "__main__":
    while True:
        user_input = input(
            "Wprowadź dane do prognozy (format: gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level), lub 'exit' aby zakończyć: ")

        if user_input.lower() == 'exit':
            break

        # Przetwórz dane wejściowe
        try:
            # Podziel dane na listę i konwertuj na odpowiednie typy
            input_data = user_input.split(',')

            # Upewnij się, że dane są odpowiednio przetworzone (np. konwersja do float/int)
            input_data[0] = int(input_data[0])  # gender (zakładam, że jest int)
            input_data[1] = float(input_data[1])  # age
            input_data[2] = int(input_data[2])  # hypertension
            input_data[3] = int(input_data[3])  # heart_disease
            input_data[4] = int(input_data[4])  # smoking_history
            input_data[5] = float(input_data[5])  # bmi
            input_data[6] = float(input_data[6])  # HbA1c_level
            input_data[7] = float(input_data[7])  # blood_glucose_level

            # Przetwarzanie danych
            X_new = preprocess_input(input_data)

            # Dokonywanie prognozy
            prediction = model.predict(X_new)

            # Wyświetlanie wyniku
            print(f"Prognoza: {'Diabetes' if prediction[0] == 1 else 'No Diabetes'}")

        except Exception as e:
            print(f"Wystąpił błąd: {e}")
