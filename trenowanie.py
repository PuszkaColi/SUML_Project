import joblib
import numpy as np
import pandas as pd

# Import Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Import Model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline


def load_data():
    df = pd.read_csv("dane.csv")
    return df


def delete_duplicates(df):
    duplicate_rows_data = df[df.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_data.shape)

    return df.drop_duplicates()


def show_distinct_values():
    for column in df.columns:
        num_distinct_values = len(df[column].unique())
        print(f"{column}: {num_distinct_values} distinct values")


def standardize_data(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


if __name__ == "__main__":
    df = load_data()
    delete_duplicates(df)
    # standardize_data(df)
    print(df)

    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')

    # Definiowanie siatki hiperparametrów
    # param_grid = {
    #     'classifier__n_estimators': [50, 100, 200],
    #     'classifier__max_depth': [3, 5, 10],
    #     'classifier__learning_rate': [0.01, 0.1, 0.2],
    #     'classifier__subsample': [0.6, 0.8, 1.0],
    #     'classifier__colsample_bytree': [0.6, 0.8, 1.0]
    # }

    # Tworzenie pipeline z XGBoost jako klasyfikatorem
    clf = Pipeline(steps=[('classifier', xgb_model)])

    # Inicjalizacja GridSearchCV
    # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Podział danych na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trenowanie modelu GridSearchCV
    # grid_search.fit(X_train, y_train)
    clf.fit(X_train, y_train)

    # Wypisanie najlepszych parametrów
    # print("Best Parameters: ", grid_search.best_params_)

    # Dokonywanie predykcji na zbiorze testowym
    # y_pred = grid_search.predict(X_test)
    y_pred = clf.predict(X_test)

    # Ocena dokładności modelu
    print("Model Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Tworzenie macierzy konfuzji
    cm = confusion_matrix(y_test, y_pred)

    # Wizualizacja macierzy konfuzji
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # Importance params

    importances = xgb_model.feature_importances_

    # Uzyskanie nazw cech

    feature_names = X.columns

    # Tworzenie DataFrame z importancją cech

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Sortowanie według importancji

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Wyświetlanie wyników

    print(feature_importance_df)

    # Wizualizacja importancji cech

    plt.figure(figsize=(10, 6))

    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)

    plt.title('Feature Importance')

    plt.xlabel('Importance')

    plt.ylabel('Feature')

    plt.show()

    joblib.dump(clf, 'diabetes_xgb_model.joblib')
