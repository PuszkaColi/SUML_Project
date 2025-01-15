import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
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
    print(df)

    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')

    clf = Pipeline(steps=[('classifier', xgb_model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Model Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    importances = xgb_model.feature_importances_

    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print(feature_importance_df)

    plt.figure(figsize=(10, 6))

    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)

    plt.title('Feature Importance')

    plt.xlabel('Importance')

    plt.ylabel('Feature')

    plt.show()

    joblib.dump(clf, 'diabetes_xgb_model.joblib')
