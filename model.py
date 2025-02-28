import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# 1. Load dataset
file_path = "C:\\Users\\vishn\\Downloads\\diabetes.csv"
df = pd.read_csv(file_path)

# 2. Basic data exploration
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Separate features and target
X = df.iloc[:, :-1]  # All columns except the last (Outcome)
y = df.iloc[:, -1]  # Outcome column

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Predictions and Evaluation
y_pred = model.predict(X_test_scaled)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Feature Importance (Traditional)
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame(
    {'Feature': feature_names, 'Importance': importances}
).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance in Diabetes Prediction (Random Forest)")
plt.show()

# 9. EXPLAINABLE AI with SHAP
# Create a SHAP explainer for the tree-based model.
explainer = shap.TreeExplainer(model)
# Compute SHAP values using the scaled test set.
shap_values = explainer.shap_values(X_test_scaled)
# Global Explanation: Summary Plot using the scaled data (ensuring shape consistency)
#shap.summary_plot(shap_values[1], X_test_scaled, feature_names=feature_names)


# 10. Interactive Prediction Function
def interactive_prediction():
    """
    Prompts for patient details, scales the input, and predicts diabetes risk.
    """
    print("\nEnter patient details for diabetes risk prediction:")
    try:
        pregnancies = float(input("Enter number of pregnancies: "))
        glucose = float(input("Enter glucose level: "))
        blood_pressure = float(input("Enter blood pressure: "))
        skin_thickness = float(input("Enter skin thickness: "))
        insulin = float(input("Enter insulin level: "))
        bmi = float(input("Enter BMI: "))
        dpf = float(input("Enter Diabetes Pedigree Function: "))
        age = float(input("Enter age: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Arrange the input data in the same order as the training features.
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    # Scale the input data using the same scaler.
    input_data_scaled = scaler.transform(input_data)

    # Make prediction.
    prediction = model.predict(input_data_scaled)
    risk_status = "at risk of diabetes" if prediction[0] == 1 else "not at risk of diabetes"
    print("\nPrediction: The patient is", risk_status)


interactive_prediction()
