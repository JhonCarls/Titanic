import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Cargar el archivo
file_path = 'Libro1.xlsx'  # Cambia esto por la ruta de tu archivo
data = pd.read_excel(file_path)

# Introducir un poco de ruido en las características numéricas
data['Age'] = data['Age'] + np.random.normal(0, 0.5, data['Age'].shape)
data['Fare'] = data['Fare'] + np.random.normal(0, 0.5, data['Fare'].shape)

# Variables de entrada y salida
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Preprocesamiento de características numéricas y categóricas
numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))])

# Crear el preprocesador y pipeline completo
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Ajuste de regularización (C) en el clasificador de Regresión Logística con un valor más bajo
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(C=0.02, max_iter=200))])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Validación cruzada con 5 pliegues
cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
cross_val_mean_score = cross_val_scores.mean()

# Entrenar en conjunto de entrenamiento completo y evaluar en el conjunto de prueba
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluación
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Resultados
print("Cross-Validation Mean Accuracy:", cross_val_mean_score)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)
print(f"\nTest Set Accuracy: {accuracy*100:.2f}%")
