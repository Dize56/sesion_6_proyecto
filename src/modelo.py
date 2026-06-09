from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def cargar_datos():
    # Cargar el conjunto de datos Iris
    iris = load_iris()
    return iris.data, iris.target


def entrenar_modelo(X_train, y_train, n_estimators=100, random_state=42):
    # Crear y entrenar el modelo de Random Forest
    modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(modelo, X_test, y_test):
    # Evaluar el modelo con métricas básicas de clasificación
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    reporte_texto = classification_report(y_test, y_pred)
    return {
        "accuracy": float(accuracy),
        "reporte_texto": reporte_texto,
    }