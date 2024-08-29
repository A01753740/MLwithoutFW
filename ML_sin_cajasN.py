import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def limpieza_datos(dirección):
    # Cargar el dataset de diabetes
    df = pd.read_csv(dirección)

    # Muestreo aleatorio del 15% de los datos
    df = df.sample(frac=0.15, random_state=42)

    df['gender_numeric'] = df['gender'].replace({'Male': 0, 'Female': 1})

    df = df[df['gender_numeric'] != 'Other']

    # df['age'] = df['age'].astype(int)

    # Definir las características (X) y la variable objetivo (y)
    X = df[['age', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender_numeric']]
    y = df['diabetes']

    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def feed_forward(X, weights, biases):
    A = X
    activations = [A]  
    Z_values = []
    
    for i in range(len(weights)):
        Z = np.dot(A, weights[i]) + biases[i]
        Z_values.append(Z)
        A = sigmoid(Z)
        activations.append(A)
    
    return activations, Z_values

def backpropagation(activations, Z_values, y, weights):
    m = y.shape[0]
    dZ = activations[-1] - y.reshape(-1, 1)

    dW = np.dot(activations[-2].T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    
    grads = [(dW, db)]
    
    for i in range(len(weights) - 2, -1, -1):
        dA = np.dot(dZ, weights[i + 1].T)
        dZ = dA * sigmoid_derivative(Z_values[i])
        dW = np.dot(activations[i].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        grads.insert(0, (dW, db)) 

    return grads

def update_parameters(weights, biases, grads, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * grads[i][0]  
        biases[i] -= learning_rate * np.squeeze(grads[i][1])  
        
        biases[i] = biases[i].reshape(-1)
    return weights, biases


def train(X_train, y_train, weights, biases, epochs, learning_rate):
    y_train = y_train.values 
    y_train = y_train.reshape(-1, 1)
    
    for epoch in range(epochs):
        activations, Z_values = feed_forward(X_train, weights, biases)
        grads = backpropagation(activations, Z_values, y_train, weights)
        weights, biases = update_parameters(weights, biases, grads, learning_rate)
        
        if epoch % 100 == 0:
            predictions = activations[-1]
            loss = np.mean((predictions - y_train) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, biases

def predict(X, weights, biases):
    activations, _ = feed_forward(X, weights, biases)
    output = activations[-1]
    predictions = (output >= 0.5).astype(int)
    return predictions

def display_menu():
    print("--------------------------")
    print("   Predicción de diabetes   ")
    print("--------------------------\n")

def get_numeric_input(prompt, type_func=float, valid_range=None):
    while True:
        try:
            value = type_func(input(prompt))
            if valid_range and value not in valid_range:
                raise ValueError
            return value
        except ValueError:
            print("Entrada no válida. Inténtelo de nuevo.")

def get_user_data():
    age = get_numeric_input("Ingrese la edad: ", type_func=float)
    heart_disease = get_numeric_input("¿Tiene enfermedad cardíaca? (No:0/Si:1): ", type_func=int, valid_range=[0, 1])
    bmi = get_numeric_input("Ingrese el índice de masa corporal (IMC): ", type_func=float)
    hba1c = get_numeric_input("Ingrese el nivel de Hemoglobin A1c: ", type_func=float)
    blood_glucose = get_numeric_input("Ingrese el nivel de glucosa en sangre: ", type_func=float)
    sex = get_numeric_input("Indique su sexo (0:hombre/1:mujer): ", type_func=int, valid_range=[0, 1])

    return np.array([[age, heart_disease, bmi, hba1c, blood_glucose, sex]])

def display_results(accuracy, conf_matrix, class_report):
    print(f"\nPrecisión del modelo: {accuracy:.2f}")
    print("Matriz de confusión:")
    print(conf_matrix)
    print("Reporte de clasificación:")
    print(class_report)

def main():
    display_menu()

    # Limpieza de datos
    X_train, X_test, y_train, y_test, scaler = limpieza_datos('diabetes_prediction_dataset.csv')

    # Inicializar pesos y biases
    input_size = X_train.shape[1]
    hidden_size = 5
    output_size = 1

    weights = [
        np.random.randn(input_size, hidden_size),
        np.random.randn(hidden_size, output_size)
    ]

    biases = [
        np.random.randn(hidden_size),
        np.random.randn(output_size)
    ]

    # Entrenamiento
    epochs = get_numeric_input("Ingrese el número de épocas: ", type_func=int)
    learning_rate = get_numeric_input("Ingrese la tasa de aprendizaje: ", type_func=float)

    weights, biases = train(X_train, y_train, weights, biases, epochs, learning_rate)

    # Predicciones con datos de prueba
    y_pred = predict(X_test, weights, biases)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    display_results(accuracy, conf_matrix, class_report)

    while True:
        # Ingresar nuevos datos para predicción
        print("\nIngrese los datos del paciente para realizar una predicción.")
        new_data = get_user_data()

        # Escalar y predecir con los nuevos datos
        new_data_scaled = scaler.transform(new_data)
        new_predictions = predict(new_data_scaled, weights, biases)

        if new_predictions[0] == 1:
            print("\nEl paciente tiene diabetes.")
        else:
            print("\nEl paciente no tiene diabetes.")

        # Preguntar si se desea hacer otra predicción
        repeat = input("\n¿Desea hacer otra predicción? (s/n): ").strip().lower()
        if repeat != 's':
            print("\nGracias por usar el sistema de predicción de diabetes. ¡Hasta luego!")
            break

if __name__ == '__main__':
    main()
