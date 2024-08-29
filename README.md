# Predicción de Diabetes

Este proyecto implementa un modelo de predicción de diabetes utilizando un algoritmo de red neuronal simple. El código permite entrenar un modelo basado en un conjunto de datos de pacientes y realizar predicciones sobre la posibilidad de que un paciente tenga diabetes.

La base de datos se obtuvo del siguiente medio: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

## Requisitos

- Python 3.x
- Bibliotecas de Python:
  - `numpy`
  - `pandas`
  - `scikit-learn`

## Descripción de los archivos

- **diabetes_prediction_dataset.csv**: Archivo CSV que contiene el conjunto de datos con características de pacientes para entrenar y probar el modelo de predicción de diabetes.

## Descripción del código

### Funciones principales

1. **limpieza_datos(dirección):**
   - Carga y limpia el conjunto de datos de diabetes desde la dirección especificada.
   - Realiza un muestreo aleatorio del 15% de los datos y convierte las variables categóricas a numéricas.
   - Divide los datos en conjuntos de entrenamiento y prueba.
   - Estandariza las características.

2. **sigmoid(Z):**
   - Calcula la función sigmoide, utilizada como función de activación en la red neuronal.

3. **sigmoid_derivative(Z):**
   - Calcula la derivada de la función sigmoide, utilizada durante el proceso de retropropagación.

4. **feed_forward(X, weights, biases):**
   - Realiza la propagación hacia adelante en la red neuronal.

5. **backpropagation(activations, Z_values, y, weights):**
   - Calcula los gradientes necesarios para actualizar los pesos y sesgos durante el entrenamiento.

6. **update_parameters(weights, biases, grads, learning_rate):**
   - Actualiza los pesos y sesgos de la red neuronal utilizando los gradientes calculados.

7. **train(X_train, y_train, weights, biases, epochs, learning_rate):**
   - Entrena la red neuronal utilizando el conjunto de entrenamiento.

8. **predict(X, weights, biases):**
   - Realiza predicciones utilizando el modelo entrenado.

9. **display_menu():**
   - Muestra un menú de bienvenida en la consola.

10. **get_numeric_input(prompt, type_func=float, valid_range=None):**
    - Obtiene una entrada numérica del usuario con validación.

11. **get_user_data():**
    - Recopila datos del usuario para realizar una predicción de diabetes.

12. **display_results(accuracy, conf_matrix, class_report):**
    - Muestra los resultados del modelo, incluyendo precisión, matriz de confusión y reporte de clasificación.

13. **main():**
    - Función principal que ejecuta el flujo completo del sistema de predicción de diabetes.

## Instrucciones de uso

1. Coloca el archivo `diabetes_prediction_dataset.csv` en el mismo directorio que el código.
2. Ejecuta el script de Python:

   ```bash
   python nombre_del_archivo.py
   ```

3. El sistema te guiará para entrenar el modelo y realizar predicciones. Deberás proporcionar el número de épocas y la tasa de aprendizaje. Luego, podrás ingresar datos de nuevos pacientes para realizar predicciones.

4. Los resultados del modelo, incluyendo la precisión y otros indicadores, se mostrarán en la consola.

## Notas

- Este código implementa un modelo de red neuronal básico con una capa oculta y utiliza la función de activación sigmoide.
- El código está diseñado para fines educativos y puede mejorarse con técnicas avanzadas de optimización y preprocesamiento de datos.