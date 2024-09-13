import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

# Título de la aplicación
st.title("Análisis de Datos y Modelado Predictivo")

# Subida de archivo CSV o Excel
uploaded_file = st.file_uploader("Sube tus datos (CSV o Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Leer el archivo dependiendo de su formato
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    st.write("Datos cargados:")
    st.write(df.head())

    # Permitir al usuario identificar la columna de fechas si existe
    date_column = st.selectbox("Selecciona la columna de fechas (si aplica)", [None] + df.columns.tolist())
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        st.write(f"Columna de fecha seleccionada: {date_column}")

    # Permitir al usuario seleccionar la variable dependiente (objetivo)
    target = st.selectbox("Selecciona la variable objetivo (dependiente)", df.columns)

    # Determinar las variables independientes automáticamente
    features = df.drop(columns=[target]).columns.tolist()

    if date_column:
        # Eliminar la columna de fecha si está en las características (features)
        if date_column in features:
            features.remove(date_column)

    st.write(f"Variables independientes detectadas: {features}")

    if len(features) > 0 and target:
        X = df[features]
        y = df[target]

        # Tomar la última observación como dato de prueba (X_test, y_test)
        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        X_test = X.iloc[-1:]
        y_test = y.iloc[-1]

        st.write("Modelos de Predicción Tradicionales")

        # Diccionario para almacenar predicciones y modelos
        predictions_dict = {}

        # Entrenamiento y predicción de los modelos tradicionales
        # Regresión Lineal
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        prediction_lr = model_lr.predict(X_test)[0]
        predictions_dict["Regresión Lineal"] = prediction_lr
        st.write(f"Regresión Lineal: Predicción: {prediction_lr}")

        # Árbol de Decisión
        model_dt = DecisionTreeRegressor()
        model_dt.fit(X_train, y_train)
        prediction_dt = model_dt.predict(X_test)[0]
        predictions_dict["Árbol de Decisión"] = prediction_dt
        st.write(f"Árbol de Decisión: Predicción: {prediction_dt}")

        # Bosque Aleatorio
        model_rf = RandomForestRegressor()
        model_rf.fit(X_train, y_train)
        prediction_rf = model_rf.predict(X_test)[0]
        predictions_dict["Bosque Aleatorio"] = prediction_rf
        st.write(f"Bosque Aleatorio: Predicción: {prediction_rf}")

        # K-Nearest Neighbors
        model_knn = KNeighborsRegressor(n_neighbors=5)
        model_knn.fit(X_train, y_train)
        prediction_knn = model_knn.predict(X_test)[0]
        predictions_dict["K-Nearest Neighbors"] = prediction_knn
        st.write(f"K-Nearest Neighbors: Predicción: {prediction_knn}")

        # Regresión Polinómica
        poly = PolynomialFeatures(degree=2)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
        model_poly = LinearRegression()
        model_poly.fit(X_poly_train, y_train)
        prediction_poly = model_poly.predict(X_poly_test)[0]
        predictions_dict["Regresión Polinómica"] = prediction_poly
        st.write(f"Regresión Polinómica: Predicción: {prediction_poly}")

        # Comparación de los valores reales vs estimados
        st.write("### Comparación de Dato Real vs Dato Estimado")

        for model_name, prediction in predictions_dict.items():
            st.write(f"**{model_name}:**")
            st.write(f"Valor Real: {y_test}")
            st.write(f"Predicción: {prediction}")

            # Cálculo de las métricas
            mae = mean_absolute_error([y_test], [prediction])
            mse = mean_squared_error([y_test], [prediction])
            mape = mean_absolute_percentage_error([y_test], [prediction])
            diff_percent = abs((y_test - prediction) / y_test) * 100

            st.write(f"Error Absoluto Medio (MAE): {mae:.4f}")
            st.write(f"Error Cuadrático Medio (MSE): {mse:.4f}")
            st.write(f"Porcentaje de Diferencia: {diff_percent:.2f}%")
            st.write("---")

        # Sección de predicción futura con datos nuevos
        st.write("### Predicción de un valor no conocido")
        new_data = []
        for feature in features:
            value = st.number_input(f"Introduce un valor para {feature}")
            new_data.append(value)

        # Pregunta al usuario por la fecha si es una serie temporal
        if date_column:
            future_date = st.date_input("Selecciona una fecha futura para predecir")

        new_data = np.array(new_data).reshape(1, -1)

        if st.button("Predecir con nuevos valores"):
            for model_name, model in zip(predictions_dict.keys(), [model_lr, model_dt, model_rf, model_knn, model_poly]):
                if model_name == "Regresión Polinómica":
                    new_data_poly = poly.transform(new_data)
                    prediction = model.predict(new_data_poly)[0]
                else:
                    prediction = model.predict(new_data)[0]

                st.write(f"Predicción con {model_name}: {prediction}")

        # Modelos de series temporales (ARIMA, ETS, SARIMA)
        if date_column:
            st.write("Modelos de Series Temporales")

            # ARIMA
            st.write("Entrenando ARIMA manualmente...")
            model_arima = ARIMA(df[target], order=(1, 1, 2))
            model_fit_arima = model_arima.fit()
            forecast_arima = model_fit_arima.forecast(steps=1)
            st.write(f"ARIMA: Predicción: {forecast_arima.iloc[0]}")

            # ETS
            st.write("Entrenando ETS...")
            model_ets = ExponentialSmoothing(df[target], trend="add", seasonal="add", seasonal_periods=12)
            model_fit_ets = model_ets.fit()
            forecast_ets = model_fit_ets.forecast(steps=1)
            st.write(f"ETS: Predicción: {forecast_ets.iloc[0]}")

            # SARIMA
            st.write("Entrenando SARIMA...")
            model_sarima = SARIMAX(df[target], order=(1, 1, 2), seasonal_order=(1, 1, 1, 12))
            model_fit_sarima = model_sarima.fit()
            forecast_sarima = model_fit_sarima.forecast(steps=1)
            st.write(f"SARIMA: Predicción: {forecast_sarima.iloc[0]}")
