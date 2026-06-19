import streamlit as st
import pandas as pd
import joblib

# Cargar modelo entrenado
modelo = joblib.load("modelo_casas.pkl")

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Precio de Viviendas",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Predicción de Precio Medio de Viviendas")
st.markdown("Modifique las características de la vivienda y obtenga una predicción del valor estimado.")

# Sidebar
st.sidebar.header("Variables de Entrada")

longitude = st.sidebar.slider(
    "Longitud",
    min_value=-125.0,
    max_value=-113.0,
    value=-118.0
)

latitude = st.sidebar.slider(
    "Latitud",
    min_value=32.0,
    max_value=42.0,
    value=34.0
)

housing_median_age = st.sidebar.slider(
    "Edad Media de las Viviendas",
    min_value=1,
    max_value=52,
    value=30
)

total_rooms = st.sidebar.number_input(
    "Número Total de Habitaciones",
    min_value=1,
    value=2000
)

total_bedrooms = st.sidebar.number_input(
    "Número Total de Dormitorios",
    min_value=1,
    value=400
)

population = st.sidebar.number_input(
    "Población",
    min_value=1,
    value=1500
)

households = st.sidebar.number_input(
    "Número de Hogares",
    min_value=1,
    value=500
)

median_income = st.sidebar.number_input(
    "Ingreso Medio",
    min_value=0.0,
    value=5.0,
    step=0.1
)

ocean_proximity = st.sidebar.selectbox(
    "Proximidad al Océano",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

# Crear DataFrame
datos = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    'ocean_proximity': [ocean_proximity]
})

st.subheader("Datos ingresados")
st.dataframe(datos, use_container_width=True)

# Botón de predicción
if st.button("🔮 Realizar Predicción", use_container_width=True):

    prediccion = modelo.predict(datos)[0]

    st.success("Predicción realizada exitosamente")

    st.metric(
        label="Precio Medio Estimado",
        value=f"${prediccion:,.2f}"
    )

    st.balloons()
