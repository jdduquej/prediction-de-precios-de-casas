"""
App de Streamlit para predecir el valor medio de la vivienda
a partir de variables socioeconómicas y geográficas.

IMPORTANTE:
- Este script espera que el modelo esté guardado como un archivo
  'modelo.pkl' (joblib) que sea, idealmente, un Pipeline de sklearn
  que incluya el preprocesamiento (escalado, codificación de
  'proximidad_oceano', etc.) junto con el estimador final.
- Si tu modelo NO incluye el preprocesamiento, debes replicarlo
  manualmente antes de llamar a modelo.predict() (ver sección
  "AJUSTE MANUAL" más abajo, comentada).
"""

import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Precio de Vivienda",
    page_icon="🏠",
    layout="centered",
)

# ----------------------------------------------------------------------
# Carga del modelo (con caché para no recargarlo en cada interacción)
# ----------------------------------------------------------------------
RUTA_MODELO = "modelo.pkl"


@st.cache_resource
def cargar_modelo(ruta: str):
    if not os.path.exists(ruta):
        return None
    return joblib.load(ruta)


modelo = cargar_modelo(RUTA_MODELO)

st.title("🏠 Predicción del Valor Medio de Vivienda")
st.write(
    "Ajusta las variables en el panel lateral y presiona **Predecir** "
    "para estimar el valor medio de las viviendas en esa zona."
)

if modelo is None:
    st.error(
        f"No se encontró el archivo '{RUTA_MODELO}'. "
        "Coloca el modelo entrenado (guardado con joblib) en el mismo "
        "directorio que este script."
    )
    st.stop()

# ----------------------------------------------------------------------
# Panel lateral: inputs del usuario
# ----------------------------------------------------------------------
st.sidebar.header("Variables de entrada")

longitud = st.sidebar.slider(
    "Longitud", min_value=-124.35, max_value=-114.31, value=-119.57, step=0.01
)
latitud = st.sidebar.slider(
    "Latitud", min_value=32.54, max_value=41.95, value=35.63, step=0.01
)
edad_mediana_vivienda = st.sidebar.slider(
    "Edad mediana de la vivienda (años)", min_value=1, max_value=52, value=28
)
total_habitaciones = st.sidebar.number_input(
    "Total de habitaciones", min_value=1, value=2000, step=10
)
total_dormitorios = st.sidebar.number_input(
    "Total de dormitorios", min_value=1, value=400, step=10
)
poblacion = st.sidebar.number_input(
    "Población", min_value=1, value=1000, step=10
)
hogares = st.sidebar.number_input(
    "Hogares", min_value=1, value=400, step=10
)
ingreso_mediano = st.sidebar.slider(
    "Ingreso mediano (en decenas de miles de USD)",
    min_value=0.0, max_value=15.0, value=3.5, step=0.1
)
proximidad_oceano = st.sidebar.selectbox(
    "Proximidad al océano",
    options=["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
)

# ----------------------------------------------------------------------
# Construcción del DataFrame con los nombres EXACTOS de las columnas
# usadas en el entrenamiento. Ajusta los nombres si en tu dataset
# difieren (ej: si usaste los nombres en inglés originales).
# ----------------------------------------------------------------------
datos_entrada = pd.DataFrame(
    {
        "longitud": [longitud],
        "latitud": [latitud],
        "edad_mediana_vivienda": [edad_mediana_vivienda],
        "total_habitaciones": [total_habitaciones],
        "total_dormitorios": [total_dormitorios],
        "poblacion": [poblacion],
        "hogares": [hogares],
        "ingreso_mediano": [ingreso_mediano],
        "proximidad_oceano": [proximidad_oceano],
    }
)

st.subheader("Resumen de los datos ingresados")
st.dataframe(datos_entrada, use_container_width=True)

# ----------------------------------------------------------------------
# AJUSTE MANUAL (opcional):
# Si tu modelo NO es un Pipeline y espera ya las variables escaladas
# o la columna 'proximidad_oceano' codificada en dummies, descomenta
# y adapta este bloque antes de predecir:
#
# datos_entrada = pd.get_dummies(datos_entrada, columns=["proximidad_oceano"])
# columnas_esperadas = ["longitud", "latitud", ..., "proximidad_oceano_INLAND", ...]
# for col in columnas_esperadas:
#     if col not in datos_entrada.columns:
#         datos_entrada[col] = 0
# datos_entrada = datos_entrada[columnas_esperadas]
# datos_entrada[columnas_numericas] = scaler.transform(datos_entrada[columnas_numericas])
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Predicción
# ----------------------------------------------------------------------
if st.button("🔮 Predecir valor de la vivienda", type="primary"):
    try:
        prediccion = modelo.predict(datos_entrada)
        valor_estimado = prediccion[0]
        st.success(f"💰 Valor medio estimado de la vivienda: **${valor_estimado:,.2f}**")
    except Exception as e:
        st.error(
            "Ocurrió un error al generar la predicción. Verifica que las "
            "columnas y el formato de los datos coincidan con los que "
            "espera el modelo."
        )
        st.exception(e)

st.caption(
    "Modelo entrenado con datos de viviendas en California "
    "(longitud, latitud, edad mediana de vivienda, proximidad al océano, "
    "total de habitaciones, total de dormitorios, población, hogares e "
    "ingreso mediano)."
)
