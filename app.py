"""
App de Streamlit para predecir el valor medio de la vivienda
a partir de variables socioeconómicas y geográficas.

IMPORTANTE:
- El modelo se descarga en tiempo de ejecución desde un repositorio
  público de GitHub (URL raw) y se carga en memoria con joblib.
  La URL y el token de GitHub se leen desde st.secrets (ver más abajo).
- Idealmente el archivo es un Pipeline de sklearn que incluye el
  preprocesamiento (escalado, codificación de 'proximidad_oceano', etc.)
  junto con el estimador final.
- Si tu modelo NO incluye el preprocesamiento, debes replicarlo
  manualmente antes de llamar a modelo.predict() (ver sección
  "AJUSTE MANUAL" más abajo, comentada).
"""

import streamlit as st
import pandas as pd
import joblib
import requests
import io

# ----------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Precio de Vivienda",
    page_icon="🏠",
    layout="centered",
)

# ----------------------------------------------------------------------
# Configuración de descarga del modelo desde GitHub
# ----------------------------------------------------------------------
# En Streamlit Cloud: Settings -> Secrets, agrega algo como:
#
#   GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"
#   MODEL_URL = "https://raw.githubusercontent.com/usuario/repositorio/main/modelo.pkl"
#
# Si corres la app localmente, crea un archivo .streamlit/secrets.toml
# con el mismo contenido.

MODEL_URL = st.secrets.get(
    "MODEL_URL",
    "https://raw.githubusercontent.com/usuario/repositorio/main/modelo.pkl",
)
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", None)


@st.cache_resource
def cargar_modelo(url: str, token: str):
    """Descarga el modelo desde GitHub (raw) y lo carga con joblib."""
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        respuesta = requests.get(url, headers=headers, timeout=30)
        respuesta.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo descargar el modelo desde GitHub: {e}")
        return None

    try:
        modelo_cargado = joblib.load(io.BytesIO(respuesta.content))
    except Exception as e:
        st.error(f"El archivo se descargó pero no pudo cargarse con joblib: {e}")
        return None

    return modelo_cargado


with st.spinner("Descargando y cargando el modelo desde GitHub..."):
    modelo = cargar_modelo(MODEL_URL, GITHUB_TOKEN)

st.title("🏠 Predicción del Valor Medio de Vivienda")
st.write(
    "Ajusta las variables en el panel lateral y presiona **Predecir** "
    "para estimar el valor medio de las viviendas en esa zona."
)

if modelo is None:
    st.error(
        "No se pudo cargar el modelo. Revisa que el secreto 'MODEL_URL' "
        "apunte a la URL raw correcta del archivo en GitHub y que el "
        "archivo sea un objeto válido guardado con joblib."
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
