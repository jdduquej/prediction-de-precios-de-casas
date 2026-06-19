import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Precios de Vivienda",
    page_icon="🏡",
    layout="wide"
)

# ---------------------------------------------------------
# ESTADO DE SESIÓN (historial)
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("⚙️ Configuración")

st.sidebar.markdown("### Endpoint del modelo")
API_URL = st.sidebar.text_input("API URL de DataRobot", "YOUR_API_URL")
API_KEY = st.sidebar.text_input("API Key", "YOUR_API_KEY", type="password")

st.sidebar.markdown("### Opciones de visualización")
show_table = st.sidebar.checkbox("Mostrar tabla de historial", value=True)
show_chart = st.sidebar.checkbox("Mostrar gráfico de predicciones", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Desarrollado para integración con DataRobot + Streamlit Cloud.")

# ---------------------------------------------------------
# TÍTULO PRINCIPAL
# ---------------------------------------------------------
st.title("🏡 Predicción del Precio Medio de Viviendas")
st.write("Modifica las variables de entrada y obtén la predicción generada por tu modelo desplegado en DataRobot.")

# ---------------------------------------------------------
# FORMULARIO DE VARIABLES
# ---------------------------------------------------------
with st.form("input_form"):
    st.subheader("Variables de entrada")

    col1, col2 = st.columns(2)

    with col1:
        longitud = st.number_input("Longitud", value=-120.0)
        latitud = st.number_input("Latitud", value=35.0)
        edad_mediana_vivienda = st.number_input("Edad mediana de la vivienda", value=20)
        proximidad_oceano = st.selectbox(
            "Proximidad al océano",
            ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "1H OCEAN"]
        )

    with col2:
        total_habitaciones = st.number_input("Total habitaciones", value=1500)
        total_dormitorios = st.number_input("Total dormitorios", value=300)
        poblacion = st.number_input("Población", value=800)
        hogares = st.number_input("Hogares", value=300)
        ingreso_mediano = st.number_input("Ingreso mediano", value=4.5)

    submitted = st.form_submit_button("🔮 Predecir precio medio")

# ---------------------------------------------------------
# LÓGICA DE PREDICCIÓN
# ---------------------------------------------------------
if submitted:
    inputs = {
        "longitud": longitud,
        "latitud": latitud,
        "edad_mediana_vivienda": edad_mediana_vivienda,
        "proximidad_oceano": proximidad_oceano,
        "total_habitaciones": total_habitaciones,
        "total_dormitorios": total_dormitorios,
        "poblacion": poblacion,
        "hogares": hogares,
        "ingreso_mediano": ingreso_mediano
    }

    st.subheader("Resultado de la predicción")

    if not API_URL or API_URL == "YOUR_API_URL":
        st.error("Configura un API URL válido en el sidebar.")
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}" if API_KEY else ""
        }

        with st.spinner("Consultando modelo en DataRobot..."):
            try:
                # Ajusta el payload según tu despliegue en DataRobot
                payload = {"data": [inputs]}
                response = requests.post(API_URL, json=payload, headers=headers)

                if response.status_code == 200:
                    prediction = response.json()

                    # Ajusta esta línea al formato real de respuesta
                    pred_value = prediction["data"][0]["prediction"]

                    st.success(f"🏠 **Precio medio estimado:** ${pred_value:,.2f}")

                    # Guardar en historial
                    record = inputs.copy()
                    record["prediccion_precio_medio"] = pred_value
                    st.session_state.history.append(record)

                else:
                    st.error(f"Error en la API: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error("Error al conectar con el modelo.")
                st.write(str(e))

# ---------------------------------------------------------
# HISTORIAL DE PREDICCIONES
# ---------------------------------------------------------
st.subheader("Historial de predicciones")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)

    if show_table:
        st.dataframe(df_history)

    if show_chart and "prediccion_precio_medio" in df_history.columns:
        st.subheader("Gráfico de precios predichos")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_history["prediccion_precio_medio"], marker="o")
        ax.set_xlabel("Número de predicción")
        ax.set_ylabel("Precio medio predicho")
        ax.set_title("Evolución de las predicciones")
        st.pyplot(fig)
else:
    st.info("Aún no hay predicciones en el historial. Realiza una para comenzar.")
