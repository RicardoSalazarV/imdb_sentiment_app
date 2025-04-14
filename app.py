import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuración
st.set_page_config(page_title="Clasificador IMDB", layout="centered")
st.title("Clasificador de Sentimientos - Reseñas IMDB")

# Cargar modelo y vectorizador
model = joblib.load("model/model_lr.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Entrada de reseña
st.header("Ingresa una reseña de película")
user_input = st.text_area("Reseña:", height=150)

if st.button("Clasificar"):
    if user_input.strip() == "":
        st.warning("Por favor, escribe una reseña.")
    else:
        clean_input = user_input.lower()
        clean_input = ''.join([c if c.isalnum() or c.isspace() else ' ' for c in clean_input])
        vectorized_input = vectorizer.transform([clean_input])
        proba = model.predict_proba(vectorized_input)[0][1]
        pred = model.predict(vectorized_input)[0]
        st.subheader("Resultado")
        st.markdown(f"**Clasificación:** {'Positiva' if pred == 1 else 'Negativa'}")
        st.markdown(f"**Probabilidad de ser positiva:** {proba:.2f}")

        if pred == 1:
            st.success("Buena reseña detectada.")
        else:
            st.error("Crítica negativa detectada.")

# Mostrar métricas y gráficos
# Mostramos las métricas promedio macro (buen punto de referencia)
if "macro avg" in metrics:
    macro_avg = metrics["macro avg"]
    metrics_df = pd.DataFrame([macro_avg])[["precision", "recall", "f1-score"]]
    st.dataframe(metrics_df.style.highlight_max(axis=1))
else:
    st.warning("No se encontraron métricas promedio.")
st.header("Métricas del Modelo")

with open("model/metrics.json", "r") as f:
    metrics = json.load(f)

metrics_df = pd.DataFrame(metrics).T.loc[["macro avg"]][["precision", "recall", "f1-score"]]
st.dataframe(metrics_df.style.highlight_max(axis=0))

# Matriz de confusión
st.subheader("Matriz de Confusión")
df = pd.read_csv("data/imdb_reviews_clean.csv")
X = vectorizer.transform(df["review_clean"])
y_true = df["label"]
y_pred = model.predict(X)

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)