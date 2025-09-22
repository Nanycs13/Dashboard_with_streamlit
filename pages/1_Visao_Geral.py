import streamlit as st

df = st.session_state["filtered_df"]

st.title("📌 Visão Geral dos Pacientes")

col1, col2, col3 = st.columns(3)
col1.metric("Total de Pacientes", len(df))
col2.metric("Casos Positivos", (df["lung_cancer"] == "YES").sum())
col3.metric("Casos Negativos", (df["lung_cancer"] == "NO").sum())

st.subheader("Amostra dos Dados Filtrados")
st.dataframe(df.head(20))
