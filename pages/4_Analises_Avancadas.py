import streamlit as st
import plotly.express as px

df = st.session_state["filtered_df"]

st.title("🧮 Análises Avançadas")

st.subheader("Correlação entre Variáveis Numéricas")
corr = df[["age","pack_years"]].corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlação")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Distribuição do Diagnóstico")
fig2 = px.pie(df, names="lung_cancer", title="Proporção Diagnóstico Câncer de Pulmão")
st.plotly_chart(fig2, use_container_width=True)
