import streamlit as st
import plotly.express as px

df = st.session_state["filtered_df"]

st.title("👤 Perfil dos Pacientes")

st.subheader("Distribuição da Idade por Diagnóstico")
fig1 = px.histogram(df, x="age", nbins=30, color="lung_cancer", barmode="overlay",
                    title="Distribuição de Idade vs Câncer")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Distribuição por Gênero")
gender_counts = df["gender"].value_counts().reset_index()
gender_counts.columns = ["Gênero", "Quantidade"]
fig2 = px.bar(gender_counts, x="Gênero", y="Quantidade", color="Gênero",
              title="Distribuição por Gênero")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Histórico Familiar")
family_counts = df["family_history"].value_counts().reset_index()
family_counts.columns = ["Histórico Familiar", "Quantidade"]
fig3 = px.pie(family_counts, names="Histórico Familiar", values="Quantidade",
              title="Proporção de Histórico Familiar")
st.plotly_chart(fig3, use_container_width=True)
