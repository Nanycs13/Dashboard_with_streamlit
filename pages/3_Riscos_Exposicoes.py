import streamlit as st
import plotly.express as px

df = st.session_state["filtered_df"]

st.title("🚬 Riscos & Exposições")

st.subheader("Histórico de Tabagismo (Pack Years)")
fig1 = px.histogram(df, x="pack_years", nbins=30, color="lung_cancer", barmode="overlay",
                    title="Distribuição de Pack Years vs Diagnóstico")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Exposições Ambientais")
exp_vars = ["radon_exposure", "asbestos_exposure", "secondhand_smoke_exposure"]
for var in exp_vars:
    counts = df[var].value_counts().reset_index()
    counts.columns = [var, "Quantidade"]
    fig = px.bar(counts, x=var, y="Quantidade", color=var,
                 title=f"Distribuição de {var.replace('_',' ').title()}")
    st.plotly_chart(fig, use_container_width=True)
