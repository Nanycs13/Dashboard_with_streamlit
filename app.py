import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Lung Cancer Risk Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("lung_cancer_dataset.csv")

df = load_data()

st.sidebar.title("Filtros")


age_range = st.sidebar.slider("Idade", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
gender_filter = st.sidebar.multiselect("Gênero", df["gender"].unique(), default=df["gender"].unique())
smoke_filter = st.sidebar.slider("Histórico de Tabagismo (Pack Years)", int(df["pack_years"].min()), int(df["pack_years"].max()), (int(df["pack_years"].min()), int(df["pack_years"].max())))
family_history_filter = st.sidebar.multiselect("Histórico Familiar", df["family_history"].unique(), default=df["family_history"].unique())
target_filter = st.sidebar.multiselect("Diagnóstico de Câncer de Pulmão", df["lung_cancer"].unique(), default=df["lung_cancer"].unique())


filtered = df[
    (df["age"].between(age_range[0], age_range[1])) &
    (df["pack_years"].between(smoke_filter[0], smoke_filter[1])) &
    (df["gender"].isin(gender_filter)) &
    (df["family_history"].isin(family_history_filter)) &
    (df["lung_cancer"].isin(target_filter))
]

st.title("Lung Cancer Risk Dashboard")

menu = st.sidebar.radio("Navegação", ["Visão Geral", "Visualizações"])


if menu == "Visão Geral":
    st.subheader("Visão Geral dos Pacientes")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Pacientes", len(filtered))
    col2.metric("Positivos", (filtered["lung_cancer"] == "YES").sum())
    col3.metric("Negativos", (filtered["lung_cancer"] == "NO").sum())
    st.dataframe(filtered.head(20))


elif menu == "Visualizações":
    st.subheader("Distribuição da Idade por Diagnóstico")
    fig1 = px.histogram(filtered, x="age", nbins=30, color="lung_cancer", barmode="overlay", title="Distribuição da Idade (Lung Cancer)")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Histórico de Tabagismo")
    fig2 = px.histogram(filtered, x="pack_years", nbins=30, color="lung_cancer", barmode="overlay", title="Distribuição de Pack Years")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Distribuição por Gênero")
    gender_counts = filtered["gender"].value_counts().reset_index()
    gender_counts.columns = ["Gênero", "Quantidade"]
    fig3 = px.bar(gender_counts, x="Gênero", y="Quantidade", color="Gênero", title="Distribuição por Gênero")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Exposições Ambientais"):
        exp_vars = ["radon_exposure", "asbestos_exposure", "secondhand_smoke_exposure"]
        for var in exp_vars:
            counts = filtered[var].value_counts().reset_index()
            counts.columns = [var, "Quantidade"]
            fig = px.bar(counts, x=var, y="Quantidade", color=var, title=f"Distribuição de {var.replace('_',' ').title()}")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Diagnóstico de Câncer de Pulmão"):
        fig4 = px.pie(filtered, names="lung_cancer", title="Distribuição Lung Cancer")
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Correlação entre Variáveis Numéricas"):
        corr = filtered[["age","pack_years"]].corr()
        fig5 = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlação")
        st.plotly_chart(fig5, use_container_width=True)
