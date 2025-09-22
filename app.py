import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Lung Cancer Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    return pd.read_csv("lung_cancer_dataset.csv")

df = load_data()

st.sidebar.title("Filtros Globais")

age_range = st.sidebar.slider("Idade", int(df["age"].min()), int(df["age"].max()),
                              (int(df["age"].min()), int(df["age"].max())))
gender_filter = st.sidebar.multiselect("Gênero", df["gender"].unique(), default=df["gender"].unique())
family_history_filter = st.sidebar.multiselect("Histórico Familiar", df["family_history"].unique(),
                                               default=df["family_history"].unique())
cancer_filter = st.sidebar.multiselect("Diagnóstico de Câncer de Pulmão", df["lung_cancer"].unique(),
                                       default=df["lung_cancer"].unique())

filtered_df = df[
    (df["age"].between(age_range[0], age_range[1])) &
    (df["gender"].isin(gender_filter)) &
    (df["family_history"].isin(family_history_filter)) &
    (df["lung_cancer"].isin(cancer_filter))
]

# Salvar dataframe filtrado no estado global (para ser usado nas páginas)
st.session_state["filtered_df"] = filtered_df

st.title("📊 Lung Cancer Risk Dashboard")
st.markdown("Use o menu lateral para navegar entre as análises.")


# --- ESTILO CUSTOMIZADO ---
st.markdown(
    """
    <style>
    /* Fundo geral */
    .stApp {
        background-color: #f9f9f9;
        color: #333333;
        font-family: "Segoe UI", sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e3d59;
        color: white;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label {
        color: white !important;
    }

    /* Títulos principais */
    h1, h2, h3 {
        color: #1e3d59;
        font-weight: 600;
    }

    /* Métricas */
    div[data-testid="stMetricValue"] {
        color: #d9534f; /* destaque em vermelho suave */
        font-size: 28px;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #555;
    }

    /* Dataframe estilizado */
    .dataframe {
        border: 1px solid #ddd;
        border-radius: 6px;
    }

    /* Expansores */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #1e3d59;
    }

    /* Botões e seletores */
    .stButton>button, .stDownloadButton>button {
        background-color: #1e3d59;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        border: none;
        font-weight: 500;
        transition: 0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #145374;
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- FIM ESTILO CUSTOMIZADO ---
