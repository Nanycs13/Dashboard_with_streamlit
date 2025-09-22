import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os

# Configuração da página
st.set_page_config(
    page_title="Lung Cancer Analytics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para carregar dados com fallback para dados de exemplo
@st.cache_data
def load_data():
    try:
        # Tentar carregar dados do CSV
        if os.path.exists("lung_cancer_dataset.csv"):
            df = pd.read_csv("lung_cancer_dataset.csv")
            st.success("✅ Dados carregados com sucesso!")
        else:
            # Gerar dados de exemplo se o arquivo não existir
            st.warning("📊 Arquivo 'lung_cancer_dataset.csv' não encontrado. Gerando dados de exemplo...")
            np.random.seed(42)
            n_samples = 500
            
            data = {
                'patient_id': range(100000, 100000 + n_samples),
                'age': np.clip(np.random.normal(65, 12, n_samples).astype(int), 30, 95),
                'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
                'pack_years': np.clip(np.random.exponential(15, n_samples), 0, 100),
                'radon_exposure': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.6, 0.3, 0.1]),
                'asbestos_exposure': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
                'secondhand_smoke_exposure': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
                'copd_diagnosis': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
                'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
                'family_history': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
            }
            
            # Calcular probabilidade de câncer baseada nos fatores de risco
            risk_factors = (
                (data['age'] > 60).astype(int) * 0.3 +
                (data['pack_years'] > 20).astype(int) * 0.4 +
                (data['radon_exposure'] == 'High').astype(int) * 0.2 +
                (data['asbestos_exposure'] == 'Yes').astype(int) * 0.3 +
                (data['secondhand_smoke_exposure'] == 'Yes').astype(int) * 0.1 +
                (data['copd_diagnosis'] == 'Yes').astype(int) * 0.3 +
                (data['alcohol_consumption'] == 'Heavy').astype(int) * 0.1 +
                (data['family_history'] == 'Yes').astype(int) * 0.2
            )
            
            # Gerar diagnóstico de câncer baseado na probabilidade
            cancer_prob = np.clip(risk_factors * 0.15, 0, 0.8)
            data['lung_cancer'] = np.random.binomial(1, cancer_prob).astype(str)
            data['lung_cancer'] = data['lung_cancer'].replace({'1': 'Yes', '0': 'No'})
            
            df = pd.DataFrame(data)
            st.info(f"📋 Gerados {n_samples} registros de exemplo")
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        # Gerar dados mínimos de fallback
        df = pd.DataFrame({
            'patient_id': [100000, 100001],
            'age': [69, 55],
            'gender': ['Male', 'Female'],
            'pack_years': [66.0, 12.5],
            'radon_exposure': ['High', 'Low'],
            'asbestos_exposure': ['No', 'Yes'],
            'secondhand_smoke_exposure': ['No', 'Yes'],
            'copd_diagnosis': ['Yes', 'No'],
            'alcohol_consumption': ['Moderate', 'Light'],
            'family_history': ['No', 'Yes'],
            'lung_cancer': ['No', 'Yes']
        })
    
    # Processar dados
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(65)
    df['pack_years'] = pd.to_numeric(df['pack_years'], errors='coerce').fillna(0)
    
    # Criar colunas derivadas para análise
    age_bins = [0, 40, 60, 80, 100]
    age_labels = ['<40', '40-60', '60-80', '80+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    # Calcular score de risco (versão simplificada)
    df['risk_score'] = (
        df['pack_years'] / 20 + 
        (df['radon_exposure'] == 'High').astype(int) * 1.5 +
        (df['asbestos_exposure'] == 'Yes').astype(int) * 1.5 +
        (df['secondhand_smoke_exposure'] == 'Yes').astype(int) * 0.5 +
        (df['copd_diagnosis'] == 'Yes').astype(int) * 2.0 +
        (df['alcohol_consumption'] == 'Heavy').astype(int) * 0.5 +
        (df['family_history'] == 'Yes').astype(int) * 1.0
    )
    
    # Classificar risco
    df['risk_category'] = pd.cut(df['risk_score'], 
                                bins=[0, 2, 4, 10], 
                                labels=['Baixo', 'Médio', 'Alto'])
    
    return df

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .section-header {
        color: #2c3e50;
        border-left: 5px solid #3498db;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
        font-size: 1.3rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Filtros globais na sidebar
def create_sidebar_filters(df):
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtros Globais")
    
    # Filtros principais
    gender_filter = st.sidebar.multiselect(
        "Gênero",
        options=sorted(df['gender'].unique()),
        default=sorted(df['gender'].unique())
    )
    
    age_filter = st.sidebar.slider(
        "Idade",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=(int(df['age'].min()), int(df['age'].max()))
    )
    
    cancer_filter = st.sidebar.multiselect(
        "Diagnóstico de Câncer",
        options=sorted(df['lung_cancer'].unique()),
        default=sorted(df['lung_cancer'].unique())
    )
    
    risk_filter = st.sidebar.slider(
        "Escala de Risco",
        min_value=float(df['risk_score'].min()),
        max_value=float(df['risk_score'].max()),
        value=(float(df['risk_score'].min()), float(df['risk_score'].max())),
        step=0.1
    )
    
    # Filtros adicionais
    st.sidebar.markdown("**Filtros Adicionais**")
    smoke_filter = st.sidebar.multiselect(
        "Exposição à Fumaça",
        options=sorted(df['secondhand_smoke_exposure'].unique()),
        default=sorted(df['secondhand_smoke_exposure'].unique())
    )
    
    copd_filter = st.sidebar.multiselect(
        "Diagnóstico de COPD",
        options=sorted(df['copd_diagnosis'].unique()),
        default=sorted(df['copd_diagnosis'].unique())
    )
    
    return {
        'gender': gender_filter,
        'age': age_filter,
        'lung_cancer': cancer_filter,
        'risk_score': risk_filter,
        'smoke': smoke_filter,
        'copd': copd_filter
    }

# Aplicar filtros
def apply_filters(df, filters):
    df_filtered = df.copy()
    
    if filters['gender']:
        df_filtered = df_filtered[df_filtered['gender'].isin(filters['gender'])]
    
    if filters['lung_cancer']:
        df_filtered = df_filtered[df_filtered['lung_cancer'].isin(filters['lung_cancer'])]
    
    if filters['smoke']:
        df_filtered = df_filtered[df_filtered['secondhand_smoke_exposure'].isin(filters['smoke'])]
    
    if filters['copd']:
        df_filtered = df_filtered[df_filtered['copd_diagnosis'].isin(filters['copd'])]
    
    df_filtered = df_filtered[
        (df_filtered['age'] >= filters['age'][0]) & 
        (df_filtered['age'] <= filters['age'][1])
    ]
    
    df_filtered = df_filtered[
        (df_filtered['risk_score'] >= filters['risk_score'][0]) & 
        (df_filtered['risk_score'] <= filters['risk_score'][1])
    ]
    
    return df_filtered

# Página inicial
def render_homepage(df):
    st.markdown('<h1 class="main-header">🫁 Lung Cancer Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Informações sobre os dados
    with st.expander("ℹ️ Sobre os Dados", expanded=False):
        st.markdown(f"""
        <div class="info-box">
        <strong>Dataset Info:</strong> {len(df)} pacientes | {df['lung_cancer'].value_counts().get('Yes', 0)} com câncer 
        | Idade média: {df['age'].mean():.1f} anos
        </div>
        """, unsafe_allow_html=True)
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(df)
        st.markdown(f'''
        <div class="metric-card">
            <h3>👥 Total</h3>
            <div style="font-size: 2rem; font-weight: bold;">{total_patients}</div>
            <div>Pacientes</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        cancer_rate = (df['lung_cancer'] == 'Yes').mean() * 100
        st.markdown(f'''
        <div class="metric-card">
            <h3>🎯 Taxa de Câncer</h3>
            <div style="font-size: 2rem; font-weight: bold;">{cancer_rate:.1f}%</div>
            <div>Prevalência</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_age = df['age'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h3>📊 Idade Média</h3>
            <div style="font-size: 2rem; font-weight: bold;">{avg_age:.1f}</div>
            <div>Anos</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        high_risk = (df['risk_category'] == 'Alto').mean() * 100
        st.markdown(f'''
        <div class="metric-card">
            <h3>⚠️ Risco Alto</h3>
            <div style="font-size: 2rem; font-weight: bold;">{high_risk:.1f}%</div>
            <div>dos Pacientes</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Primeira linha de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Distribuição por Idade e Diagnóstico</h3>', unsafe_allow_html=True)
        fig = px.histogram(df, x='age', color='lung_cancer', 
                          nbins=20, barmode='overlay', opacity=0.7,
                          title='Distribuição de Idade por Diagnóstico',
                          color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'})
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">Fatores de Risco por Gênero</h3>', unsafe_allow_html=True)
        
        # Preparar dados para fatores de risco categóricos
        risk_data = []
        factors = ['radon_exposure', 'asbestos_exposure', 'copd_diagnosis', 'family_history']
        
        for factor in factors:
            cross_tab = pd.crosstab(df[factor], df['gender'], normalize='index') * 100
            for category in cross_tab.index:
                for gender in cross_tab.columns:
                    risk_data.append({
                        'Fator': f"{factor.replace('_', ' ').title()} - {category}",
                        'Gênero': gender,
                        'Percentual': cross_tab.loc[category, gender]
                    })
        
        risk_df = pd.DataFrame(risk_data)
        if not risk_df.empty:
            fig = px.bar(risk_df, x='Fator', y='Percentual', color='Gênero',
                        barmode='group', title='Distribuição de Fatores de Risco por Gênero (%)')
            st.plotly_chart(fig, use_container_width=True)
    
    # Segunda linha de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Relação Idade vs Maços/Ano</h3>', unsafe_allow_html=True)
        fig = px.scatter(df, x='age', y='pack_years', color='lung_cancer',
                        size='risk_score', hover_data=['gender', 'risk_category'],
                        title='Relação entre Idade e Consumo de Cigarros',
                        color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">Categorias de Risco</h3>', unsafe_allow_html=True)
        risk_counts = df['risk_category'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title='Distribuição por Categoria de Risco')
        st.plotly_chart(fig, use_container_width=True)

# Página de análise detalhada
def render_analysis(df):
    st.markdown('<h1 class="main-header">📊 Análise Detalhada</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<h3 class="section-header">Configurações</h3>', unsafe_allow_html=True)
        
        # Seletores de eixos
        numeric_columns = ['age', 'pack_years', 'risk_score']
        x_axis = st.selectbox("Eixo X", numeric_columns, index=0)
        y_axis = st.selectbox("Eixo Y", numeric_columns, index=1)
        
        # Seletores de cor e tamanho
        color_options = ['lung_cancer', 'gender', 'copd_diagnosis', 'risk_category', 'radon_exposure']
        color_by = st.selectbox("Colorir por", color_options, index=0)
        
        size_by = st.selectbox("Tamanho por", ['Nenhum'] + numeric_columns, index=0)
        
        # Opções de visualização
        st.markdown("---")
        st.markdown("**Opções de Visualização**")
        chart_type = st.radio("Tipo de Gráfico", ["Dispersão", "Boxplot", "Histograma"])
    
    with col2:
        try:
            if chart_type == "Dispersão":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                               size=None if size_by == 'Nenhum' else size_by,
                               hover_data=['patient_id'],
                               title=f'Relação entre {x_axis} e {y_axis}')
                
            elif chart_type == "Boxplot":
                fig = px.box(df, x=color_by, y=y_axis, 
                           title=f'Distribuição de {y_axis} por {color_by}')
                
            else:  # Histograma
                fig = px.histogram(df, x=x_axis, color=color_by, barmode='overlay',
                                 title=f'Distribuição de {x_axis} por {color_by}')
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao criar gráfico: {e}")
            st.info("Tente ajustar as configurações do gráfico")
    
    # Análise de correlação
    st.markdown("---")
    st.markdown('<h3 class="section-header">Análise de Correlação</h3>', unsafe_allow_html=True)
    
    # Selecionar apenas colunas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       title='Matriz de Correlação entre Variáveis Numéricas',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar correlações mais fortes
        st.markdown("**Correlações Significativas (|r| > 0.3):**")
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:
                    strong_corrs.append({
                        'Variável 1': corr_matrix.columns[i],
                        'Variável 2': corr_matrix.columns[j],
                        'Correlação': f"{corr_val:.3f}"
                    })
        
        if strong_corrs:
            st.dataframe(pd.DataFrame(strong_corrs), use_container_width=True)
        else:
            st.info("Não foram encontradas correlações fortes entre as variáveis numéricas")

# Página de relatórios
def render_reports(df):
    st.markdown('<h1 class="main-header">📋 Relatórios e Dados</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dados Completos", "📈 Estatísticas", "💾 Exportar Dados"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Dataset Completo</h3>', unsafe_allow_html=True)
        
        # Filtros rápidos na tabela
        col1, col2 = st.columns(2)
        with col1:
            rows_to_show = st.slider("Linhas por página", 10, 100, 20)
        with col2:
            search_term = st.text_input("🔍 Pesquisar na tabela")
        
        display_df = df.copy()
        if search_term:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            display_df = display_df[mask]
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        st.markdown(f"**Total de registros:** {len(display_df)}")
    
    with tab2:
        st.markdown('<h3 class="section-header">Estatísticas Descritivas</h3>', unsafe_allow_html=True)
        
        # Estatísticas numéricas
        st.markdown("**Variáveis Numéricas:**")
        numeric_stats = df.select_dtypes(include=[np.number]).describe()
        st.dataframe(numeric_stats, use_container_width=True)
        
        # Estatísticas por grupo
        st.markdown("**Estatísticas por Diagnóstico de Câncer:**")
        if 'lung_cancer' in df.columns:
            grouped_stats = df.groupby('lung_cancer').agg({
                'age': ['mean', 'std', 'min', 'max'],
                'pack_years': ['mean', 'std', 'min', 'max'],
                'risk_score': ['mean', 'std', 'min', 'max']
            }).round(2)
            st.dataframe(grouped_stats, use_container_width=True)
    
    with tab3:
        st.markdown('<h3 class="section-header">Exportar Dados</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Opções de Exportação:</strong> Você pode exportar os dados filtrados ou completos em formato CSV.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.radio("Formato", ["CSV", "JSON"])
            include_filters = st.checkbox("Incluir apenas dados filtrados", value=True)
        
        with col2:
            filename = st.text_input("Nome do arquivo", "lung_cancer_data")
            include_timestamp = st.checkbox("Incluir timestamp", value=True)
        
        export_df = df if include_filters else load_data()
        
        if st.button("📥 Gerar Arquivo para Download", type="primary"):
            try:
                if include_timestamp:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{filename}_{timestamp}"
                
                if export_format == "CSV":
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
                else:
                    json_data = export_df.to_json(indent=2, orient='records')
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_data,
                        file_name=f"{filename}.json",
                        mime="application/json"
                    )
                
                st.success("✅ Arquivo gerado com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro ao gerar arquivo: {e}")

# Carregar dados
df = load_data()

# Sidebar principal
with st.sidebar:
    st.title("🫁 Lung Cancer Analytics")
    st.markdown("*Dashboard de análise de dados de câncer de pulmão*")
    st.markdown("---")
    
    # Menu de navegação
    selected = option_menu(
        menu_title="Navegação Principal",
        options=["Dashboard", "Análise", "Relatórios"],
        icons=["speedometer2", "graph-up-arrow", "file-earmark-text"],
        menu_icon="app-indicator",
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px"},
            "nav-link-selected": {"background-color": "#2c3e50"},
        }
    )

# Aplicar filtros globais
filters = create_sidebar_filters(df)
df_filtered = apply_filters(df, filters)

# Mostrar estatísticas de filtragem na sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Estatísticas dos Filtros")
st.sidebar.markdown(f"**Pacientes visíveis:** {len(df_filtered):,} / {len(df):,}")
st.sidebar.markdown(f"**Taxa de câncer:** {(df_filtered['lung_cancer'] == 'Yes').mean()*100:.1f}%")

if len(df_filtered) < len(df):
    st.sidebar.progress(len(df_filtered) / len(df))

# Navegação entre páginas
if selected == "Dashboard":
    render_homepage(df_filtered)
elif selected == "Análise":
    render_analysis(df_filtered)
elif selected == "Relatórios":
    render_reports(df_filtered)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.markdown("🫁 *Lung Cancer Analytics v2.0*  \n*Desenvolvido com Streamlit*")