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
    page_title="Análise de Câncer de Pulmão",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dicionários de tradução
TRADUCOES = {
    # Colunas
    'patient_id': 'ID do Paciente',
    'age': 'Idade',
    'gender': 'Gênero',
    'pack_years': 'Anos-Maço',
    'radon_exposure': 'Exposição ao Radônio',
    'asbestos_exposure': 'Exposição ao Amianto',
    'secondhand_smoke_exposure': 'Exposição ao Fumo Passivo',
    'copd_diagnosis': 'Diagnóstico de DPOC',
    'alcohol_consumption': 'Consumo de Álcool',
    'family_history': 'Histórico Familiar',
    'lung_cancer': 'Câncer de Pulmão',
    'risk_score': 'Escala de Risco',
    'risk_category': 'Categoria de Risco',
    'age_group': 'Faixa Etária',
    
    # Valores
    'Male': 'Masculino',
    'Female': 'Feminino',
    'Yes': 'Sim',
    'No': 'Não',
    'High': 'Alta',
    'Medium': 'Média',
    'Low': 'Baixa',
    'None': 'Nenhum',
    'Light': 'Leve',
    'Moderate': 'Moderado',
    'Heavy': 'Pesado',
    'Baixo': 'Baixo',
    'Médio': 'Médio',
    'Alto': 'Alto'
}

# Explicações das variáveis
EXPLICACOES = {
    'pack_years': 'Medida do consumo de cigarro: 1 ano-maço = fumar 1 maço por dia durante 1 ano',
    'risk_score': 'Pontuação de risco calculada com base em todos os fatores de risco combinados',
    'radon_exposure': 'Exposição ao gás radônio, um fator de risco conhecido para câncer de pulmão',
    'asbestos_exposure': 'Exposição ao amianto, material associado a doenças pulmonares',
    'copd_diagnosis': 'Doença Pulmonar Obstrutiva Crônica (DPOC)'
}

@st.cache_data
def load_data():
    df = pd.read_csv("lung_cancer_dataset.csv")
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(65)
    df['pack_years'] = pd.to_numeric(df['pack_years'], errors='coerce').fillna(0)
    
    # Traduzir valores categóricos
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(lambda x: TRADUCOES.get(x, x))
    
    age_bins = [0, 40, 60, 80, 100]
    age_labels = ['<40', '40-60', '60-80', '80+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    df['risk_score'] = (
        df['pack_years'] / 20 + 
        (df['radon_exposure'] == 'Alta').astype(int) * 1.5 +
        (df['asbestos_exposure'] == 'Sim').astype(int) * 1.5 +
        (df['secondhand_smoke_exposure'] == 'Sim').astype(int) * 0.5 +
        (df['copd_diagnosis'] == 'Sim').astype(int) * 2.0 +
        (df['alcohol_consumption'] == 'Pesado').astype(int) * 0.5 +
        (df['family_history'] == 'Sim').astype(int) * 1.0
    )
    
    df['risk_category'] = pd.cut(df['risk_score'], 
                                bins=[0, 2, 4, 10], 
                                labels=['Baixo', 'Médio', 'Alto'])
    
    return df

# Função para criar gráficos com legendas em português
def criar_grafico(tipo, df, x, y=None, color=None, title="", explicacao=""):
    # Mapear nomes das colunas para português
    x_label = TRADUCOES.get(x, x)
    y_label = TRADUCOES.get(y, y) if y else None
    color_label = TRADUCOES.get(color, color) if color else None
    
    if tipo == "histogram":
        fig = px.histogram(df, x=x, color=color, 
                          nbins=20, barmode='overlay', opacity=0.7,
                          title=title,
                          labels=TRADUCOES,
                          color_discrete_map={'Sim': '#FF6B6B', 'Não': '#4ECDC4'})
    elif tipo == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color,
                        hover_data=['risk_category'],
                        title=title,
                        labels=TRADUCOES,
                        color_discrete_map={'Sim': '#FF6B6B', 'Não': '#4ECDC4'})
    elif tipo == "pie":
        fig = px.pie(df, names=color, title=title)
    elif tipo == "bar":
        fig = px.bar(df, x=x, y=y, color=color, barmode='group',
                    title=title, labels=TRADUCOES)
    elif tipo == "box":
        fig = px.box(df, x=color, y=y, title=title, labels=TRADUCOES)
    
    # Adicionar explicação como subtítulo se fornecida
    if explicacao:
        fig.update_layout(
            title=f"{title}<br><sub style='font-size:12px; color:gray'>{explicacao}</sub>"
        )
    
    fig.update_layout(
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

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
    .var-explanation {
        background-color: #ba4e4e;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #ffeded;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def create_sidebar_filters(df):
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtros Globais")
    
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
    
    st.sidebar.markdown("**Filtros Adicionais**")
    smoke_filter = st.sidebar.multiselect(
        "Exposição ao Fumo Passivo",
        options=sorted(df['secondhand_smoke_exposure'].unique()),
        default=sorted(df['secondhand_smoke_exposure'].unique())
    )
    
    copd_filter = st.sidebar.multiselect(
        "Diagnóstico de DPOC",
        options=sorted(df['copd_diagnosis'].unique()),
        default=sorted(df['copd_diagnosis'].unique())
    )
    
    # Adicionar explicações rápidas
    with st.sidebar.expander("💡 Explicações"):
        st.markdown("**Anos-Maço**: Medida de consumo de cigarro (1 maço/dia × 1 ano)")
        st.markdown("**DPOC**: Doença Pulmonar Obstrutiva Crônica")
        st.markdown("**Escala de Risco**: Combina todos os fatores de risco")
    
    return {
        'gender': gender_filter,
        'age': age_filter,
        'lung_cancer': cancer_filter,
        'risk_score': risk_filter,
        'smoke': smoke_filter,
        'copd': copd_filter
    }

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

def render_homepage(df):
    st.markdown('<h1 class="main-header">🫁 Dashboard de Análise de Câncer de Pulmão</h1>', unsafe_allow_html=True)
    
    # Informações sobre as variáveis
    with st.expander("📋 Glossário de Variáveis", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Variáveis Principais:**")
            st.markdown(f'<div class="var-explanation"><strong>Idade:</strong> Idade do paciente em anos</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="var-explanation"><strong>Anos-Maço:</strong> {EXPLICACOES["pack_years"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="var-explanation"><strong>Escala de Risco:</strong> {EXPLICACOES["risk_score"]}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Fatores de Risco:**")
            st.markdown(f'<div class="var-explanation"><strong>Exposição ao Radônio:</strong> {EXPLICACOES["radon_exposure"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="var-explanation"><strong>Exposição ao Amianto:</strong> {EXPLICACOES["asbestos_exposure"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="var-explanation"><strong>DPOC:</strong> {EXPLICACOES["copd_diagnosis"]}</div>', unsafe_allow_html=True)
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(df)
        st.markdown(f'''
        <div class="metric-card">
            <h3>👥 Total de Pacientes</h3>
            <div style="font-size: 2rem; font-weight: bold;">{total_patients}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        cancer_rate = (df['lung_cancer'] == 'Sim').mean() * 100
        st.markdown(f'''
        <div class="metric-card">
            <h3>🎯 Taxa de Câncer</h3>
            <div style="font-size: 2rem; font-weight: bold;">{cancer_rate:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_age = df['age'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <h3>📊 Idade Média</h3>
            <div style="font-size: 2rem; font-weight: bold;">{avg_age:.1f}</div>
            <div>anos</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        high_risk = (df['risk_category'] == 'Alto').mean() * 100
        st.markdown(f'''
        <div class="metric-card">
            <h3>⚠️ Risco Alto</h3>
            <div style="font-size: 2rem; font-weight: bold;">{high_risk:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Primeira linha de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Distribuição por Idade e Diagnóstico</h3>', unsafe_allow_html=True)
        fig = criar_grafico("histogram", df, x='age', color='lung_cancer',
                           title="Distribuição de Idade por Diagnóstico de Câncer")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">Fatores de Risco por Gênero</h3>', unsafe_allow_html=True)
        
        # Preparar dados para fatores de risco
        risk_data = []
        factors = ['radon_exposure', 'asbestos_exposure', 'copd_diagnosis', 'family_history']
        
        for factor in factors:
            cross_tab = pd.crosstab(df[factor], df['gender'], normalize='index') * 100
            for category in cross_tab.index:
                for gender in cross_tab.columns:
                    risk_data.append({
                        'Fator': f"{TRADUCOES.get(factor, factor)} - {category}",
                        'Gênero': gender,
                        'Percentual': cross_tab.loc[category, gender]
                    })
        
        risk_df = pd.DataFrame(risk_data)
        if not risk_df.empty:
            fig = px.bar(risk_df, x='Fator', y='Percentual', color='Gênero',
                        barmode='group', 
                        title='Distribuição de Fatores de Risco por Gênero (%)',
                        labels={'Percentual': 'Percentual (%)', 'Fator': 'Fator de Risco'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Segunda linha de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Relação entre Idade e Anos-Maço</h3>', unsafe_allow_html=True)
        fig = criar_grafico("scatter", df, x='age', y='pack_years', color='lung_cancer',
                           title="Relação: Idade vs Anos-Maço",
                           explicacao="Cada ponto representa um paciente. Tamanho indica escala de risco.")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">Distribuição por Categoria de Risco</h3>', unsafe_allow_html=True)
        risk_counts = df['risk_category'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title='Proporção de Pacientes por Categoria de Risco',
                    labels={'names': 'Categoria de Risco', 'value': 'Número de Pacientes'})
        st.plotly_chart(fig, use_container_width=True)

def render_analysis(df):
    st.markdown('<h1 class="main-header">📊 Análise Detalhada</h1>', unsafe_allow_html=True)
    
    # Legenda interativa
    with st.expander("📖 Legenda das Variáveis", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Variáveis Numéricas:**")
            st.markdown("- **Idade**: Idade do paciente")
            st.markdown("- **Anos-Maço**: Medida de consumo de cigarro")
            st.markdown("- **Escala de Risco**: Pontuação combinada de risco")
        
        with col2:
            st.markdown("**Variáveis Categóricas:**")
            st.markdown("- **Câncer de Pulmão**: Diagnóstico (Sim/Não)")
            st.markdown("- **Gênero**: Masculino/Feminino")
            st.markdown("- **DPOC**: Diagnóstico de doença pulmonar")
        
        with col3:
            st.markdown("**Dicas de Uso:**")
            st.markdown("- Use diferentes combinações de eixos")
            st.markdown("- Experimente colorir por diferentes variáveis")
            st.markdown("- Passe o mouse sobre os pontos para mais info")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<h3 class="section-header">Configurações do Gráfico</h3>', unsafe_allow_html=True)
        
        # Seletores com labels em português
        opcoes_numericas = [
            ('age', 'Idade'),
            ('pack_years', 'Anos-Maço'), 
            ('risk_score', 'Escala de Risco')
        ]
        
        opcoes_categoricas = [
            ('lung_cancer', 'Câncer de Pulmão'),
            ('gender', 'Gênero'),
            ('copd_diagnosis', 'DPOC'),
            ('risk_category', 'Categoria de Risco'),
            ('radon_exposure', 'Exposição ao Radônio')
        ]
        
        x_axis = st.selectbox("Eixo X", 
                             options=[opt[0] for opt in opcoes_numericas],
                             format_func=lambda x: dict(opcoes_numericas)[x])
        
        y_axis = st.selectbox("Eixo Y", 
                             options=[opt[0] for opt in opcoes_numericas],
                             index=1,
                             format_func=lambda x: dict(opcoes_numericas)[x])
        
        color_by = st.selectbox("Colorir por", 
                               options=[opt[0] for opt in opcoes_categoricas],
                               format_func=lambda x: dict(opcoes_categoricas)[x])
        
        chart_type = st.radio("Tipo de Gráfico", 
                             ["Dispersão", "Boxplot", "Histograma"])
        
        # Explicação da variável selecionada
        var_explicacao = EXPLICACOES.get(x_axis, "") or EXPLICACOES.get(y_axis, "")
        if var_explicacao:
            st.markdown(f'<div class="var-explanation"><strong>Explicação:</strong> {var_explicacao}</div>', 
                       unsafe_allow_html=True)
    
    with col2:
        try:
            if chart_type == "Dispersão":
                fig = criar_grafico("scatter", df, x=x_axis, y=y_axis, color=color_by,
                                   title=f"{TRADUCOES.get(y_axis, y_axis)} vs {TRADUCOES.get(x_axis, x_axis)}")
                
            elif chart_type == "Boxplot":
                fig = criar_grafico("box", df, y=y_axis, color=color_by,
                                   title=f"Distribuição de {TRADUCOES.get(y_axis, y_axis)} por {TRADUCOES.get(color_by, color_by)}")
                
            else:  # Histograma
                fig = criar_grafico("histogram", df, x=x_axis, color=color_by,
                                   title=f"Distribuição de {TRADUCOES.get(x_axis, x_axis)}")
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao criar gráfico: {e}")
            st.info("Tente ajustar as configurações do gráfico")

def render_reports(df):
    st.markdown('<h1 class="main-header">📋 Relatórios e Dados</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dados Completos", "📈 Estatísticas", "💾 Exportar Dados"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Dataset Completo</h3>', unsafe_allow_html=True)
        
        # Mostrar descrição das colunas
        with st.expander("📝 Descrição das Colunas"):
            descricoes = {
                'ID do Paciente': 'Identificador único do paciente',
                'Idade': 'Idade em anos',
                'Gênero': 'Masculino ou Feminino',
                'Anos-Maço': 'Medida de consumo de cigarro (1 maço/dia × 1 ano)',
                'Exposição ao Radônio': 'Nível de exposição ao gás radônio',
                'Exposição ao Amianto': 'Se foi exposto ao amianto',
                'Exposição ao Fumo Passivo': 'Se foi exposto ao fumo passivo',
                'Diagnóstico de DPOC': 'Se tem diagnóstico de DPOC',
                'Consumo de Álcool': 'Nível de consumo de álcool',
                'Histórico Familiar': 'Histórico familiar de câncer de pulmão',
                'Câncer de Pulmão': 'Diagnóstico de câncer de pulmão'
            }
            
            for col, desc in descricoes.items():
                st.markdown(f"**{col}**: {desc}")
        
        st.dataframe(df, use_container_width=True, height=400)
        
        st.markdown(f"**Total de registros:** {len(df)}")
    
    with tab2:
        st.markdown('<h3 class="section-header">Estatísticas Descritivas</h3>', unsafe_allow_html=True)
        
        st.markdown("**Variáveis Numéricas:**")
        numeric_stats = df.select_dtypes(include=[np.number]).describe()
        # Traduzir índices das estatísticas
        numeric_stats.index = ['Contagem', 'Média', 'Desvio Padrão', 'Mínimo', '25%', 'Mediana', '75%', 'Máximo']
        st.dataframe(numeric_stats, use_container_width=True)
        
        st.markdown("**Estatísticas por Diagnóstico de Câncer:**")
        if 'lung_cancer' in df.columns:
            grouped_stats = df.groupby('lung_cancer').agg({
                'age': ['mean', 'std', 'min', 'max'],
                'pack_years': ['mean', 'std', 'min', 'max'],
                'risk_score': ['mean', 'std', 'min', 'max']
            }).round(2)
            grouped_stats.columns = [
                f"{TRADUCOES.get(var, var)} - {stat}"
                for var, stat in grouped_stats.columns
            ]
            grouped_stats.columns = [
                col.replace('mean', 'Média')
                    .replace('std', 'Desvio Padrão')
                    .replace('min', 'Mínimo')
                    .replace('max', 'Máximo')
                for col in grouped_stats.columns
            ]
            st.dataframe(grouped_stats, use_container_width=True)
    
    with tab3:
        st.markdown('<h3 class="section-header">Exportar Dados</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.radio("Formato", ["CSV", "JSON"])
            include_filters = st.checkbox("Incluir apenas dados filtrados", value=True)
        
        with col2:
            filename = st.text_input("Nome do arquivo", "dados_cancer_pulmao")
            include_timestamp = st.checkbox("Incluir data e hora", value=True)
        
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
    st.title("🫁 Análise de Câncer de Pulmão")
    st.markdown("*Dashboard de análise de fatores de risco*")
    st.markdown("---")
    
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

# Estatísticas na sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Estatísticas dos Filtros")
st.sidebar.markdown(f"**Pacientes visíveis:** {len(df_filtered):,} / {len(df):,}")
st.sidebar.markdown(f"**Taxa de câncer:** {(df_filtered['lung_cancer'] == 'Sim').mean()*100:.1f}%")

if len(df_filtered) < len(df):
    st.sidebar.progress(len(df_filtered) / len(df))

# Navegação
if selected == "Dashboard":
    render_homepage(df_filtered)
elif selected == "Análise":
    render_analysis(df_filtered)
elif selected == "Relatórios":
    render_reports(df_filtered)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.markdown("🫁 *Análise de Câncer de Pulmão v2.0*")