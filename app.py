import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="Sistema Oil&Gas",
    page_icon="🛢️",
    layout="wide"
)

# Lista de ativos
ATIVOS = [
    "PETR4.SA", "VBBR3.SA", "CSAN3.SA", "UGPA3.SA", "RAIZ4.SA",
    "RECV3.SA", "PRIO3.SA", "BRAV3.SA", "RPMG3.SA",
    "EXXO34.SA", "CHVX34.SA", "B1PP34.SA", "E1QN34.SA", "COPH34.SA",
    "E1OG34.SA", "M1PC34.SA", "VLOE34.SA", "D1VN34.SA", "B1KR34.SA", "C1OG34.SA"
]

# Nomes amigáveis para TODOS os ativos
NOMES_ATIVOS = {
    "PETR4.SA": "Petrobras (PETR4)",
    "VBBR3.SA": "Vibra Energia (VBBR3)",
    "CSAN3.SA": "Cosan (CSAN3)",
    "UGPA3.SA": "Ultrapar (UGPA3)",
    "RAIZ4.SA": "Raízen (RAIZ4)",
    "RECV3.SA": "PetroRecôncavo (RECV3)",
    "PRIO3.SA": "PetroRio (PRIO3)",
    "BRAV3.SA": "Brava Energia (BRAV3)",
    "RPMG3.SA": "Refinaria Manguinhos (RPMG3)",
    "EXXO34.SA": "Exxon Mobil (EXXO34)",
    "CHVX34.SA": "Chevron Corp (CHVX34)",
    "B1PP34.SA": "BP PLC (B1PP34)",
    "E1QN34.SA": "Equinor (E1QN34)",
    "COPH34.SA": "ConocoPhillips (COPH34)",
    "E1OG34.SA": "EOG Resources (E1OG34)",
    "M1PC34.SA": "Marathon Petroleum (M1PC34)",
    "VLOE34.SA": "Valero Energy (VLOE34)",
    "D1VN34.SA": "Devon Energy (D1VN34)",
    "B1KR34.SA": "Baker Hughes (B1KR34)",
    "C1OG34.SA": "Coterra Energy (C1OG34)"
}

@st.cache_data(ttl=3600)
def carregar_dados():
    """Carrega dados históricos dos ativos"""
    inicio = "2021-10-01"
    fim = datetime.now().strftime("%Y-%m-%d")
    
    dados = yf.download(ATIVOS, start=inicio, end=fim, auto_adjust=True, progress=False)
    if isinstance(dados.columns, pd.MultiIndex):
        fechamento = dados['Close']
    else:
        fechamento = dados
    
    return fechamento

@st.cache_data
def calcular_retornos(fechamento):
    """Calcula retornos diários"""
    retornos = fechamento.pct_change().dropna()
    return retornos

@st.cache_data
def calcular_rsi(prices, period=14):
    """Calcula RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data
def calcular_sharpe_ratio(retornos, rf_rate=0.1075):
    """Calcula Sharpe Ratio"""
    retorno_medio = retornos.mean() * 252
    volatilidade = retornos.std() * np.sqrt(252)
    if volatilidade == 0:
        return 0
    sharpe = (retorno_medio - rf_rate) / volatilidade
    return sharpe

@st.cache_data
def calcular_fundamentalistas():
    """Calcula indicadores fundamentalistas"""
    fundamentals = pd.DataFrame(columns=["Ativo", "Margem_Lucro", "ROA", "ROE"])
    
    for ativo in ATIVOS:
        try:
            ticker = yf.Ticker(ativo)
            info = ticker.info
            receita = info.get('totalRevenue', None)
            
            try:
                lucro = ticker.financials.loc['Net Income'][0]
            except:
                lucro = None
            
            try:
                ativos_total = ticker.balance_sheet.loc['Total Assets'][0]
            except:
                ativos_total = None
            
            try:
                patrimonio = ticker.balance_sheet.loc['Stockholders Equity'][0]
            except:
                patrimonio = None
            
            margem_lucro = (lucro / receita * 100) if (receita and lucro) else None
            roa = (lucro / ativos_total * 100) if (ativos_total and lucro) else None
            roe = (lucro / patrimonio * 100) if (patrimonio and lucro) else None
            
            fundamentals = pd.concat([fundamentals, pd.DataFrame([{
                "Ativo": ativo,
                "Margem_Lucro": margem_lucro,
                "ROA": roa,
                "ROE": roe
            }])], ignore_index=True)
        except:
            fundamentals = pd.concat([fundamentals, pd.DataFrame([{
                "Ativo": ativo,
                "Margem_Lucro": None,
                "ROA": None,
                "ROE": None
            }])], ignore_index=True)
    
    return fundamentals

@st.cache_data
def treinar_modelo_ml(retornos):
    """Treina Random Forest para todos os ativos"""
    resultados = {}
    
    for ativo in ATIVOS:
        if ativo not in retornos.columns:
            resultados[ativo] = {'acuracia': 0, 'prob_retorno_positivo': 0.5}
            continue
            
        df = pd.DataFrame(retornos[ativo]).dropna()
        
        if len(df) < 50:
            resultados[ativo] = {'acuracia': 0, 'prob_retorno_positivo': 0.5}
            continue
        
        for i in range(1, 6):
            df[f'lag_{i}'] = df[ativo].shift(i)
        
        df['ma_5'] = df[ativo].rolling(5).mean()
        df['ma_10'] = df[ativo].rolling(10).mean()
        df['volatilidade'] = df[ativo].rolling(10).std()
        df['target'] = (df[ativo].shift(-1) > 0).astype(int)
        
        df = df.dropna()
        
        if len(df) < 50:
            resultados[ativo] = {'acuracia': 0, 'prob_retorno_positivo': 0.5}
            continue
        
        X = df.drop([ativo, 'target'], axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        modelo.fit(X_train, y_train)
        
        score = modelo.score(X_test, y_test)
        prob_positivo = modelo.predict_proba(X.iloc[[-1]])[0][1]
        
        resultados[ativo] = {
            'acuracia': score,
            'prob_retorno_positivo': prob_positivo
        }
    
    return resultados

def perfilacao_usuario():
    """Sistema de perfilação"""
    st.header("📋 Perfilação de Investidor")
    st.write("Responda as perguntas abaixo para personalizarmos sua carteira:")
    
    pontos = 0
    
    idade = st.selectbox(
        "1. Qual sua idade?",
        ["18-25 anos", "26-35 anos", "36-50 anos", "51-65 anos", "Acima de 65 anos"]
    )
    pontos += [3, 3, 2, 1, 0][["18-25 anos", "26-35 anos", "36-50 anos", "51-65 anos", "Acima de 65 anos"].index(idade)]
    
    experiencia = st.selectbox(
        "2. Há quanto tempo você investe?",
        ["Menos de 1 ano", "1-3 anos", "3-5 anos", "5-10 anos", "Mais de 10 anos"]
    )
    pontos += [0, 1, 2, 3, 3][["Menos de 1 ano", "1-3 anos", "3-5 anos", "5-10 anos", "Mais de 10 anos"].index(experiencia)]
    
    objetivo = st.selectbox(
        "3. Seu principal objetivo é:",
        ["Preservar capital", "Crescimento moderado", "Ganhos agressivos", "Especulação de curto prazo"]
    )
    pontos += [0, 1, 2, 3][["Preservar capital", "Crescimento moderado", "Ganhos agressivos", "Especulação de curto prazo"].index(objetivo)]
    
    prazo = st.selectbox(
        "4. Qual seu horizonte de investimento?",
        ["Menos de 1 ano", "1-3 anos", "3-5 anos", "5-10 anos", "Mais de 10 anos"]
    )
    pontos += [0, 1, 2, 3, 3][["Menos de 1 ano", "1-3 anos", "3-5 anos", "5-10 anos", "Mais de 10 anos"].index(prazo)]
    
    reserva = st.selectbox(
        "5. Você possui reserva de emergência (6-12 meses de despesas)?",
        ["Não", "Parcialmente", "Sim, completa"]
    )
    pontos += [0, 1, 2][["Não", "Parcialmente", "Sim, completa"].index(reserva)]
    
    reacao = st.selectbox(
        "6. Como você reage a oscilações negativas na carteira?",
        ["Vende tudo imediatamente", "Fica preocupado e considera vender", "Mantém a posição", "Aceito quedas relevantes se houver perspectiva de retorno"]
    )
    pontos += [0, 1, 2, 3][["Vende tudo imediatamente", "Fica preocupado e considera vender", "Mantém a posição", "Aceito quedas relevantes se houver perspectiva de retorno"].index(reacao)]
    
    conhecimento = st.selectbox(
        "7. Qual seu nível de conhecimento sobre o setor de Oil & Gas?",
        ["Nenhum", "Básico", "Intermediário", "Avançado"]
    )
    pontos += [0, 1, 2, 3][["Nenhum", "Básico", "Intermediário", "Avançado"].index(conhecimento)]
    
    if pontos <= 7:
        perfil = "Conservador"
        cor = "🟢"
    elif pontos <= 14:
        perfil = "Moderado"
        cor = "🟡"
    else:
        perfil = "Agressivo"
        cor = "🔴"
    
    st.success(f"### {cor} Seu perfil: **{perfil}** (Pontuação: {pontos}/21)")
    
    return perfil, pontos

def montar_carteira_personalizada(perfil, retornos, fechamento, fundamentals, ml_results):
    """Monta carteira integrando todas as análises"""
    
    ativos_disponiveis = [a for a in ATIVOS if a in retornos.columns]
    
    # 1. Score de Correlação
    corr = retornos.corr()
    corr_sum = corr.sum().sort_values()
    
    score_corr = {}
    for ativo in ativos_disponiveis:
        if ativo in corr_sum.index:
            valor_normalizado = (corr_sum[ativo] - corr_sum.min()) / (corr_sum.max() - corr_sum.min())
            score_corr[ativo] = 1 - valor_normalizado
        else:
            score_corr[ativo] = 0.5
    
    # 2. Score Fundamentalista
    score_fund = {}
    for _, row in fundamentals.iterrows():
        ativo = row['Ativo']
        if ativo not in ativos_disponiveis:
            continue
        roa = row['ROA'] if pd.notna(row['ROA']) else 0
        roe = row['ROE'] if pd.notna(row['ROE']) else 0
        margem = row['Margem_Lucro'] if pd.notna(row['Margem_Lucro']) else 0
        score_fund[ativo] = (roa + roe + margem) / 3
    
    if score_fund:
        max_fund = max(score_fund.values())
        min_fund = min(score_fund.values())
        if max_fund > min_fund:
            score_fund = {k: (v - min_fund) / (max_fund - min_fund) for k, v in score_fund.items()}
        else:
            score_fund = {k: 0.5 for k in score_fund}
    
    # 3. Score ML
    score_ml = {ativo: ml_results[ativo]['prob_retorno_positivo'] for ativo in ativos_disponiveis}
    
    # 4. Score RSI
    score_rsi = {}
    for ativo in ativos_disponiveis:
        if ativo in fechamento.columns:
            rsi = calcular_rsi(fechamento[ativo])
            rsi_atual = rsi.iloc[-1]
            
            if pd.isna(rsi_atual):
                score_rsi[ativo] = 0.5
            elif rsi_atual < 30:
                score_rsi[ativo] = 0.8
            elif rsi_atual > 70:
                score_rsi[ativo] = 0.2
            else:
                score_rsi[ativo] = 0.5 + (50 - abs(rsi_atual - 50)) / 100
        else:
            score_rsi[ativo] = 0.5
    
    # 5. Score Sharpe
    score_sharpe = {}
    sharpe_values = {}
    
    for ativo in ativos_disponiveis:
        if ativo in retornos.columns:
            sharpe = calcular_sharpe_ratio(retornos[ativo])
            sharpe_values[ativo] = sharpe
        else:
            sharpe_values[ativo] = 0
    
    if sharpe_values:
        max_sharpe = max(sharpe_values.values())
        min_sharpe = min(sharpe_values.values())
        if max_sharpe > min_sharpe:
            for ativo in ativos_disponiveis:
                score_sharpe[ativo] = (sharpe_values[ativo] - min_sharpe) / (max_sharpe - min_sharpe)
        else:
            score_sharpe = {k: 0.5 for k in sharpe_values}
    
    # 6. Pesos por perfil
    score_final = {}
    detalhes_scores = {}
    
    if perfil == "Conservador":
        peso_sharpe, peso_corr, peso_fund, peso_rsi, peso_ml = 0.30, 0.25, 0.25, 0.15, 0.05
    elif perfil == "Moderado":
        peso_sharpe, peso_corr, peso_fund, peso_rsi, peso_ml = 0.25, 0.20, 0.20, 0.15, 0.20
    else:
        peso_sharpe, peso_corr, peso_fund, peso_rsi, peso_ml = 0.20, 0.10, 0.15, 0.20, 0.35
    
    for ativo in ativos_disponiveis:
        s_corr = score_corr.get(ativo, 0.5)
        s_fund = score_fund.get(ativo, 0.5)
        s_ml = score_ml.get(ativo, 0.5)
        s_rsi = score_rsi.get(ativo, 0.5)
        s_sharpe = score_sharpe.get(ativo, 0.5)
        
        score_total = (peso_corr * s_corr) + (peso_fund * s_fund) + (peso_ml * s_ml) + (peso_rsi * s_rsi) + (peso_sharpe * s_sharpe)
        
        score_final[ativo] = score_total
        detalhes_scores[ativo] = {
            'Correlação': s_corr,
            'Fundamentos': s_fund,
            'ML': s_ml,
            'RSI': s_rsi,
            'Sharpe': s_sharpe,
            'Score Final': score_total
        }
    
    # 7. Top 5
    top_5 = sorted(score_final.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # 8. Pesos
    total_score = sum([score for _, score in top_5])
    carteira = {ativo: score / total_score for ativo, score in top_5}
    
    detalhes_top5 = {ativo: detalhes_scores[ativo] for ativo, _ in top_5}
    
    return carteira, detalhes_top5, (peso_sharpe, peso_corr, peso_fund, peso_rsi, peso_ml)

def main():
    st.title("🛢️ Sistema Inteligente de Recomendação Oil & Gas")
    st.markdown("---")
    
    st.sidebar.title("Navegação")
    opcao = st.sidebar.radio(
        "Escolha uma opção:",
        ["🏠 Início", "📊 Análise de Mercado", "💰 Simulador de Investimento", "🤖 Previsão ML", "📈 Indicadores"]
    )
    
    fechamento = carregar_dados()
    retornos = calcular_retornos(fechamento)
    
    if opcao == "🏠 Início":
        st.header("Bem-vindo ao Sistema de Recomendação Oil & Gas!")
        
        st.markdown("""
        ### 📌 Sobre o Sistema
        
        Este sistema utiliza **5 metodologias complementares** para recomendar investimentos:
        
        1. **📊 Análise de Correlação**: Diversificação ótima
        2. **💼 Análise Fundamentalista**: ROA, ROE e Margem de Lucro
        3. **🤖 Machine Learning**: Random Forest para previsões
        4. **📈 RSI**: Timing técnico de entrada
        5. **⚖️ Sharpe Ratio**: Retorno ajustado ao risco
        
        ### 🚀 Como Usar
        
        1. Vá em **"Simulador de Investimento"**
        2. Responda 7 perguntas
        3. Insira o valor a investir
        4. Receba sua carteira personalizada!
        """)
    
    elif opcao == "📊 Análise de Mercado":
        st.header("📊 Análise de Mercado")
        
        tab1, tab2, tab3 = st.tabs(["Correlação", "Preços Históricos", "Volatilidade"])
        
        with tab1:
            st.subheader("Matriz de Correlação")
            corr = retornos.corr()
            
            fig = px.imshow(
                corr,
                labels=dict(color="Correlação"),
                color_continuous_scale="RdBu_r",
                aspect="auto",
                x=[NOMES_ATIVOS.get(col, col) for col in corr.columns],
                y=[NOMES_ATIVOS.get(col, col) for col in corr.index]
            )
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Evolução dos Preços")
            
            ativos_selecionados = st.multiselect(
                "Selecione os ativos:",
                options=ATIVOS,
                default=[ATIVOS[0]],
                format_func=lambda x: NOMES_ATIVOS.get(x, x)
            )
            
            if ativos_selecionados:
                fig = go.Figure()
                for ativo in ativos_selecionados:
                    if ativo in fechamento.columns:
                        fig.add_trace(go.Scatter(
                            x=fechamento.index,
                            y=fechamento[ativo],
                            mode='lines',
                            name=NOMES_ATIVOS.get(ativo, ativo)
                        ))
                
                fig.update_layout(
                    title="Preços Históricos",
                    xaxis_title="Data",
                    yaxis_title="Preço (R$)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Volatilidade")
            
            ativos_vol = st.multiselect(
                "Selecione os ativos:",
                options=ATIVOS,
                default=ATIVOS[:5],
                format_func=lambda x: NOMES_ATIVOS.get(x, x),
                key="vol_select"
            )
            
            if ativos_vol:
                volatilidade = retornos[ativos_vol].std() * np.sqrt(252) * 100
                
                fig = px.bar(
                    x=[NOMES_ATIVOS.get(idx, idx) for idx in volatilidade.index],
                    y=volatilidade.values,
                    labels={'x': 'Ativo', 'y': 'Volatilidade Anualizada (%)'},
                    title="Volatilidade Anualizada"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif opcao == "💰 Simulador de Investimento":
        st.header("💰 Simulador de Investimento")
        
        perfil, pontos = perfilacao_usuario()
        
        st.markdown("---")
        
        st.subheader("💵 Quanto você deseja investir?")
        capital = st.number_input(
            "Valor (R$):",
            min_value=100.0,
            max_value=10000000.0,
            value=10000.0,
            step=100.0
        )
        
        if st.button("🚀 Calcular Alocação Personalizada", type="primary"):
            with st.spinner("Montando sua carteira..."):
                fundamentals = calcular_fundamentalistas()
                ml_results = treinar_modelo_ml(retornos)
                carteira, detalhes_scores, pesos_perfil = montar_carteira_personalizada(perfil, retornos, fechamento, fundamentals, ml_results)
            
            st.success("### ✅ Sua Carteira Personalizada")
            
            peso_sharpe, peso_corr, peso_fund, peso_rsi, peso_ml = pesos_perfil
            
            with st.expander("📚 Como sua carteira foi construída", expanded=True):
                st.markdown(f"""
                ### Perfil: **{perfil}**
                
                | Análise | Peso |
                |---------|------|
                | Sharpe Ratio | {peso_sharpe*100:.0f}% |
                | Correlação | {peso_corr*100:.0f}% |
                | Fundamentos | {peso_fund*100:.0f}% |
                | RSI | {peso_rsi*100:.0f}% |
                | ML | {peso_ml*100:.0f}% |
                """)
            
            st.subheader("🔍 Scores dos Ativos Selecionados")
            
            scores_df = pd.DataFrame(detalhes_scores).T
            scores_df['Ativo'] = [NOMES_ATIVOS.get(idx, idx) for idx in scores_df.index]
            scores_df = scores_df[['Ativo', 'Correlação', 'Fundamentos', 'ML', 'RSI', 'Sharpe', 'Score Final']]
            scores_df = scores_df.sort_values('Score Final', ascending=False)
            
            for col in ['Correlação', 'Fundamentos', 'ML', 'RSI', 'Sharpe', 'Score Final']:
                scores_df[col] = scores_df[col].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(scores_df, use_container_width=True)
            
            st.subheader("💼 Alocação de Capital")
            
            alocacao_df = pd.DataFrame({
                'Ativo': [NOMES_ATIVOS.get(k, k) for k in carteira.keys()],
                'Peso (%)': [v * 100 for v in carteira.values()],
                'Valor (R$)': [capital * v for v in carteira.values()]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(alocacao_df, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    alocacao_df,
                    values='Peso (%)',
                    names='Ativo',
                    title="Distribuição"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Métricas da Carteira")
            
            retornos_carteira = retornos[[k for k in carteira.keys() if k in retornos.columns]]
            pesos = np.array(list(carteira.values()))
            
            retorno_esperado = (retornos_carteira.mean() @ pesos) * 252 * 100
            volatilidade_carteira = np.sqrt(pesos @ retornos_carteira.cov() @ pesos) * np.sqrt(252) * 100
            sharpe = calcular_sharpe_ratio(retornos_carteira @ pesos)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Retorno Anual Esperado", f"{retorno_esperado:.2f}%")
            col2.metric("Volatilidade Anual", f"{volatilidade_carteira:.2f}%")
            col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    elif opcao == "🤖 Previsão ML":
        st.header("🤖 Previsão com Machine Learning")
        
        st.info("Random Forest prevê probabilidade de retorno positivo")
        
        with st.spinner("Treinando modelos..."):
            resultados_ml = treinar_modelo_ml(retornos)
        
        st.subheader("📈 Resultados")
        
        ml_df = pd.DataFrame(resultados_ml).T
        ml_df['Nome'] = [NOMES_ATIVOS.get(idx, idx) for idx in ml_df.index]
        ml_df['Probabilidade (%)'] = ml_df['prob_retorno_positivo'] * 100
        ml_df['Acurácia (%)'] = ml_df['acuracia'] * 100
        
        ml_df = ml_df[['Nome', 'Probabilidade (%)', 'Acurácia (%)']].reset_index(drop=True)
        
        st.dataframe(ml_df, use_container_width=True)
        
        ativos_ml = st.multiselect(
            "Selecione ativos para o gráfico:",
            options=range(len(ml_df)),
            default=list(range(min(10, len(ml_df)))),
            format_func=lambda x: ml_df.iloc[x]['Nome'],
            key="ml_select"
        )
        
        if ativos_ml:
            ml_selected = ml_df.iloc[ativos_ml]
            
            fig = px.bar(
                ml_selected,
                x='Nome',
                y='Probabilidade (%)',
                title="Probabilidade de Retorno Positivo",
                color='Probabilidade (%)',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    elif opcao == "📈 Indicadores":
        st.header("📈 Indicadores")
        
        tab1, tab2, tab3 = st.tabs(["RSI", "Sharpe Ratio", "Retornos Acumulados"])
        
        with tab1:
            st.subheader("RSI (14 dias)")
            st.info("RSI < 30 = sobrevendido (compra), RSI > 70 = sobrecomprado (venda)")
            
            ativo_rsi = st.selectbox("Selecione:", ATIVOS, format_func=lambda x: NOMES_ATIVOS.get(x, x), key="rsi")
            
            if ativo_rsi in fechamento.columns:
                rsi = calcular_rsi(fechamento[ativo_rsi])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines', name='RSI'))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido")
                fig.update_layout(
                    title=f"RSI - {NOMES_ATIVOS.get(ativo_rsi, ativo_rsi)}",
                    yaxis_title="RSI",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                rsi_atual = rsi.iloc[-1]
                if rsi_atual > 70:
                    st.warning(f"⚠️ RSI: {rsi_atual:.2f} - Sobrecomprado")
                elif rsi_atual < 30:
                    st.success(f"✅ RSI: {rsi_atual:.2f} - Sobrevendido")
                else:
                    st.info(f"ℹ️ RSI: {rsi_atual:.2f} - Neutro")
        
        with tab2:
            st.subheader("Sharpe Ratio")
            st.info("Sharpe > 1 é bom, > 2 é muito bom")
            
            sharpe_dict = {}
            for ativo in ATIVOS:
                if ativo in retornos.columns:
                    sharpe_val = calcular_sharpe_ratio(retornos[ativo])
                    sharpe_dict[ativo] = sharpe_val
            
            try:
                ibov = yf.download("^BVSP", start="2021-10-01", progress=False)['Close']
                retornos_ibov = ibov.pct_change().dropna()
                sharpe_ibov = calcular_sharpe_ratio(retornos_ibov)
                sharpe_dict['IBOVESPA'] = sharpe_ibov
            except:
                pass
            
            sharpe_df = pd.DataFrame(list(sharpe_dict.items()), columns=['Ticker', 'Sharpe Ratio'])
            sharpe_df['Nome'] = sharpe_df['Ticker'].apply(lambda x: NOMES_ATIVOS.get(x, x) if x != 'IBOVESPA' else 'IBOVESPA')
            sharpe_df = sharpe_df[['Nome', 'Sharpe Ratio']].sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)
            
            ativos_sharpe = st.multiselect(
                "Selecione ativos:",
                options=range(len(sharpe_df)),
                default=list(range(min(10, len(sharpe_df)))),
                format_func=lambda x: sharpe_df.iloc[x]['Nome'],
                key="sharpe_select"
            )
            
            if ativos_sharpe:
                sharpe_selected = sharpe_df.iloc[ativos_sharpe]
                
                fig = px.bar(
                    sharpe_selected,
                    x='Nome',
                    y='Sharpe Ratio',
                    title="Sharpe Ratio",
                    color='Sharpe Ratio',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(sharpe_selected, use_container_width=True)
        
        with tab3:
            st.subheader("Retornos Acumulados")
            
            ativos_retorno = st.multiselect(
                "Selecione ativos:",
                options=ATIVOS,
                default=ATIVOS[:5],
                format_func=lambda x: NOMES_ATIVOS.get(x, x),
                key="retorno_select"
            )
            
            if ativos_retorno:
                retornos_acum = (1 + retornos[ativos_retorno]).cumprod()
                
                fig = go.Figure()
                for col in retornos_acum.columns:
                    fig.add_trace(go.Scatter(
                        x=retornos_acum.index,
                        y=retornos_acum[col],
                        mode='lines',
                        name=NOMES_ATIVOS.get(col, col)
                    ))
                
                fig.update_layout(
                    title="Retornos Acumulados",
                    xaxis_title="Data",
                    yaxis_title="Valor Acumulado (Base 1)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()