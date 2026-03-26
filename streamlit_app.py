import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="PreçoBoiBR", page_icon="🐂", layout="wide")
st.title("🐂 PreçoBoiBR - Preço do Boi Gordo no Brasil")

# ==================== CARREGAR DADOS ====================
@st.cache_data
def carregar_dados():
    df = pd.read_excel('data/Boi_gordo_corrigido.xlsx', engine='openpyxl')
    df = df.rename(columns={'Valor': 'Preco_Arroba'})
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
    df['Preco_Arroba'] = pd.to_numeric(
        df['Preco_Arroba'].astype(str)
        .str.replace(r'R\$', '', regex=True)
        .str.replace(',', '.', regex=False)
        .str.replace('-', '', regex=False)
        .str.strip(), errors='coerce'
    )
    df = df.dropna(subset=['Preco_Arroba', 'Data']).sort_values('Data').reset_index(drop=True)
    return df

cepea = carregar_dados()
preco_atual = cepea['Preco_Arroba'].iloc[-1]
data_atual = cepea['Data'].iloc[-1].strftime("%d/%m/%Y")

st.success(f"✅ **Cepea/SP (Referência Nacional)**: **R$ {preco_atual:.2f}** em {data_atual}")

# ==================== CALCULADORA ====================
st.header("🧮 Calculadora de Preço do Animal")
col1, col2, col3 = st.columns(3)
with col1:
    peso = st.number_input("Peso vivo do animal (kg)", min_value=300, max_value=800, value=510, step=10)
with col2:
    tipo = st.selectbox("Tipo de animal", ["Boi Gordo", "Vaca Gorda", "Novilha"])
with col3:
    rendimento = st.slider("Rendimento de carcaça (%)", 48, 55, 50, step=1) / 100

arrobas = round((peso * rendimento) / 15, 2)
valor_estimado = round(arrobas * preco_atual, 2)

st.metric("📦 Arrobas estimadas", f"{arrobas}")
st.metric("💰 Valor total estimado", f"R$ {valor_estimado:,.2f}")

# ==================== PREVISÃO ====================
st.header("🔮 Previsão dos próximos 90 dias - Cepea/SP (Referência Nacional)")

@st.cache_resource
def treinar_prophet():
    df_prophet = cepea[['Data', 'Preco_Arroba']].copy()
    df_prophet.columns = ['ds', 'y']
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
    model.fit(df_prophet)
    return model

modelo = treinar_prophet()
futuro = modelo.make_future_dataframe(periods=90)
previsao = modelo.predict(futuro)

hoje = cepea['Data'].iloc[-1]

# Gráfico principal (foco nos últimos 12 meses + previsão)
ultimos_12_meses = cepea[cepea['Data'] >= hoje - pd.Timedelta(days=365)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=ultimos_12_meses['Data'], y=ultimos_12_meses['Preco_Arroba'],
                         mode='lines', name='Histórico', line=dict(color='#1f77b4', width=3)))
fig.add_trace(go.Scatter(x=previsao['ds'], y=previsao['yhat'],
                         mode='lines', name='Previsão 90 dias', line=dict(color='#2ca02c', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=previsao['ds'], y=previsao['yhat_lower'],
                         mode='lines', fill='tonexty', fillcolor='rgba(44,160,44,0.25)',
                         line=dict(width=0), name='Limite inferior', showlegend=False))

fig.add_vline(x=hoje, line_dash="dash", line_color="red")
fig.add_annotation(x=hoje, y=1.06, text="HOJE", showarrow=False, font=dict(color="red", size=16))

fig.update_layout(
    title="🔥 FOCO: Previsão dos próximos 90 dias",
    xaxis_title="Data",
    yaxis_title="Preço por Arroba (R$)",
    hovermode="x unified",
    template="plotly_white",
    height=700
)
fig.update_xaxes(range=[hoje - pd.Timedelta(days=380), hoje + pd.Timedelta(days=95)])

st.plotly_chart(fig, use_container_width=True)

# Previsão +30 dias
target_date = hoje + pd.Timedelta(days=30)
idx = previsao['ds'].sub(target_date).abs().idxmin()
preco_30d = previsao.loc[idx, 'yhat']
st.info(f"🔮 Previsão +30 dias: **R$ {preco_30d:.2f}** por arroba")

# ==================== HISTÓRICO + DÓLAR ====================
with st.expander("📜 Ver histórico completo + comparação com Dólar", expanded=False):
    dolar = yf.download('USDBRL=X', period='10y', progress=False)['Close'].reset_index()
    dolar.columns = ['Data', 'Dolar']
    dolar['Data'] = pd.to_datetime(dolar['Data'])

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=cepea['Data'], y=cepea['Preco_Arroba'],
                                  mode='lines', name='Cepea/SP - Arroba', line=dict(color='#1f77b4', width=2)))
    fig_hist.add_trace(go.Scatter(x=dolar['Data'], y=dolar['Dolar'],
                                  mode='lines', name='Dólar USD/BRL', line=dict(color='#ff7f0e', width=2), yaxis='y2'))

    fig_hist.update_layout(
        title="Comparação Histórica: Preço da Arroba × Cotação do Dólar",
        xaxis_title="Data",
        yaxis_title="Preço Arroba (R$)",
        yaxis2=dict(title="Dólar (R$)", overlaying='y', side='right'),
        template="plotly_white",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.caption("PreçoBoiBR • Dados Cepea/SP (Referência Nacional) • Projeto de Ciência de Dados")