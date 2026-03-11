import streamlit as st
from models.seirsd import SEIRSD
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import truncnorm

seirsd = SEIRSD()

st.title("Modelo SEIRSD")

with st.expander("Mostrar Modelo SEIRSD"):
    st.image("pages/figures/seirsd.png", caption="modelo SEIRSD", use_container_width=True)

# with st.expander("Mostrar Equações do Modelo SEIRSD"):
    st.latex(r'''
    \frac{dS}{dt} = -\beta \cdot I \cdot \frac{S}{N} + \alpha \cdot R
    ''')
    st.latex(r'''
    \frac{dE}{dt} = \beta \cdot I \cdot \frac{S}{N} - \sigma \cdot E
    ''')
    st.latex(r'''
    \frac{dI}{dt} = \sigma \cdot  E - \gamma \cdot I - \mu \cdot I
    ''')
    st.latex(r'''
    \frac{dR}{dt} = \gamma \cdot I - \alpha \cdot R
    ''')
    st.latex(r'''
    \frac{dD}{dt} = \mu \cdot I
    ''')
    st.markdown(r'''
    **Onde:**
    - S: suscetíveis  
    - E: expostos  
    - I: infectados  
    - R: recuperados
    - D: mortos  
    - β - beta: taxa de infecção  
    - σ - sigma: taxa de incubação  
    - γ - gamma: taxa de recuperação  
    - α - alfa: taxa de perda de imunidade
    - μ - mu: taxa de mortalidade
    - N = S + E + I + R + D: população total  
    ''')

st.title("Parametrização da População Inicial")

days = st.number_input("Dias", value=int(seirsd.get_default("days")), min_value=0)
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2 = st.columns(2)
with row1_col1:
    S = st.number_input("Susceptíveis (S)", value=int(seirsd.get_default("S")), min_value=0)
with row1_col2:
    E = st.number_input("Expostos (E)", value=int(seirsd.get_default("E")), min_value=0)
with row1_col3:
    I = st.number_input("Infectados (I)", value=int(seirsd.get_default("I")), min_value=0)
with row2_col1:
    R = st.number_input("Recuperados (R)", value=int(seirsd.get_default("R")), min_value=0)
with row2_col2:
    D = st.number_input("Mortos (D)", value=int(seirsd.get_default("D")), min_value=0)

st.title("Simular SEIRSD")

use_beta = st.toggle("Inserir β (desligue para inserir R₀)", value=True)

row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2 = st.columns(2)
with row1_col1:
    if use_beta:
        beta = st.number_input("β (beta) - Transmissão", value=float(seirsd.get_default("beta")), min_value=0.0, step=0.0001, format="%.4f")
    else:   
        r0 = st.number_input("R0", value=float(seirsd.get_default("r0")), min_value=0.0, step=0.0001, format="%.4f")
with row1_col2:
    sigma = st.number_input("σ (sigma) - Incubação", value=float(seirsd.get_default("sigma")), min_value=0.0, step=0.0001, format="%.4f")
with row1_col3:
    gamma = st.number_input("γ (gamma) - Recuperação", value=float(seirsd.get_default("gamma")), min_value=0.0, step=0.0001, format="%.4f")
with row2_col1:
    alfa = st.number_input("α (alfa) - Perda de imunidade", value=float(seirsd.get_default("alfa")), min_value=0.0, step=0.0001, format="%.4f")
with row2_col2:
    mu = st.number_input("μ (mu) - Mortalidade", value=float(seirsd.get_default("mu")), min_value=0.0, step=0.0001, format="%.4f")

if st.button("Rodar Simulação"):
    beta = beta if use_beta else r0 * (gamma + mu)
    initial_conditions = [S, E, I, R, D]
    transfer_rates = [beta, sigma, gamma, alfa, mu]

    fig = seirsd.run_simulation(
        days=days,
        initial_conditions=initial_conditions,
        transfer_rates=transfer_rates,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Configure os parâmetros e clique em **Rodar Simulação**.")

st.subheader("Simular SEIRSD com variações em α")

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    quantity_simulation = st.number_input("Simulações", value=1000, min_value=1, max_value=10000)
with row1_col2:
    confidence_level = st.number_input("Intervalo de confiança", value=0.1, min_value=0.01, max_value=1.0, help="Em decimal. Ex 0.1 para 95%")

alfa_initial_values = [0, 0.0027, 0.0056, 0.0111, 0.0333, 0.0714, 1]
if "alfa_values" not in st.session_state:
    st.session_state.alfa_values = alfa_initial_values

novo_valor = st.number_input(
    "Adicionar novo valor de α",
    min_value=0.0,
    step=0.0001,
    format="%.4f"
)

if st.button("Adicionar valor"):
    if novo_valor not in st.session_state.alfa_values:
        st.session_state.alfa_values.append(novo_valor)
        st.session_state.alfa_values = sorted(st.session_state.alfa_values)

st.markdown("#### Valores atuais de α")
if len(st.session_state.alfa_values) == 0:
    st.write("Nenhum valor adicionado.")
else:
    for i, val in enumerate(st.session_state.alfa_values):
        col1, col2, col3 = st.columns([2, 2, 1])

        col1.write(f"α = {val:.4f}")
        if val > 0:
            dias = 1 / val
            col2.write(f"Aprox {dias:.0f} dias")
        else:
            col2.write("∞ dias")
        if col3.button("Remover", key=f"remover_{i}"):
            st.session_state.alfa_values.pop(i)
            st.session_state.alfa_values = sorted(st.session_state.alfa_values)
            st.rerun()

if st.button("Rodar Métricas"):    
    timespan_days = np.linspace(0, days, days+1)
    beta = beta if use_beta else r0 * (gamma + mu)
    initial_conditions = [S, E, I, R, D]
    alfa_values = st.session_state.alfa_values

    st.markdown("### Cenário base")

    basic_scenarios_results = seirsd.run_alfa_metric_basic_scenario(
        alfa_values=alfa_values,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        mu=mu,
        days=days,
        initial_conditions=initial_conditions,
        timespan_days=timespan_days
    )

    fig1 = go.Figure()
    
    for alfa, data in basic_scenarios_results.items():
        label = f"α = {alfa:.4f} por dia"
        fig1.add_trace(go.Scatter(
            x=data["t"], 
            y=data["D"],
            mode='lines',
            name=label
        ))
    
    fig1.update_layout(
        title="Impacto da perda de imunidade na mortalidade (Modelo SEIRSD)",
        xaxis_title="Tempo (dias)",
        yaxis_title="Mortos acumulados",
        legend_title="Taxa α de perda de imunidade",
    )

    omegas_txt = [f"{w:.4f}" for w in alfa_values]
    final_deaths = [basic_scenarios_results[w]["total_deaths"] for w in alfa_values]
    fig2 = px.bar(
        x=omegas_txt,
        y=final_deaths,
        labels={'x': 'Taxa de perda de imunidade (por dia)', 'y': 'Mortos acumulados'},
        title="Mortalidade final por taxa de perda de imunidade",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Monte Carlo")
    st.text(f"Com {quantity_simulation} simulações por valor de alfa")

    monte_carlo_results = seirsd.run_alfa_metric_monte_carlo(
        alfa_values=alfa_values,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        mu=mu,
        days=days,
        initial_conditions=initial_conditions,
        N_sim=quantity_simulation,
        cv=confidence_level
    )

    fig1 = go.Figure()
    for alfa in alfa_values:
        res = monte_carlo_results[alfa]
        fig1.add_trace(go.Scatter(
            x=timespan_days, y=res["mean"],
            mode='lines',
            name=f"α = {alfa:.4f} (média)"
        ))
        fig1.add_trace(go.Scatter(
            x=np.concatenate([timespan_days, timespan_days[::-1]]),
            y=np.concatenate([res["high"], res["low"][::-1]]),
            fill='toself',
            fillcolor='rgba(200,100,100,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

    fig1.update_layout(
        title="Impacto da perda de imunidade na mortalidade (Monte Carlo)",
        xaxis_title="Tempo (dias)",
        yaxis_title="Mortos acumulados",
        template="plotly_white"
    )

    fig2 = go.Figure()
    alfas_txt = [f"{a:.4f}" for a in alfa_values]
    final_means = [monte_carlo_results[a]["final_mean"] for a in alfa_values]
    final_lows = [monte_carlo_results[a]["final_low"] for a in alfa_values]
    final_highs = [monte_carlo_results[a]["final_high"] for a in alfa_values]

    fig2.add_trace(go.Bar(
        x=alfas_txt, y=final_means,
        error_y=dict(type='data', array=np.array(final_highs)-np.array(final_means),
                    arrayminus=np.array(final_means)-np.array(final_lows)),
        marker_color='salmon'
    ))

    fig2.update_layout(
        title=f"Mortalidade final por taxa de perda de imunidade (IC {int((1 - (confidence_level / 2)) * 100)}%)",
        xaxis_title="Taxa de perda de imunidade α (por dia)",
        yaxis_title="Mortos acumulados ao final (1 ano)",
        template="plotly_white"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Configure os parâmetros e clique em **Rodar Métricas**.")