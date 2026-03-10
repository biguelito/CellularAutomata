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

    fig = seirsd.solve(
        days=days,
        initial_conditions=initial_conditions,
        transfer_rates=transfer_rates,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Configure os parâmetros e clique em **Rodar Simulação**.")

st.subheader("Simular SEIRSD com variações em α")

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

    results = {}

    for alfa in alfa_values:
        rates = (beta, sigma, gamma, alfa, mu)
        sol = odeint(seirsd.odes, initial_conditions, timespan_days, args=(rates,))
        S, E, I, R, D = sol.T
        results[alfa] = {
            "t": timespan_days,
            "S": S,
            "E": E,
            "I": I,
            "R": R,
            "D": D,
            "total_deaths": D[-1]
        }

    fig1 = go.Figure()

    for alfa, data in results.items():
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
    final_deaths = [results[w]["total_deaths"] for w in alfa_values]
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

    def sample_truncnorm(mean, sd, lower, upper, size):
        a, b = (lower - mean) / sd, (upper - mean) / sd
        return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size, random_state=42)

    N_sim = 1000  # por valor de alfa

    cv = 0.1
    results = {}

    for alfa in alfa_values:
        D_curves = []
        D_final = []
        
        sigma_samples = sample_truncnorm(sigma, sigma*cv, sigma*0.5, sigma*1.5, N_sim)
        gamma_samples = sample_truncnorm(gamma, gamma*cv, gamma*0.5, gamma*1.5, N_sim)
        mu_samples = sample_truncnorm(mu, mu*cv, mu*0.5, mu*1.5, N_sim)
        beta_samples = sample_truncnorm(beta, beta*cv, beta*0.5, beta*1.5, N_sim)
        
        for i in range(N_sim):
            sigma = sigma_samples[i]
            gamma = gamma_samples[i]
            mu = mu_samples[i]
            beta = beta_samples[i]
            rates = (beta, sigma, gamma, alfa, mu)

            sol = odeint(seirsd.odes, initial_conditions, timespan_days, args=(rates, ))
            D_t = sol[:, 4]
            
            D_curves.append(D_t)
            D_final.append(D_t[-1])
        
        D_curves = np.array(D_curves)
        D_final = np.array(D_final)
        
        mean_curve = np.mean(D_curves, axis=0)
        low_curve = np.percentile(D_curves, 2.5, axis=0)
        high_curve = np.percentile(D_curves, 97.5, axis=0)
        
        results[alfa] = {
            "mean": mean_curve,
            "low": low_curve,
            "high": high_curve,
            "final_mean": np.mean(D_final),
            "final_low": np.percentile(D_final, 2.5),
            "final_high": np.percentile(D_final, 97.5)
        }

    fig1 = go.Figure()
    for alfa in alfa_values:
        res = results[alfa]
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
    final_means = [results[a]["final_mean"] for a in alfa_values]
    final_lows = [results[a]["final_low"] for a in alfa_values]
    final_highs = [results[a]["final_high"] for a in alfa_values]

    fig2.add_trace(go.Bar(
        x=alfas_txt, y=final_means,
        error_y=dict(type='data', array=np.array(final_highs)-np.array(final_means),
                    arrayminus=np.array(final_means)-np.array(final_lows)),
        marker_color='salmon'
    ))

    fig2.update_layout(
        title="Mortalidade final por taxa de perda de imunidade (IC 95%)",
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