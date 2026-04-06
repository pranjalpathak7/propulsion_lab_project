import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Combustor Optimization Dashboard", layout="wide", page_icon="🔥")

# 2. Load the Pre-trained Models
@st.cache_resource
def load_models():
    co_model = joblib.load('krr_co_model.pkl')
    nox_model = joblib.load('svr_nox_model.pkl')
    return co_model, nox_model

champion_co, champion_nox = load_models()

# 3. Sidebar: Intelligent User Input
st.sidebar.title("🧪 Fuel Blend Configurator")
st.sidebar.markdown("Adjust the fractions of Biodiesel and Jet A-1. Ethanol will auto-calculate to ensure a 100% mixture.")

# Logic: F3 automatically adapts to F1 and F2
f1_perc = st.sidebar.slider("Biodiesel (F1) %", min_value=0.0, max_value=100.0, value=33.3, step=0.1)
# Restrict Jet A-1 so the user cannot exceed 100% total
f2_perc = st.sidebar.slider("Jet A-1 (F2) %", min_value=0.0, max_value=100.0 - f1_perc, value=33.3, step=0.1)
f3_perc = 100.0 - f1_perc - f2_perc

st.sidebar.metric(label="Ethanol (F3) Auto-Calculated", value=f"{f3_perc:.1f} %")

# Predict based on user input
user_X = np.array([[f1_perc/100.0, f2_perc/100.0, f3_perc/100.0]])
pred_co = np.exp(champion_co.predict(user_X)[0])
pred_nox = np.exp(champion_nox.predict(user_X)[0])

st.sidebar.markdown("---")
st.sidebar.header("📊 Real-Time Predictions")
st.sidebar.metric(label="Predicted CO", value=f"{pred_co:.1f} ppm", delta="Normalized to 3kW")
st.sidebar.metric(label="Predicted NOx", value=f"{pred_nox:.2f} ppm", delta="Normalized to 3kW", delta_color="inverse")

# 4. Main Dashboard Area
st.title("Triple-Blend Combustor Optimization")
st.markdown("Explore the thermodynamic response surfaces and multi-objective optimization for Biodiesel, Jet A-1, and Ethanol blends.")

tab1, tab2 = st.tabs(["🗺️ Interactive Response Surfaces", "📈 Multi-Objective Optimization"])

# 5. Generate High-Res Grid Data (Cached for speed)
@st.cache_data
def generate_grid_data():
    resolution = 80 # Slightly lower resolution for web speed
    grid = []
    for i in np.linspace(0, 1, resolution):
        for j in np.linspace(0, 1 - i, resolution):
            k = 1.0 - i - j
            if np.abs(i + j + k - 1.0) < 1e-5:
                grid.append([i, j, k])
    X_grid = np.array(grid)
    
    co_preds = np.exp(champion_co.predict(X_grid))
    nox_preds = np.exp(champion_nox.predict(X_grid))
    
    df = pd.DataFrame({
        'Biodiesel (F1)': X_grid[:, 0],
        'Jet A-1 (F2)': X_grid[:, 1],
        'Ethanol (F3)': X_grid[:, 2],
        'CO (ppm)': co_preds,
        'NOx (ppm)': nox_preds
    })
    return df

df_grid = generate_grid_data()

# 6. Tab 1: Ternary Heatmaps
with tab1:
    col1, col2 = st.columns(2)
    
    def plot_ternary(df, target, title, color_scale):
        fig = px.scatter_ternary(
            df, a='Biodiesel (F1)', b='Jet A-1 (F2)', c='Ethanol (F3)', 
            color=target, color_continuous_scale=color_scale, title=title
        )
        fig.update_traces(marker=dict(size=8, symbol='hexagon', line=dict(width=0)))
        fig.update_layout(ternary=dict(sum=1), height=500, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    with col1:
        st.plotly_chart(plot_ternary(df_grid, 'CO (ppm)', 'Carbon Monoxide Map', 'Viridis'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_ternary(df_grid, 'NOx (ppm)', 'Nitrogen Oxides Map', 'Inferno'), use_container_width=True)

# 7. Tab 2: The Pareto Front
with tab2:
    st.subheader("CO vs. NOx Trade-off (Pareto Front)")
    st.markdown("This scatter plot maps 5,000 virtual fuel blends, demonstrating the inverse relationship between complete combustion (low CO) and Zeldovich thermal pathways (high NOx).")
    
    fig_pareto = px.scatter(
        df_grid, x='NOx (ppm)', y='CO (ppm)', color='CO (ppm)', 
        hover_data=['Biodiesel (F1)', 'Jet A-1 (F2)', 'Ethanol (F3)'],
        color_continuous_scale='plasma'
    )
    fig_pareto.update_layout(height=600)
    st.plotly_chart(fig_pareto, use_container_width=True)