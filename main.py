import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mealpy.utils.problem import Problem
from mealpy.utils.space import IntegerVar
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.PSO import OriginalPSO
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Smart Energy Management", layout="wide")
st.title("🏢 AI-Driven Smart Building Energy Management System")

# =========================
# CONSTANTS
# =========================
COST_PER_KWH = 0.12
COMFORT_PENALTY = 120

years = list(range(2019, 2026))
file_paths = {y: {f: f"{y}Floor{f}.csv" for f in range(1, 8)} for y in years}

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data(year, floor):
    df = pd.read_csv(file_paths[year][floor], parse_dates=["Date"])
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙ Configuration")
year = st.sidebar.selectbox("Year", years)
floor = st.sidebar.selectbox("Floor", range(1, 8))
zone = st.sidebar.selectbox("Zone", [1, 2, 3, 4, 5])

st.sidebar.header("📅 Target Date")
month = st.sidebar.selectbox(
    "Month", range(1, 13),
    format_func=lambda m: datetime(2024, m, 1).strftime("%B")
)
day = st.sidebar.selectbox("Day", range(1, 32))

st.sidebar.header("⚡ Interventions")
ac = st.sidebar.selectbox("AC", ["None", "Temperature Adjustment", "Usage Reduction"])
light = st.sidebar.selectbox("Lighting", ["None", "LED Upgrade", "Dimming Control"])
plug = st.sidebar.selectbox("Plug Load", ["None", "Smart Plugs", "Scheduled Usage"])

st.sidebar.header("🌡 Comfort Constraints")
temp_range = st.sidebar.slider("Temperature (°C)", 16, 30, (22, 26))
humidity_range = st.sidebar.slider("Humidity (%)", 20, 80, (40, 60))
light_range = st.sidebar.slider("Light (Lux)", 100, 800, (300, 600))

# =========================
# SAVINGS MAP
# =========================
savings_map = {
    "Temperature Adjustment": 0.08,
    "Usage Reduction": 0.12,
    "LED Upgrade": 0.10,
    "Dimming Control": 0.07,
    "Smart Plugs": 0.05,
    "Scheduled Usage": 0.09
}
total_savings = (
    savings_map.get(ac, 0)
    + savings_map.get(light, 0)
    + savings_map.get(plug, 0)
)

# =========================
# FEATURE ENGINEERING
# =========================
df = load_data(year, floor)

df["Hour"] = df["Date"].dt.hour
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.weekday
df["Year"] = df["Date"].dt.year   # ✅ FIXED BUG

energy_cols = [c for c in df.columns if f"z{zone}_" in c and "kW" in c]
df[f"z{zone}_Energy"] = df[energy_cols].sum(axis=1)

target = f"z{zone}_Energy"

# =========================
# ENERGY MODEL
# =========================
X = df[["Hour", "Day", "Month", "Weekday"]]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

energy_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
energy_model.fit(X_train, y_train)

# =========================
# PRINT METRICS TO CONSOLE
# =========================
y_pred = energy_model.predict(X_test)

print("\n📊 ENERGY MODEL PERFORMANCE")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE : {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R²   : {r2_score(y_test, y_pred):.3f}")

# =========================
# COMFORT MODELS
# =========================
def train_comfort(col):
    if col not in df.columns:
        return None
    model = GradientBoostingRegressor()
    model.fit(X, df[col])
    return model

temp_model = train_comfort(f"z{zone}_Temp")
humidity_model = train_comfort(f"z{zone}_Humidity")
light_model = train_comfort(f"z{zone}_Light")

def safe_predict(model, row, fallback):
    if model is None:
        return np.mean(fallback)
    val = model.predict(row)[0]
    return val if not np.isnan(val) else np.mean(fallback)

# =========================
# OPTIMIZATION OBJECTIVE
# =========================
def objective(solution):
    hours = np.clip(np.round(solution), 0, 23).astype(int)

    X_future = pd.DataFrame({
        "Hour": hours,
        "Day": day,
        "Month": month,
        "Weekday": datetime(year, month, day).weekday()
    })

    energy = energy_model.predict(X_future).sum()
    energy *= (1 - total_savings)

    penalty = 0
    for _, r in X_future.iterrows():
        r_df = r.to_frame().T
        if not temp_range[0] <= safe_predict(temp_model, r_df, temp_range) <= temp_range[1]:
            penalty += COMFORT_PENALTY
        if not humidity_range[0] <= safe_predict(humidity_model, r_df, humidity_range) <= humidity_range[1]:
            penalty += COMFORT_PENALTY
        if not light_range[0] <= safe_predict(light_model, r_df, light_range) <= light_range[1]:
            penalty += COMFORT_PENALTY

    return energy + penalty

# =========================
# OPTIMIZATION
# =========================
problem = Problem(
    obj_func=objective,
    bounds=[IntegerVar(0, 23)] * 3,
    minmax="min"
)

gwo = OriginalGWO(epoch=60, pop_size=40)
pso = OriginalPSO(epoch=60, pop_size=40)

gwo_hours = np.clip(np.round(gwo.solve(problem).solution), 0, 23).astype(int)
pso_hours = np.clip(np.round(pso.solve(problem).solution), 0, 23).astype(int)

optimized_hours = sorted(set(gwo_hours.tolist() + pso_hours.tolist()))[:3]

# =========================
# RESULTS
# =========================
st.subheader("⏱ Optimized Intervention Windows")
st.table(pd.DataFrame({
    "Time Slot": [f"{h}:00 - {h+1}:00" for h in optimized_hours]
}))

# =========================
# COST & IMPACT
# =========================
monthly_energy = df[
    (df["Year"] == year) &
    (df["Month"] == month)
][target].sum()

optimized_energy = monthly_energy * (1 - total_savings)
carbon_saved = (monthly_energy - optimized_energy) * 0.82

col1, col2, col3 = st.columns(3)
col1.metric("Monthly Energy (kWh)", f"{monthly_energy:,.0f}")
col2.metric("Optimized Energy (kWh)", f"{optimized_energy:,.0f}")
col3.metric("Cost Savings ($)", f"{(monthly_energy - optimized_energy) * COST_PER_KWH:,.2f}")

st.metric("🌍 Carbon Reduction (kg CO₂)", f"{carbon_saved:,.1f}")

# =========================
# VISUALIZATIONS
# =========================
st.subheader("📊 Hourly Energy Profile")
hourly = df.groupby("Hour")[target].mean()

fig, ax = plt.subplots()
ax.plot(hourly.index, hourly.values, marker="o")
ax.set_xlabel("Hour")
ax.set_ylabel("Energy (kWh)")
ax.grid(True)
st.pyplot(fig)

st.subheader("📅 Monthly Energy: Original vs Optimized")
monthly = df.groupby("Month")[target].sum()
optimized_monthly = monthly * (1 - total_savings)

fig2, ax2 = plt.subplots()
x = np.arange(len(monthly))
ax2.bar(x - 0.2, monthly, width=0.4, label="Original")
ax2.bar(x + 0.2, optimized_monthly, width=0.4, label="Optimized")
ax2.set_xticks(x)
ax2.set_xticklabels(
    [datetime(2024, m, 1).strftime("%b") for m in monthly.index]
)
ax2.set_ylabel("kWh")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
