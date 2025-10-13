# ===================== D) Stress Testing (Factor-level, sector & systemic) =====================
st.subheader("D. Stress Testing")

# Ensure sector_raw is valid and not empty
if not sector_raw or pd.isna(sector_raw.strip()):
    sector_raw = "__default__"  # Default value for invalid or empty sector

# Debugging: log the sector_raw value
st.write(f"Sector raw: {sector_raw}")  # Display sector for debugging purposes

# ------ Define realistic sector-specific crisis scenarios --------
SECTOR_CRISIS_SCENARIOS = {
    "Technology": {
        "Tech Crunch": {
            "Revenue_CAGR_3Y": 0.70,  # Impacted by market slowdowns
            "ROA": 0.60,              # Decreased profitability
            "Net_Profit_Margin": 0.50, # Lower margins
            "Interest_Coverage": 0.70,
            "Debt_to_Equity": 1.05,
            "EBITDA_to_Interest": 0.65,
            "Sentiment Score": 0.70    # Sentiment downturn
        },
        "Supply Chain Disruption": {
            "Revenue_CAGR_3Y": 0.80,
            "ROA": 0.75,
            "Net_Profit_Margin": 0.70,
            "Interest_Coverage": 0.75,
            "Debt_to_Equity": 1.10,
            "EBITDA_to_Interest": 0.75,
            "Sentiment Score": 0.75
        },
        "Pandemic Shock": {
            "Revenue_CAGR_3Y": 0.60,
            "ROA": 0.55,
            "Net_Profit_Margin": 0.50,
            "Interest_Coverage": 0.60,
            "Debt_to_Equity": 1.15,
            "EBITDA_to_Interest": 0.60,
            "Sentiment Score": 0.65
        }
    },
    "Real Estate": {
        "Housing Downturn": {
            "Revenue_CAGR_3Y": 0.75,  # Impact of lower sales
            "ROA": 0.65,              # Profitability decrease
            "Net_Profit_Margin": 0.60,
            "Interest_Coverage": 0.70,
            "Debt_to_Equity": 1.30,
            "EBITDA_to_Interest": 0.80,
            "Sentiment Score": 0.60
        },
        "Credit Crunch": {
            "Revenue_CAGR_3Y": 0.70,
            "ROA": 0.60,
            "Net_Profit_Margin": 0.55,
            "Interest_Coverage": 0.65,
            "Debt_to_Equity": 1.25,
            "EBITDA_to_Interest": 0.75,
            "Sentiment Score": 0.65
        },
        "Government Policy Change": {
            "Revenue_CAGR_3Y": 0.80,
            "ROA": 0.70,
            "Net_Profit_Margin": 0.65,
            "Interest_Coverage": 0.75,
            "Debt_to_Equity": 1.20,
            "EBITDA_to_Interest": 0.85,
            "Sentiment Score": 0.70
        }
    },
    # Add more specific scenarios for other sectors here

    # Default fallback for sectors not defined in the dictionary
    "__default__": {
        "General Crisis": {
            "Revenue_CAGR_3Y": 0.70,
            "ROA": 0.65,
            "Net_Profit_Margin": 0.60,
            "Interest_Coverage": 0.65,
            "Debt_to_Equity": 1.20,
            "EBITDA_to_Interest": 0.70,
            "Sentiment Score": 0.60
        }
    }
}

# -------------------------------------------------------------
# Select appropriate sector-based stress scenarios based on user input
def build_sector_scenarios(sector_name: str) -> dict:
    if sector_name in SECTOR_CRISIS_SCENARIOS:
        return SECTOR_CRISIS_SCENARIOS[sector_name]
    else:
        # Default to general crisis scenarios if sector is unknown
        return SECTOR_CRISIS_SCENARIOS["__default__"]

# -------------------------------------------------------------
# Define scale_multiplier function to adjust the crisis impact
def scale_multiplier(factor_dict: dict, sev: float, exch_int: float) -> dict:
    """
    Scales the crisis impact multipliers for each factor in the crisis dictionary
    based on severity and exchange intensity.

    :param factor_dict: Dictionary of crisis factors for each scenario (e.g., "Revenue_CAGR_3Y", "ROA").
    :param sev: The severity multiplier (e.g., 1.0 for moderate, 1.5 for severe).
    :param exch_int: The exchange intensity multiplier based on the selected exchange.
    :return: Scaled crisis factor dictionary.
    """
    # Scale each factor in the dictionary
    return {key: value * sev * exch_int for key, value in factor_dict.items()}

# -------------------------------------------------------------
# Add a slider to select the severity of the crisis
severity_label = st.selectbox("Select Crisis Severity", ["Mild", "Moderate", "Severe"])
severity_map = {"Mild": 0.6, "Moderate": 1.0, "Severe": 1.5}
sev = severity_map.get(severity_label, 1.0)  # Default to moderate if not selected

# Get the exchange intensity from the user or use the default based on exchange
exch_intensity = EXCHANGE_INTENSITY.get(exchange, 1.0)  # Default value for exchange intensity

# -------------------------------------------------------------
# Retrieve the row of features for the selected company and year (X_base_row)
X_base_row = model_align_row(row_model, model, fallbacks=final_features)
X_base_row = align_features_to_model(X_base_row, model)  # Ensure the row matches model's expected input format

# -------------------------------------------------------------
# Run stress test and apply dynamic factors to the model input
sector_scenarios = build_sector_scenarios(sector_raw)

# Scale the crisis multipliers based on selected severity and exchange intensity
sector_scenarios_scaled = {scenario: scale_multiplier(factor, sev, exch_intensity) 
                           for scenario, factor in sector_scenarios.items()}

# -------------------------------------------------------------
# Define run_scenarios function to apply the crisis factors and get PD values
def run_scenarios(model, X_base_row, sector_scenarios_scaled):
    """
    This function takes the model, base data row, and scaled sector scenarios
    and applies the factors to compute the Probability of Default (PD) for each scenario.
    """
    results = []
    for scenario, factors in sector_scenarios_scaled.items():
        # Apply the scaled factors to the base row
        X_scaled = apply_multipliers_once(X_base_row.copy(), factors)
        
        # Get PD value for this scenario
        pd_value = score_pd(model, X_scaled)
        
        # Append the result
        results.append({"Scenario": scenario, "PD": pd_value})
    
    return pd.DataFrame(results)

# -------------------------------------------------------------
# Run the scenario test and get PD (Probability of Default) values for each scenario
df_sector = run_scenarios(model, X_base_row, sector_scenarios_scaled)

# -------------------------------------------------------------
# Visualization of the stress test results for sector-specific crises
if not df_sector.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_sector["Scenario"], y=df_sector["PD"]))
    fig.update_layout(
        title=f"Sector Crisis Impact — {sector_raw}",
        yaxis=dict(tickformat=".0%"),
        height=340
    )
    st.plotly_chart(fig, width='stretch')  # Updated to match Streamlit's new API
else:
    st.info("No sector scenarios generated results.")
    
# ---------- Monte Carlo CVaR ----------
st.markdown("**Monte Carlo CVaR 95%**")

mc_results = mc_cvar_pd(model, X_base_row, feats_df, sims=sim_count, alpha=0.95)

if isinstance(mc_results, dict) and "PD_sims" in mc_results:
    pd_var = mc_results["VaR"]
    pd_cvar = mc_results["CVaR"]
    st.metric("VaR 95% (PD)", f"{pd_var:.2%}")
    st.metric("CVaR 95% (PD)", f"{pd_cvar:.2%}")
else:
    st.warning("Monte Carlo CVaR simulation failed.")
