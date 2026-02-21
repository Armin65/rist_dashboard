"""Medical Expat Scheme – Risk Dashboard (Streamlit App)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from demo_data import create_demo_excel

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Expat Scheme – Risk Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
    }
    .warning-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .warning-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
    .good-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
    }
    .good-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .good-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_excel(file) -> dict[str, pd.DataFrame]:
    """Read Claims, Premiums and Reserves sheets from an Excel file."""
    xls = pd.ExcelFile(file, engine="openpyxl")
    data = {}

    if "Claims" in xls.sheet_names:
        df = pd.read_excel(xls, "Claims")
        df["Claim_Date"] = pd.to_datetime(df["Claim_Date"])
        df["Year"] = df["Claim_Date"].dt.year
        df["Quarter"] = df["Claim_Date"].dt.quarter
        df["YearQuarter"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)
        df["Month"] = df["Claim_Date"].dt.to_period("M").astype(str)
        data["claims"] = df

    if "Premiums" in xls.sheet_names:
        data["premiums"] = pd.read_excel(xls, "Premiums")

    if "Reserves" in xls.sheet_names:
        data["reserves"] = pd.read_excel(xls, "Reserves")

    return data


def metric_card(label: str, value: str, card_type: str = "metric") -> str:
    return f"""
    <div class="{card_type}-card">
        <h3>{label}</h3>
        <h1>{value}</h1>
    </div>
    """


# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
st.sidebar.title("Data Source")

use_demo = st.sidebar.checkbox("Use demo data", value=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file", type=["xlsx", "xls"],
    help="Upload a file with sheets: Claims, Premiums, Reserves",
)

if uploaded_file is not None:
    data = load_excel(uploaded_file)
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
elif use_demo:
    data = load_excel(create_demo_excel())
    st.sidebar.info("Using demo data")
else:
    st.title("Medical Expat Scheme – Risk Dashboard")
    st.info("Upload an Excel file or enable demo data to get started.")
    st.stop()

claims = data.get("claims", pd.DataFrame())
premiums = data.get("premiums", pd.DataFrame())
reserves = data.get("reserves", pd.DataFrame())

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("Filters")

if not claims.empty:
    all_countries = sorted(claims["Country"].unique())
    selected_countries = st.sidebar.multiselect(
        "Countries", all_countries, default=all_countries,
    )

    all_benefits = sorted(claims["Benefit_Type"].unique())
    selected_benefits = st.sidebar.multiselect(
        "Benefit Types", all_benefits, default=all_benefits,
    )

    year_range = st.sidebar.slider(
        "Year Range",
        int(claims["Year"].min()),
        int(claims["Year"].max()),
        (int(claims["Year"].min()), int(claims["Year"].max())),
    )

    # Apply filters
    mask = (
        claims["Country"].isin(selected_countries)
        & claims["Benefit_Type"].isin(selected_benefits)
        & claims["Year"].between(*year_range)
    )
    fc = claims[mask].copy()

    # Filter premiums
    if not premiums.empty:
        fp = premiums[premiums["Country"].isin(selected_countries)].copy()
    else:
        fp = premiums
else:
    fc = claims
    fp = premiums

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Medical Expat Scheme – Risk Dashboard")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
tab_overview, tab_trends, tab_risk, tab_country, tab_warnings = st.tabs([
    "Overview", "Claims Trend", "Risk Distribution",
    "Country Analysis", "Early Warnings",
])

# ===================== TAB 1: OVERVIEW =====================
with tab_overview:
    if fc.empty:
        st.warning("No claims data available for the selected filters.")
    else:
        total_claims = len(fc)
        total_incurred = fc["Claim_Amount_USD"].sum()
        avg_claim = fc["Claim_Amount_USD"].mean()
        total_premium = fp["Earned_Premium_USD"].sum() if not fp.empty else 0
        loss_ratio = (total_incurred / total_premium * 100) if total_premium > 0 else 0
        frequency = total_claims / fp["Lives_Covered"].mean() if (
            not fp.empty and fp["Lives_Covered"].mean() > 0
        ) else 0

        # Large claims (> 50k)
        large_claims = fc[fc["Claim_Amount_USD"] > 50_000]
        large_claim_pct = (
            large_claims["Claim_Amount_USD"].sum() / total_incurred * 100
            if total_incurred > 0 else 0
        )

        # KPI cards
        cols = st.columns(6)
        card_type_lr = "warning-card" if loss_ratio > 80 else (
            "good-card" if loss_ratio < 60 else "metric-card"
        )
        with cols[0]:
            st.markdown(
                metric_card("Loss Ratio", f"{loss_ratio:.1f}%", card_type_lr),
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                metric_card("Total Claims", f"{total_claims:,}"),
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                metric_card("Total Incurred", f"${total_incurred:,.0f}"),
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown(
                metric_card("Avg Claim", f"${avg_claim:,.0f}"),
                unsafe_allow_html=True,
            )
        with cols[4]:
            st.markdown(
                metric_card("Earned Premium", f"${total_premium:,.0f}"),
                unsafe_allow_html=True,
            )
        with cols[5]:
            lc_type = "warning-card" if large_claim_pct > 40 else "metric-card"
            st.markdown(
                metric_card("Large Claims %", f"{large_claim_pct:.1f}%", lc_type),
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Two charts side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Claims by Benefit Type")
            benefit_agg = (
                fc.groupby("Benefit_Type")
                .agg(Count=("Claim_ID", "count"), Total=("Claim_Amount_USD", "sum"))
                .reset_index()
                .sort_values("Total", ascending=False)
            )
            fig = px.bar(
                benefit_agg, x="Benefit_Type", y="Total",
                text="Count", color="Total",
                color_continuous_scale="Viridis",
                labels={"Total": "Total Incurred (USD)", "Benefit_Type": ""},
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Claims by Status")
            status_agg = (
                fc.groupby("Status")
                .agg(Count=("Claim_ID", "count"), Total=("Claim_Amount_USD", "sum"))
                .reset_index()
            )
            fig = px.pie(
                status_agg, values="Total", names="Status",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Age distribution
        st.subheader("Claims by Age Group")
        fc_age = fc.copy()
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        fc_age["Age_Group"] = pd.cut(fc_age["Claimant_Age"], bins=bins, labels=labels)
        age_agg = (
            fc_age.groupby("Age_Group", observed=True)
            .agg(Count=("Claim_ID", "count"), Avg_Amount=("Claim_Amount_USD", "mean"))
            .reset_index()
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=age_agg["Age_Group"], y=age_agg["Count"], name="# Claims",
                   marker_color="#667eea"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=age_agg["Age_Group"], y=age_agg["Avg_Amount"],
                       name="Avg Claim (USD)", marker_color="#f5576c",
                       mode="lines+markers"),
            secondary_y=True,
        )
        fig.update_layout(height=400, legend=dict(orientation="h", y=1.12))
        fig.update_yaxes(title_text="Number of Claims", secondary_y=False)
        fig.update_yaxes(title_text="Avg Claim Amount (USD)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


# ===================== TAB 2: CLAIMS TREND =====================
with tab_trends:
    if fc.empty:
        st.warning("No claims data available.")
    else:
        st.subheader("Monthly Claims Development")

        monthly = (
            fc.groupby("Month")
            .agg(
                Count=("Claim_ID", "count"),
                Total_Incurred=("Claim_Amount_USD", "sum"),
                Avg_Claim=("Claim_Amount_USD", "mean"),
            )
            .reset_index()
            .sort_values("Month")
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=monthly["Month"], y=monthly["Total_Incurred"],
                   name="Total Incurred", marker_color="#667eea", opacity=0.7),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=monthly["Month"], y=monthly["Count"],
                       name="# Claims", marker_color="#f5576c",
                       mode="lines+markers"),
            secondary_y=True,
        )
        fig.update_layout(height=450, legend=dict(orientation="h", y=1.12))
        fig.update_yaxes(title_text="Total Incurred (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Claims", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # Quarterly loss ratio trend
        if not fp.empty:
            st.subheader("Quarterly Loss Ratio Trend")
            q_claims = (
                fc.groupby("YearQuarter")["Claim_Amount_USD"].sum().reset_index()
            )
            q_claims.columns = ["Period", "Incurred"]
            q_prem = fp.groupby("Period")["Earned_Premium_USD"].sum().reset_index()
            q_prem.columns = ["Period", "Premium"]
            q_merged = q_claims.merge(q_prem, on="Period", how="inner")
            q_merged["Loss_Ratio"] = (
                q_merged["Incurred"] / q_merged["Premium"] * 100
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=q_merged["Period"], y=q_merged["Loss_Ratio"],
                mode="lines+markers+text",
                text=[f"{v:.0f}%" for v in q_merged["Loss_Ratio"]],
                textposition="top center",
                line=dict(color="#667eea", width=3),
                marker=dict(size=10),
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                          annotation_text="Target: 80%")
            fig.add_hline(y=60, line_dash="dash", line_color="green",
                          annotation_text="Good: 60%")
            fig.update_layout(
                height=400,
                yaxis_title="Loss Ratio (%)",
                xaxis_title="Quarter",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Reserve development
        if not reserves.empty:
            st.subheader("Reserve Development")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=reserves["Period"],
                y=reserves["Outstanding_Reserves_USD"],
                name="Outstanding Reserves",
                marker_color="#667eea",
            ))
            fig.add_trace(go.Bar(
                x=reserves["Period"],
                y=reserves["IBNR_USD"],
                name="IBNR",
                marker_color="#f093fb",
            ))
            fig.update_layout(
                barmode="stack", height=400,
                yaxis_title="Reserves (USD)",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)


# ===================== TAB 3: RISK DISTRIBUTION =====================
with tab_risk:
    if fc.empty:
        st.warning("No claims data available.")
    else:
        st.subheader("Claim Amount Distribution")
        fig = px.histogram(
            fc, x="Claim_Amount_USD", nbins=80,
            color="Benefit_Type",
            labels={"Claim_Amount_USD": "Claim Amount (USD)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Loss Ratio by Benefit Type")
            if not fp.empty:
                benefit_incurred = (
                    fc.groupby("Benefit_Type")["Claim_Amount_USD"].sum().reset_index()
                )
                total_prem = fp["Earned_Premium_USD"].sum()
                # Approximate allocation by claim share
                benefit_incurred["Share"] = (
                    benefit_incurred["Claim_Amount_USD"]
                    / benefit_incurred["Claim_Amount_USD"].sum()
                )
                benefit_incurred["Allocated_Premium"] = (
                    benefit_incurred["Share"] * total_prem
                )
                benefit_incurred["Loss_Ratio"] = (
                    benefit_incurred["Claim_Amount_USD"]
                    / benefit_incurred["Allocated_Premium"] * 100
                )
                fig = px.bar(
                    benefit_incurred.sort_values("Loss_Ratio", ascending=False),
                    x="Benefit_Type", y="Loss_Ratio",
                    color="Loss_Ratio",
                    color_continuous_scale="RdYlGn_r",
                    labels={"Loss_Ratio": "Loss Ratio (%)", "Benefit_Type": ""},
                )
                fig.add_hline(y=80, line_dash="dash", line_color="red")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Large Claims Analysis (> $50k)")
            large = fc[fc["Claim_Amount_USD"] > 50_000].copy()
            if not large.empty:
                fig = px.scatter(
                    large, x="Claim_Date", y="Claim_Amount_USD",
                    color="Benefit_Type", size="Claim_Amount_USD",
                    hover_data=["Country", "Claimant_Age"],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No large claims (> $50k) in the filtered data.")

        # Gender split
        st.subheader("Claims by Gender")
        gender_agg = (
            fc.groupby("Gender")
            .agg(Count=("Claim_ID", "count"), Total=("Claim_Amount_USD", "sum"))
            .reset_index()
        )
        gender_agg["Gender"] = gender_agg["Gender"].map({"M": "Male", "F": "Female"})
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(gender_agg, values="Count", names="Gender",
                         title="By Count", hole=0.4,
                         color_discrete_sequence=["#667eea", "#f093fb"])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(gender_agg, values="Total", names="Gender",
                         title="By Amount", hole=0.4,
                         color_discrete_sequence=["#667eea", "#f093fb"])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# ===================== TAB 4: COUNTRY ANALYSIS =====================
with tab_country:
    if fc.empty:
        st.warning("No claims data available.")
    else:
        st.subheader("Claims by Country")

        country_agg = (
            fc.groupby("Country")
            .agg(
                Claim_Count=("Claim_ID", "count"),
                Total_Incurred=("Claim_Amount_USD", "sum"),
                Avg_Claim=("Claim_Amount_USD", "mean"),
                Max_Claim=("Claim_Amount_USD", "max"),
            )
            .reset_index()
            .sort_values("Total_Incurred", ascending=False)
        )

        # Country premium data
        if not fp.empty:
            country_prem = (
                fp.groupby("Country")
                .agg(
                    Total_Premium=("Earned_Premium_USD", "sum"),
                    Avg_Lives=("Lives_Covered", "mean"),
                )
                .reset_index()
            )
            country_agg = country_agg.merge(country_prem, on="Country", how="left")
            country_agg["Loss_Ratio"] = (
                country_agg["Total_Incurred"]
                / country_agg["Total_Premium"] * 100
            ).round(1)
            country_agg["Frequency"] = (
                country_agg["Claim_Count"] / country_agg["Avg_Lives"]
            ).round(2)

        # Map (choropleth)
        country_iso = {
            "United Arab Emirates": "ARE", "Singapore": "SGP",
            "Hong Kong": "HKG", "United Kingdom": "GBR",
            "United States": "USA", "Switzerland": "CHE",
            "Germany": "DEU", "China": "CHN", "Thailand": "THA",
            "Brazil": "BRA", "Saudi Arabia": "SAU", "Nigeria": "NGA",
            "South Africa": "ZAF", "India": "IND", "Australia": "AUS",
        }
        country_agg["ISO"] = country_agg["Country"].map(country_iso)

        fig = px.choropleth(
            country_agg,
            locations="ISO",
            color="Total_Incurred",
            hover_name="Country",
            hover_data=["Claim_Count", "Avg_Claim"],
            color_continuous_scale="YlOrRd",
            labels={"Total_Incurred": "Total Incurred (USD)"},
        )
        fig.update_layout(height=500, geo=dict(showframe=False))
        st.plotly_chart(fig, use_container_width=True)

        # Country table
        st.subheader("Country Details")
        display_cols = [
            "Country", "Claim_Count", "Total_Incurred", "Avg_Claim", "Max_Claim",
        ]
        if "Loss_Ratio" in country_agg.columns:
            display_cols += ["Total_Premium", "Loss_Ratio", "Frequency"]

        st.dataframe(
            country_agg[display_cols].style.format({
                "Total_Incurred": "${:,.0f}",
                "Avg_Claim": "${:,.0f}",
                "Max_Claim": "${:,.0f}",
                "Total_Premium": "${:,.0f}" if "Total_Premium" in display_cols else "",
                "Loss_Ratio": "{:.1f}%" if "Loss_Ratio" in display_cols else "",
            }),
            use_container_width=True,
            height=500,
        )

        # Country comparison
        if "Loss_Ratio" in country_agg.columns:
            st.subheader("Country Risk Matrix")
            fig = px.scatter(
                country_agg, x="Frequency", y="Loss_Ratio",
                size="Total_Incurred", color="Country",
                hover_data=["Claim_Count", "Avg_Claim"],
                labels={
                    "Frequency": "Claim Frequency (per life)",
                    "Loss_Ratio": "Loss Ratio (%)",
                },
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                          annotation_text="Target LR")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)


# ===================== TAB 5: EARLY WARNINGS =====================
with tab_warnings:
    st.subheader("Early Warning Indicators")

    if fc.empty:
        st.warning("No claims data available.")
    else:
        # Configurable thresholds
        with st.expander("Configure Thresholds", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                lr_threshold = st.number_input(
                    "Loss Ratio Warning (%)", value=80, min_value=0, max_value=200,
                )
            with col2:
                large_claim_threshold = st.number_input(
                    "Large Claim Threshold (USD)", value=50000, min_value=0,
                )
            with col3:
                freq_threshold = st.number_input(
                    "Frequency Warning (per life)", value=5.0, min_value=0.0,
                    step=0.5,
                )

        warnings = []

        # 1) Overall Loss Ratio
        if not fp.empty:
            total_incurred = fc["Claim_Amount_USD"].sum()
            total_prem = fp["Earned_Premium_USD"].sum()
            overall_lr = total_incurred / total_prem * 100 if total_prem > 0 else 0
            if overall_lr > lr_threshold:
                warnings.append({
                    "Indicator": "Overall Loss Ratio",
                    "Value": f"{overall_lr:.1f}%",
                    "Threshold": f"{lr_threshold}%",
                    "Status": "CRITICAL" if overall_lr > lr_threshold * 1.2 else "WARNING",
                    "Detail": "Overall loss ratio exceeds target.",
                })

        # 2) Country-level loss ratios
        if not fp.empty:
            c_inc = fc.groupby("Country")["Claim_Amount_USD"].sum()
            c_prem = fp.groupby("Country")["Earned_Premium_USD"].sum()
            c_lr = (c_inc / c_prem * 100).dropna()
            for country, lr in c_lr.items():
                if lr > lr_threshold:
                    warnings.append({
                        "Indicator": f"Loss Ratio – {country}",
                        "Value": f"{lr:.1f}%",
                        "Threshold": f"{lr_threshold}%",
                        "Status": "CRITICAL" if lr > lr_threshold * 1.2 else "WARNING",
                        "Detail": f"{country} exceeds target loss ratio.",
                    })

        # 3) Large claims concentration
        large = fc[fc["Claim_Amount_USD"] > large_claim_threshold]
        if len(large) > 0:
            large_pct = large["Claim_Amount_USD"].sum() / fc["Claim_Amount_USD"].sum() * 100
            if large_pct > 40:
                warnings.append({
                    "Indicator": "Large Claims Concentration",
                    "Value": f"{large_pct:.1f}%",
                    "Threshold": "40%",
                    "Status": "CRITICAL" if large_pct > 60 else "WARNING",
                    "Detail": (
                        f"{len(large)} large claims (> ${large_claim_threshold:,}) "
                        f"account for {large_pct:.1f}% of total incurred."
                    ),
                })

        # 4) Recent trend (last quarter vs. previous)
        if len(fc["YearQuarter"].unique()) >= 2:
            quarters = sorted(fc["YearQuarter"].unique())
            last_q = fc[fc["YearQuarter"] == quarters[-1]]["Claim_Amount_USD"].sum()
            prev_q = fc[fc["YearQuarter"] == quarters[-2]]["Claim_Amount_USD"].sum()
            if prev_q > 0:
                change = (last_q - prev_q) / prev_q * 100
                if change > 20:
                    warnings.append({
                        "Indicator": "Quarter-over-Quarter Change",
                        "Value": f"+{change:.1f}%",
                        "Threshold": "+20%",
                        "Status": "CRITICAL" if change > 40 else "WARNING",
                        "Detail": (
                            f"Incurred claims increased {change:.1f}% from "
                            f"{quarters[-2]} to {quarters[-1]}."
                        ),
                    })

        # 5) IBNR growth
        if not reserves.empty and len(reserves) >= 2:
            last_ibnr = reserves.iloc[-1]["IBNR_USD"]
            prev_ibnr = reserves.iloc[-2]["IBNR_USD"]
            if prev_ibnr > 0:
                ibnr_change = (last_ibnr - prev_ibnr) / prev_ibnr * 100
                if ibnr_change > 15:
                    warnings.append({
                        "Indicator": "IBNR Growth",
                        "Value": f"+{ibnr_change:.1f}%",
                        "Threshold": "+15%",
                        "Status": "CRITICAL" if ibnr_change > 30 else "WARNING",
                        "Detail": "IBNR reserves growing faster than expected.",
                    })

        # Display warnings
        if not warnings:
            st.success("No warnings – all indicators within acceptable ranges.")
        else:
            # Summary
            critical = sum(1 for w in warnings if w["Status"] == "CRITICAL")
            warning_count = sum(1 for w in warnings if w["Status"] == "WARNING")

            cols = st.columns(3)
            with cols[0]:
                st.markdown(
                    metric_card("Total Alerts", str(len(warnings))),
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(
                    metric_card("Critical", str(critical), "warning-card"),
                    unsafe_allow_html=True,
                )
            with cols[2]:
                st.markdown(
                    metric_card("Warnings", str(warning_count), "metric-card"),
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Warning table
            warn_df = pd.DataFrame(warnings)
            warn_df = warn_df.sort_values(
                "Status", key=lambda x: x.map({"CRITICAL": 0, "WARNING": 1}),
            )

            for _, row in warn_df.iterrows():
                icon = "🔴" if row["Status"] == "CRITICAL" else "🟡"
                with st.expander(
                    f"{icon} {row['Status']}: {row['Indicator']} = {row['Value']}"
                ):
                    st.write(f"**Threshold:** {row['Threshold']}")
                    st.write(f"**Detail:** {row['Detail']}")

        # Trend sparklines for key indicators
        if not fp.empty:
            st.markdown("---")
            st.subheader("Quarterly Indicator Trends")

            q_claims = fc.groupby("YearQuarter")["Claim_Amount_USD"].sum().reset_index()
            q_claims.columns = ["Period", "Incurred"]
            q_count = fc.groupby("YearQuarter")["Claim_ID"].count().reset_index()
            q_count.columns = ["Period", "Count"]
            q_prem = fp.groupby("Period")["Earned_Premium_USD"].sum().reset_index()
            q_prem.columns = ["Period", "Premium"]

            q_all = q_claims.merge(q_prem, on="Period", how="inner")
            q_all = q_all.merge(q_count, on="Period", how="inner")
            q_all["Loss_Ratio"] = q_all["Incurred"] / q_all["Premium"] * 100
            q_all["Avg_Claim"] = q_all["Incurred"] / q_all["Count"]
            q_all = q_all.sort_values("Period")

            col1, col2, col3 = st.columns(3)
            with col1:
                fig = px.line(q_all, x="Period", y="Loss_Ratio",
                              title="Loss Ratio Trend")
                fig.add_hline(y=lr_threshold, line_dash="dash", line_color="red")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.line(q_all, x="Period", y="Count",
                              title="Claim Count Trend")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col3:
                fig = px.line(q_all, x="Period", y="Avg_Claim",
                              title="Avg Claim Trend")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar: download template
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("Tools")

if st.sidebar.button("Download Excel Template"):
    template_buffer = create_demo_excel()
    st.sidebar.download_button(
        "Download",
        data=template_buffer,
        file_name="risk_dashboard_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.sidebar.markdown("---")
st.sidebar.caption("Medical Expat Scheme – Risk Dashboard v1.0")
