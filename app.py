import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import os


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AMR Intelligence Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# PLOTLY CONFIG
# ─────────────────────────────────────────────
PLOTLY_CONFIG = {"displayModeBar": False}


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #c0392b;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 4px;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        border-left: 4px solid #c0392b;
        padding-left: 12px;
        margin: 20px 0 15px 0;
    }

    .small-note {
        font-size: 0.85rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILE LOCATIONS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TS_DIR = os.path.join(BASE_DIR, "By timeseries")
RES_DIR = os.path.join(BASE_DIR, "By antibiotic")
MAP_DIR = os.path.join(BASE_DIR, "Map")


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("%", "", regex=False),
        errors="coerce"
    )


def standardize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("\n", "", regex=False)
        .str.replace("\r", "", regex=False)
    )
    return df


def clip_percent_column(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0, upper=100)
    return df


def safe_plotly_chart(fig):
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_all_data():

    def parse_ts_country(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        country_lines = []
        in_country = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('"Iso3"') or stripped.startswith("Iso3,"):
                in_country = True

            if in_country:
                country_lines.append(line)

        if not country_lines:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO("".join(country_lines)), on_bad_lines="skip")
        df = standardize_columns(df)

        for col in [
            "Year",
            "PercentResistant",
            "TotalSpecimenIsolates",
            "InterpretableAST",
            "Resistant",
            "PercentAST"
        ]:
            if col in df.columns:
                df[col] = clean_numeric(df[col])

        df = clip_percent_column(df, "PercentResistant")
        df = clip_percent_column(df, "PercentAST")

        return df


    def parse_ts_summary(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        summary_lines = []
        in_summary = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('"Year"') or stripped.startswith("Year,"):
                in_summary = True

            if in_summary:
                summary_lines.append(line)

            if in_summary and stripped.startswith('"Iso3"'):
                break

        if not summary_lines:
            return pd.DataFrame()

        text = "".join(summary_lines)
        text = text.split('"Iso3"')[0]
        df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        df = standardize_columns(df)

        for col in ["Year", "Median", "Min", "Max", "Q1", "Q3", "CTAs", "BCIs"]:
            if col in df.columns:
                df[col] = clean_numeric(df[col])

        for col in ["Median", "Min", "Max", "Q1", "Q3"]:
            df = clip_percent_column(df, col)

        return df


    def parse_resistance(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith('"Specimen"') or stripped.startswith("Specimen,"):
                df = pd.read_csv(io.StringIO("".join(lines[i:])), on_bad_lines="skip")
                df = standardize_columns(df)

                for col in [
                    "Median",
                    "Q1",
                    "Q3",
                    "Min",
                    "Max",
                    "CTAsCount",
                    "TotalBCIsWithAST",
                    "TotalBCIs",
                    "BCIsWithAST"
                ]:
                    if col in df.columns:
                        df[col] = clean_numeric(df[col])

                for col in ["Median", "Q1", "Q3", "Min", "Max"]:
                    df = clip_percent_column(df, col)

                return df

        return pd.DataFrame()


    def parse_map(path, year):
        df = pd.read_csv(path, skiprows=4, on_bad_lines="skip")
        df = standardize_columns(df)
        df["Year"] = year

        for col in df.columns:
            if col not in ["CountryTerritoryArea", "Iso3", "WHORegionName"]:
                df[col] = clean_numeric(df[col])

        return df


    # ───────────── Time series files ─────────────
    ts_files = [
        "ts_Ecoli_Cipro.csv",
        "ts_Ecoli_Meropenem.csv",
        "ts_Kpneumoniae_Meropenem.csv",
        "ts_Kpneumoniae_Ceftriaxone.csv",
        "ts_Salmonella_Cipro.csv",
        "ts_Acinetobacter_Meropenem.csv",
    ]

    ts_dfs = []
    ts_summary_dfs = {}

    for filename in ts_files:
        path = os.path.join(TS_DIR, filename)

        if os.path.exists(path):
            df_country = parse_ts_country(path)
            df_summary = parse_ts_summary(path)

            if not df_country.empty:
                ts_dfs.append(df_country)

            ts_summary_dfs[filename] = df_summary

    ts_all = pd.concat(ts_dfs, ignore_index=True) if ts_dfs else pd.DataFrame()


    # ───────────── Resistance 2023 files ─────────────
    res_files = [
        "resistance2023_Escherichia coli.csv",
        "resistance2023_Kpneumoniae.csv",
        "resistance2023_Acinetobacter.csv",
        "resistance2023_Salmonella.csv",
        "resistance2023_Spneumoniae.csv",
    ]

    res_dfs = []

    for filename in res_files:
        path = os.path.join(RES_DIR, filename)

        if os.path.exists(path):
            df = parse_resistance(path)

            if not df.empty:
                res_dfs.append(df)

    res_all = pd.concat(res_dfs, ignore_index=True) if res_dfs else pd.DataFrame()

    if not res_all.empty and "PathogenName" in res_all.columns:
        res_all = res_all[
            res_all["PathogenName"].notna()
            & (res_all["PathogenName"] != "PathogenName")
        ]


    # ───────────── Map files ─────────────
    map_years = [2019, 2020, 2021, 2022, 2023]
    map_dfs = []

    for year in map_years:
        filename = f"Global maps of testing coverage by infection type_{year}-BLOOD.csv"
        path = os.path.join(MAP_DIR, filename)

        if os.path.exists(path):
            df = parse_map(path, year)
            map_dfs.append(df)

    map_all = pd.concat(map_dfs, ignore_index=True) if map_dfs else pd.DataFrame()

    return ts_all, res_all, map_all, ts_summary_dfs


ts_all, res_all, map_all, ts_summary_dfs = load_all_data()


# ─────────────────────────────────────────────
# DATA CHECK
# ─────────────────────────────────────────────
missing_messages = []

if ts_all.empty:
    missing_messages.append("No time-series data loaded. Check the folder name: By timeseries")

if res_all.empty:
    missing_messages.append("No resistance 2023 data loaded. Check the folder name: By antibiotic")

if map_all.empty:
    missing_messages.append("No map data loaded. Check the folder name: Map")

if missing_messages:
    st.error("Some data files were not loaded.")

    for msg in missing_messages:
        st.warning(msg)

    st.info(f"""
    Current app folder:
    `{BASE_DIR}`

    Expected folders:
    - `{TS_DIR}`
    - `{RES_DIR}`
    - `{MAP_DIR}`
    """)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦠 AMR Dashboard")
    st.markdown("**WHO GLASS Data · 2018–2023**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🌍 Global Map",
            "📈 Resistance Trends",
            "🦠 Pathogen Analysis",
            "💊 Antibiotic Resistance Ranking",
            "🔬 Country Explorer"
        ]
    )

    st.markdown("---")
    st.markdown("**Data Source**")
    st.markdown(
        "[Global antimicrobial resistance data · Bloodstream infections · 2016–2023]"
        "(https://worldhealthorg.shinyapps.io/glass-dashboard/_w_c2889df754ce4e3fbce33d10b3fc0e7c/#!/amr)"
    )

    st.markdown("---")
    st.caption("MSBA382 · Healthcare Analytics")


# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":

    st.markdown("# 🦠 Global Antibiotic Resistance Surveillance Dashboard")
    st.markdown("##### Bloodstream Infection Surveillance · WHO GLASS · 2018–2023")
    st.caption(
        "This dashboard supports AMR surveillance and public-health planning. "
        "It is not an individual clinical treatment tool."
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    n_countries = (
        map_all["CountryTerritoryArea"].nunique()
        if not map_all.empty and "CountryTerritoryArea" in map_all.columns
        else 0
    )
    n_pathogens = (
        res_all["PathogenName"].nunique()
        if not res_all.empty and "PathogenName" in res_all.columns
        else 0
    )
    n_antibiotics = (
        res_all["AntibioticName"].nunique()
        if not res_all.empty and "AntibioticName" in res_all.columns
        else 0
    )
    n_ts_countries = (
        ts_all["CountryTerritoryArea"].nunique()
        if not ts_all.empty and "CountryTerritoryArea" in ts_all.columns
        else 0
    )

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{n_countries}</div>
            <div class='metric-label'>Countries in Map Data</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{n_pathogens}</div>
            <div class='metric-label'>Pathogens</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{n_antibiotics}</div>
            <div class='metric-label'>Antibiotics</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{n_ts_countries}</div>
            <div class='metric-label'>Countries in Trend Data</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            "<div class='section-header'>Resistance Heatmap by Pathogen and Antibiotic</div>",
            unsafe_allow_html=True
        )

        if not res_all.empty and {"AntibioticName", "PathogenName", "Median"}.issubset(res_all.columns):
            df_heat = res_all.dropna(subset=["Median"]).copy()

            key_antibiotics = [
                "Ciprofloxacin",
                "Meropenem",
                "Ceftriaxone",
                "Cefotaxime",
                "Ceftazidime",
                "Ampicillin",
                "Co-trimoxazole",
                "Imipenem",
                "Vancomycin",
                "Oxacillin"
            ]

            df_heat = df_heat[df_heat["AntibioticName"].isin(key_antibiotics)]

            df_heat = df_heat.groupby(
                ["AntibioticName", "PathogenName"],
                as_index=False
            ).agg(
                Median=("Median", "mean")
            )

            df_heat["Median"] = df_heat["Median"].clip(lower=0, upper=100)

            pivot = df_heat.pivot_table(
                index="AntibioticName",
                columns="PathogenName",
                values="Median",
                aggfunc="mean"
            )

            if not pivot.empty:
                fig = px.imshow(
                    pivot,
                    color_continuous_scale="RdYlGn_r",
                    aspect="auto",
                    labels={"color": "Median Resistance %"},
                    title="Median Resistance Percentage, 2023"
                )

                fig.update_layout(height=440)
                safe_plotly_chart(fig)
            else:
                st.info("No matching antibiotic data found for the heatmap.")
        else:
            st.info("Resistance data not available.")

    with col_right:
        st.markdown(
            "<div class='section-header'>Top Resistant Pathogen–Antibiotic Pairs</div>",
            unsafe_allow_html=True
        )

        if not res_all.empty and {"PathogenName", "AntibioticName", "Median"}.issubset(res_all.columns):
            df_top = res_all.dropna(subset=["Median"]).copy()
            df_top["Combo"] = df_top["PathogenName"] + " · " + df_top["AntibioticName"]

            if "TotalBCIsWithAST" in df_top.columns:
                df_top = df_top.groupby("Combo", as_index=False).agg(
                    Median=("Median", "mean"),
                    TotalBCIsWithAST=("TotalBCIsWithAST", "sum")
                )
            else:
                df_top = df_top.groupby("Combo", as_index=False).agg(
                    Median=("Median", "mean")
                )

            df_top["Median"] = df_top["Median"].clip(lower=0, upper=100)

            cols_to_show = ["Combo", "Median"]

            if "TotalBCIsWithAST" in df_top.columns:
                cols_to_show.append("TotalBCIsWithAST")

            df_top = df_top.nlargest(10, "Median")[cols_to_show]

            rename_dict = {
                "Combo": "Pathogen · Antibiotic",
                "Median": "Median Resistance %",
                "TotalBCIsWithAST": "Isolates Tested"
            }

            df_top = df_top.rename(columns=rename_dict)
            df_top["Median Resistance %"] = df_top["Median Resistance %"].round(1)

            st.dataframe(df_top, use_container_width=True, hide_index=True)
        else:
            st.info("Resistance data not available.")

    st.markdown("---")

    st.markdown("<div class='section-header'>Healthcare Problem</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "**The Problem**\n\n"
            "Antimicrobial resistance occurs when bacteria become resistant to antibiotics, making infections harder to treat."
        )

    with c2:
        st.warning(
            "**Why It Matters**\n\n"
            "Bloodstream infections can become life-threatening when first-line antibiotics no longer work effectively."
        )

    with c3:
        st.success(
            "**Dashboard Goal**\n\n"
            "This dashboard helps identify which pathogens, antibiotics, and countries show concerning resistance patterns."
        )


# ─────────────────────────────────────────────
# PAGE 2 — GLOBAL MAP
# ─────────────────────────────────────────────
elif page == "🌍 Global Map":

    st.markdown("# 🌍 Global Testing Coverage Map")
    st.markdown("Bloodstream infection surveillance capacity by country and year")
    st.markdown("---")

    if map_all.empty:
        st.error("Map data was not loaded.")
    else:
        col1, col2 = st.columns([1, 3])

        with col1:
            selected_year = st.selectbox(
                "Select Year",
                sorted(map_all["Year"].dropna().unique()),
                index=len(sorted(map_all["Year"].dropna().unique())) - 1
            )

            possible_metrics = [
                "BCIsPerMillion",
                "TotalSpecimenIsolates",
                "ASTResult",
                "TotalBCIspermillionpopulation",
                "AbsolutenumberofBCIs",
                "TotalBCIswithASTpermillionpopulation",
                "AbsolutenumberofBCIswithAST",
                "%ofBCIswithAST"
            ]

            available_metrics = [m for m in possible_metrics if m in map_all.columns]

            if available_metrics:
                metric = st.selectbox(
                    "Map Metric",
                    available_metrics,
                    format_func=lambda x: {
                        "BCIsPerMillion": "BCIs per Million",
                        "TotalSpecimenIsolates": "Total Isolates",
                        "ASTResult": "AST Coverage %",
                        "TotalBCIspermillionpopulation": "Total BCIs per Million",
                        "AbsolutenumberofBCIs": "Absolute BCIs",
                        "TotalBCIswithASTpermillionpopulation": "BCIs with AST per Million",
                        "AbsolutenumberofBCIswithAST": "Absolute BCIs with AST",
                        "%ofBCIswithAST": "% BCIs with AST"
                    }.get(x, x)
                )
            else:
                metric = None
                st.warning("No valid map metric column was found.")

        if metric is not None:
            df_map = map_all[map_all["Year"] == selected_year].copy()
            df_map = df_map.dropna(subset=[metric])

            fig = px.choropleth(
                df_map,
                locations="CountryTerritoryArea",
                locationmode="country names",
                color=metric,
                hover_name="CountryTerritoryArea",
                color_continuous_scale="YlOrRd",
                title=f"Global AMR Testing Coverage · {selected_year}",
            )

            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
                height=560,
                margin=dict(t=50, b=0, l=0, r=0)
            )

            with col2:
                safe_plotly_chart(fig)

            st.markdown("---")
            st.markdown("<div class='section-header'>Testing Coverage Over Time</div>", unsafe_allow_html=True)

            if "CountryTerritoryArea" in map_all.columns:
                agg_dict = {"Countries": ("CountryTerritoryArea", "nunique")}

                if metric in map_all.columns:
                    agg_dict["Average_Metric"] = (metric, "mean")

                coverage = map_all.groupby("Year").agg(**agg_dict).reset_index()

                c1, c2 = st.columns(2)

                with c1:
                    fig1 = px.bar(
                        coverage,
                        x="Year",
                        y="Countries",
                        title="Number of Reporting Countries"
                    )
                    safe_plotly_chart(fig1)

                with c2:
                    if "Average_Metric" in coverage.columns:
                        fig2 = px.line(
                            coverage,
                            x="Year",
                            y="Average_Metric",
                            markers=True,
                            title=f"Average {metric} Over Time"
                        )
                        safe_plotly_chart(fig2)

            st.markdown("---")
            st.markdown("<div class='section-header'>Country-Level Table</div>", unsafe_allow_html=True)

            display_cols = [c for c in ["CountryTerritoryArea", "Year", metric] if c in df_map.columns]

            st.dataframe(
                df_map[display_cols].sort_values(metric, ascending=False),
                use_container_width=True,
                hide_index=True
            )


# ─────────────────────────────────────────────
# PAGE 3 — RESISTANCE TRENDS
# ─────────────────────────────────────────────
elif page == "📈 Resistance Trends":

    st.markdown("# 📈 Resistance Trends Over Time")
    st.markdown("Country-level and global antimicrobial resistance patterns")
    st.markdown("---")

    if ts_all.empty:
        st.error("Time-series data was not loaded.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            pathogens = sorted(ts_all["PathogenName"].dropna().unique())
            selected_pathogen = st.selectbox("Pathogen", pathogens)

        with col2:
            antibiotics = sorted(
                ts_all[ts_all["PathogenName"] == selected_pathogen]["AbTargets"].dropna().unique()
            )
            selected_antibiotic = st.selectbox("Antibiotic", antibiotics)

        df_filtered = ts_all[
            (ts_all["PathogenName"] == selected_pathogen)
            & (ts_all["AbTargets"] == selected_antibiotic)
        ].copy()

        st.markdown("---")

        st.markdown("<div class='section-header'>Global Median Trend</div>", unsafe_allow_html=True)

        file_lookup = {
            ("Escherichia coli", "Ciprofloxacin"): "ts_Ecoli_Cipro.csv",
            ("Escherichia coli", "Meropenem"): "ts_Ecoli_Meropenem.csv",
            ("Klebsiella pneumoniae", "Meropenem"): "ts_Kpneumoniae_Meropenem.csv",
            ("Klebsiella pneumoniae", "Ceftriaxone"): "ts_Kpneumoniae_Ceftriaxone.csv",
            ("Salmonella spp.", "Ciprofloxacin"): "ts_Salmonella_Cipro.csv",
            ("Acinetobacter spp.", "Meropenem"): "ts_Acinetobacter_Meropenem.csv",
        }

        summary_file = file_lookup.get((selected_pathogen, selected_antibiotic))
        df_summary = ts_summary_dfs.get(summary_file, pd.DataFrame())

        if not df_summary.empty and {"Year", "Median"}.issubset(df_summary.columns):
            fig = go.Figure()

            if {"Q1", "Q3"}.issubset(df_summary.columns):
                fig.add_trace(go.Scatter(
                    x=df_summary["Year"],
                    y=df_summary["Q3"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=df_summary["Year"],
                    y=df_summary["Q1"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(192,57,43,0.18)",
                    line=dict(width=0),
                    name="IQR"
                ))

            fig.add_trace(go.Scatter(
                x=df_summary["Year"],
                y=df_summary["Median"],
                mode="lines+markers",
                name="Median Resistance",
                line=dict(width=3)
            ))

            fig.update_layout(
                title=f"{selected_pathogen} resistance to {selected_antibiotic}",
                xaxis_title="Year",
                yaxis_title="% Resistant",
                height=420,
                yaxis_range=[0, 100]
            )

            safe_plotly_chart(fig)
        else:
            st.info("Global summary trend not available for this combination.")

        st.markdown("---")
        st.markdown("<div class='section-header'>Country-Level Trends</div>", unsafe_allow_html=True)

        regions = ["All"] + sorted(df_filtered["WHORegionName"].dropna().unique()) if "WHORegionName" in df_filtered.columns else ["All"]
        selected_region = st.selectbox("Filter by WHO Region", regions)

        df_plot = df_filtered.copy()

        if selected_region != "All":
            df_plot = df_plot[df_plot["WHORegionName"] == selected_region]

        if not df_plot.empty:
            fig2 = px.line(
                df_plot,
                x="Year",
                y="PercentResistant",
                color="CountryTerritoryArea",
                markers=True,
                title=f"Country Trends: {selected_pathogen} · {selected_antibiotic}",
                labels={"PercentResistant": "% Resistant", "CountryTerritoryArea": "Country"}
            )

            fig2.update_layout(height=520, yaxis_range=[0, 100])
            safe_plotly_chart(fig2)
        else:
            st.info("No country-level data available for this selection.")


# ─────────────────────────────────────────────
# PAGE 4 — PATHOGEN ANALYSIS
# ─────────────────────────────────────────────
elif page == "🦠 Pathogen Analysis":

    st.markdown("# 🦠 Pathogen Analysis")
    st.markdown("Compare resistance profiles across bacterial pathogens")
    st.markdown("---")

    if res_all.empty:
        st.error("Resistance data was not loaded.")
    else:
        tab1, tab2, tab3 = st.tabs(["Resistance Profile", "Pathogen Comparison", "Isolate Volume"])

        with tab1:
            selected_pathogen = st.selectbox(
                "Select Pathogen",
                sorted(res_all["PathogenName"].dropna().unique())
            )

            df_p = res_all[
                res_all["PathogenName"] == selected_pathogen
            ].dropna(subset=["Median"]).copy()

            agg_dict = {"Median": ("Median", "mean")}

            if "TotalBCIsWithAST" in df_p.columns:
                agg_dict["TotalBCIsWithAST"] = ("TotalBCIsWithAST", "sum")

            df_p = df_p.groupby("AntibioticName", as_index=False).agg(**agg_dict)
            df_p["Median"] = df_p["Median"].clip(lower=0, upper=100)
            df_p = df_p.sort_values("Median", ascending=True)

            fig = px.bar(
                df_p,
                x="Median",
                y="AntibioticName",
                orientation="h",
                color="Median",
                color_continuous_scale="RdYlGn_r",
                title=f"Resistance Profile · {selected_pathogen}",
                labels={"Median": "Median Resistance %", "AntibioticName": "Antibiotic"}
            )

            fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="50%")
            fig.add_vline(x=25, line_dash="dot", line_color="orange", annotation_text="25%")
            fig.update_layout(height=520, xaxis_range=[0, 100])

            safe_plotly_chart(fig)

            c1, c2, c3 = st.columns(3)

            high = len(df_p[df_p["Median"] > 50])
            moderate = len(df_p[(df_p["Median"] >= 25) & (df_p["Median"] <= 50)])
            low = len(df_p[df_p["Median"] < 25])

            with c1:
                st.error(f"High resistance >50%: **{high} antibiotics**")
            with c2:
                st.warning(f"Moderate resistance 25–50%: **{moderate} antibiotics**")
            with c3:
                st.success(f"Low resistance <25%: **{low} antibiotics**")

        with tab2:
            common_ab = res_all.groupby("AntibioticName")["PathogenName"].nunique()
            common_ab = common_ab[common_ab >= 2].index.tolist()

            if common_ab:
                selected_ab = st.selectbox("Select Antibiotic", sorted(common_ab))

                df_comp = res_all[
                    res_all["AntibioticName"] == selected_ab
                ].dropna(subset=["Median"]).copy()

                df_comp = df_comp.groupby("PathogenName", as_index=False).agg(
                    Median=("Median", "mean")
                )

                df_comp["Median"] = df_comp["Median"].clip(lower=0, upper=100)

                fig = px.bar(
                    df_comp.sort_values("Median", ascending=False),
                    x="PathogenName",
                    y="Median",
                    color="Median",
                    color_continuous_scale="RdYlGn_r",
                    title=f"Resistance to {selected_ab} by Pathogen",
                    labels={"Median": "Median Resistance %", "PathogenName": "Pathogen"}
                )

                fig.update_layout(height=430, yaxis_range=[0, 100])
                safe_plotly_chart(fig)
            else:
                st.info("No antibiotic appears across multiple pathogens.")

        with tab3:
            if "TotalBCIsWithAST" in res_all.columns:
                df_vol = res_all.dropna(subset=["TotalBCIsWithAST"]).copy()

                df_vol = df_vol.groupby("PathogenName")["TotalBCIsWithAST"].sum().reset_index()
                df_vol = df_vol.sort_values("TotalBCIsWithAST", ascending=False)

                fig = px.pie(
                    df_vol,
                    values="TotalBCIsWithAST",
                    names="PathogenName",
                    title="Share of Tested Isolates by Pathogen"
                )

                safe_plotly_chart(fig)
                st.dataframe(df_vol, use_container_width=True, hide_index=True)
            else:
                st.info("Isolate volume column not found.")


# ─────────────────────────────────────────────
# PAGE 5 — ANTIBIOTIC RESISTANCE RANKING
# ─────────────────────────────────────────────
elif page == "💊 Antibiotic Resistance Ranking":

    st.markdown("# 💊 Antibiotic Resistance Ranking")
    st.markdown(
        "Lower observed resistance indicates lower reported resistance in the surveillance data; "
        "this is not a treatment recommendation."
    )
    st.markdown("---")

    if res_all.empty:
        st.error("Resistance data was not loaded.")
    else:
        pathogens = ["All"] + sorted(res_all["PathogenName"].dropna().unique())
        selected_pathogen = st.selectbox("Filter by Pathogen", pathogens)

        df_ab = res_all.copy()

        if selected_pathogen != "All":
            df_ab = df_ab[df_ab["PathogenName"] == selected_pathogen]

        df_ab = df_ab.dropna(subset=["Median"]).copy()
        df_ab["Median"] = df_ab["Median"].clip(lower=0, upper=100)

        if "TotalBCIsWithAST" in df_ab.columns:
            df_ab["TotalBCIsWithAST"] = clean_numeric(df_ab["TotalBCIsWithAST"])
        else:
            df_ab["TotalBCIsWithAST"] = np.nan

        df_rank = df_ab.groupby("AntibioticName").agg(
            Average_Resistance=("Median", "mean"),
            Total_Isolates=("TotalBCIsWithAST", "sum"),
            Pathogens_Tested=("PathogenName", "nunique")
        ).reset_index()

        df_rank["Average_Resistance"] = df_rank["Average_Resistance"].clip(lower=0, upper=100)

        df_rank["Status"] = df_rank["Average_Resistance"].apply(
            lambda x: "Low reported resistance" if x < 10 else
            "Moderate reported resistance" if x < 30 else
            "High reported resistance" if x < 50 else
            "Very high reported resistance"
        )

        df_rank = df_rank.sort_values("Average_Resistance", ascending=True)

        c1, c2 = st.columns([3, 2])

        with c1:
            fig = px.bar(
                df_rank,
                x="Average_Resistance",
                y="AntibioticName",
                orientation="h",
                color="Average_Resistance",
                color_continuous_scale="RdYlGn_r",
                title="Antibiotic Resistance Ranking",
                labels={
                    "Average_Resistance": "Average Median Resistance %",
                    "AntibioticName": "Antibiotic"
                }
            )

            fig.add_vline(x=10, line_dash="dot", line_color="green")
            fig.add_vline(x=30, line_dash="dot", line_color="orange")
            fig.add_vline(x=50, line_dash="dash", line_color="red")
            fig.update_layout(height=560, yaxis=dict(autorange="reversed"), xaxis_range=[0, 100])

            safe_plotly_chart(fig)

        with c2:
            st.markdown("<div class='section-header'>Status Summary</div>", unsafe_allow_html=True)

            status_counts = df_rank["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]

            fig2 = px.pie(
                status_counts,
                values="Count",
                names="Status",
                title="Antibiotic Resistance Category Distribution"
            )

            safe_plotly_chart(fig2)

            st.markdown("<div class='section-header'>Ranked Table</div>", unsafe_allow_html=True)

            table = df_rank[["AntibioticName", "Average_Resistance", "Status"]].copy()
            table.columns = ["Antibiotic", "Average Resistance %", "Status"]
            table["Average Resistance %"] = table["Average Resistance %"].round(1)

            st.dataframe(table, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 6 — COUNTRY EXPLORER
# ─────────────────────────────────────────────
elif page == "🔬 Country Explorer":

    st.markdown("# 🔬 Country Explorer")
    st.markdown("Drill into antimicrobial resistance patterns by country")
    st.markdown("---")

    if ts_all.empty:
        st.error("Time-series data was not loaded.")
    else:
        countries = sorted(ts_all["CountryTerritoryArea"].dropna().unique())
        default_index = countries.index("Lebanon") if "Lebanon" in countries else 0

        selected_country = st.selectbox("Select Country", countries, index=default_index)

        df_c = ts_all[ts_all["CountryTerritoryArea"] == selected_country].copy()

        if df_c.empty:
            st.info("No data for this country.")
        else:
            region = df_c["WHORegionName"].iloc[0] if "WHORegionName" in df_c.columns else "Unknown"

            st.markdown(f"**WHO Region:** {region}")

            latest_year = df_c["Year"].max()
            latest = df_c[df_c["Year"] == latest_year].copy()

            avg_res = latest["PercentResistant"].mean()
            combos = df_c[["PathogenName", "AbTargets"]].drop_duplicates().shape[0]
            years_available = df_c["Year"].nunique()

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{avg_res:.1f}%</div>
                    <div class='metric-label'>Average Resistance in {latest_year}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{combos}</div>
                    <div class='metric-label'>Pathogen-Antibiotic Combos</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{years_available}</div>
                    <div class='metric-label'>Years Available</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            df_c["Combo"] = df_c["PathogenName"] + " · " + df_c["AbTargets"]

            c1, c2 = st.columns(2)

            with c1:
                fig = px.line(
                    df_c,
                    x="Year",
                    y="PercentResistant",
                    color="Combo",
                    markers=True,
                    title=f"Resistance Trends · {selected_country}",
                    labels={"PercentResistant": "% Resistant"}
                )

                fig.update_layout(yaxis_range=[0, 100])
                safe_plotly_chart(fig)

            with c2:
                latest["Combo"] = latest["PathogenName"] + " · " + latest["AbTargets"]

                latest_snapshot = latest.groupby("Combo", as_index=False).agg(
                    PercentResistant=("PercentResistant", "mean")
                )

                latest_snapshot["PercentResistant"] = latest_snapshot["PercentResistant"].clip(
                    lower=0,
                    upper=100
                )

                fig2 = px.bar(
                    latest_snapshot.sort_values("PercentResistant", ascending=False),
                    x="Combo",
                    y="PercentResistant",
                    color="PercentResistant",
                    color_continuous_scale="RdYlGn_r",
                    title=f"Latest Resistance Snapshot · {selected_country}"
                )

                fig2.add_hline(y=50, line_dash="dash", line_color="red")
                fig2.update_layout(xaxis_tickangle=-30, yaxis_range=[0, 100])
                safe_plotly_chart(fig2)

            st.markdown("---")
            st.markdown("<div class='section-header'>Raw Country Data</div>", unsafe_allow_html=True)

            st.dataframe(df_c, use_container_width=True, hide_index=True)
