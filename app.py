import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Agriculture Forecast data", layout="wide")

# Load data
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Must-have columns for all pages
    required = {"country", "commodity", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce year
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country", "commodity", "year"]).copy()
    df["year"] = df["year"].astype(int)

    # Convert known numeric columns if present
    for col in ["production", "exports", "imports", "consumption", "yield", "area"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# Page 1: Production Trends
def show_production_page(df: pd.DataFrame):
    st.title("Forecast Production Trends")
    st.caption("Prediction based data")

    if "production" not in df.columns:
        st.error("Column 'production' is missing in processed_data.csv, so this page can't be displayed.")
        return

    df_prod = df.dropna(subset=["production"]).copy()
    if df_prod.empty:
        st.warning("No valid production rows found.")
        return

    st.sidebar.header("Filters")

    commodities = sorted(df_prod["commodity"].unique().tolist())
    selected_commodity = st.sidebar.selectbox("Select commodity", commodities, key="prod_commodity")

    df_c = df_prod[df_prod["commodity"] == selected_commodity].copy()

    countries = sorted(df_c["country"].unique().tolist())
    default_countries = countries[:6]
    selected_countries = st.sidebar.multiselect(
        "Countries",
        options=countries,
        default=default_countries if len(countries) >= len(default_countries) else countries,
        key="prod_countries",
    )

    min_year = int(df_c["year"].min())
    max_year = int(df_c["year"].max())
    year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year), key="prod_years")

    show_avg = st.sidebar.checkbox("Show average of all countries", value=True, key="prod_avg")

    df_plot = df_c[
        (df_c["country"].isin(selected_countries)) &
        (df_c["year"].between(year_range[0], year_range[1]))
    ].copy()

    if df_plot.empty:
        st.warning("No data matches the filters. Try selecting more countries or widening the year range.")
        return

    fig = px.line(
        df_plot,
        x="year",
        y="production",
        color="country",
        markers=True,
        labels={"year": "Year", "production": "Predicted production", "country": "Country"},
        title=f"Predicted production over time – {selected_commodity}",
    )

    if show_avg:
        avg = (
            df_c[df_c["year"].between(year_range[0], year_range[1])]
            .groupby("year", as_index=False)["production"].mean()
        )
        fig_avg = px.line(avg, x="year", y="production", markers=True)
        for tr in fig_avg.data:
            tr.name = "Average (all countries)"
            tr.showlegend = True
            fig.add_trace(tr)

    fig.update_layout(height=600, legend_title_text="Country")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show sample rows"):
        st.dataframe(df_plot.head(50), use_container_width=True)


# Page 2: Trade / Throughput Comparison
def show_trade_throughput_page(df: pd.DataFrame):
    st.title("Compare Forecasts of Import, Export, Production, Consumption")
    st.caption("Comparison between countries and commodities")

    st.sidebar.markdown("---")
    st.sidebar.header("Comparison Filters")

    # Metrics that exist in this dataset
    possible_metrics = ["exports", "imports", "consumption", "production"]
    metrics = [m for m in possible_metrics if m in df.columns]

    if not metrics:
        st.error(
            "None of the required metric columns exist in processed_data.csv.\n"
            "Need at least one of: exports/imports/consumption/production."
        )
        return

    df_metrics = df.dropna(subset=metrics, how="all").copy()
    if df_metrics.empty:
        st.warning("No valid metric rows found for comparison.")
        return

    unique_commodities = sorted(df_metrics["commodity"].unique().tolist())
    comp_commodities = ["Average of All"] + unique_commodities

    selected_comp_commodity = st.sidebar.selectbox(
        "Select Commodity",
        comp_commodities,
        index=0,
        key="comp_commodity_select",
    )

    min_comp_year = int(df_metrics["year"].min())
    max_comp_year = int(df_metrics["year"].max())
    selected_comp_year = st.sidebar.slider(
        "Select Year",
        min_comp_year,
        max_comp_year,
        max_comp_year,
        key="comp_year_slider",
    )

    comp_countries_list = sorted(df_metrics["country"].unique().tolist())
    default_comp_countries = comp_countries_list[:3] if len(comp_countries_list) >= 3 else comp_countries_list
    selected_comp_countries = st.sidebar.multiselect(
        "Select Countries",
        comp_countries_list,
        default=default_comp_countries,
        key="comp_country_select",
    )

    df_filtered = df_metrics[
        (df_metrics["year"] == selected_comp_year) &
        (df_metrics["country"].isin(selected_comp_countries))
    ].copy()

    if selected_comp_commodity == "Average of All":
        chart_title = f"Average Distribution in {selected_comp_year}"
        if df_filtered.empty:
            df_comp = df_filtered
        else:
            df_comp = df_filtered.groupby("country", as_index=False)[metrics].mean()
    else:
        df_comp = df_filtered[df_filtered["commodity"] == selected_comp_commodity]
        chart_title = f"Distribution of {selected_comp_commodity} in {selected_comp_year}"

    if df_comp.empty:
        st.warning("No data available for the selected filters.")
        return

    df_melted = df_comp.melt(
        id_vars=["country"],
        value_vars=metrics,
        var_name="Category",
        value_name="Value",
    )

    fig_comp = px.bar(
        df_melted,
        x="country",
        y="Value",
        color="Category",
        barmode="group",
        title=chart_title,
        labels={"Value": "Quantity", "country": "Country"},
        height=500,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    with st.expander("View Data"):
        st.dataframe(df_comp, use_container_width=True)


# Page 3: Yield Choropleth Map
def show_yield_map_page(df: pd.DataFrame):
    st.title("🌍 Global Yield – Interactive Choropleth Map")
    st.caption("Uses processed_data.csv only")

    df_map = df.copy()

    if "yield" in df_map.columns:
        df_map = df_map.dropna(subset=["yield"]).copy()
        value_col = "yield"
    else:
        # Optional computed yield if possible
        if "production" in df_map.columns and "area" in df_map.columns:
            df_map = df_map.dropna(subset=["production", "area"]).copy()
            df_map = df_map[df_map["area"] != 0].copy()
            df_map["yield"] = df_map["production"] / df_map["area"]
            value_col = "yield"
        else:
            st.error(
                "To show the Yield Map using processed_data.csv, you need either:\n"
                "- a 'yield' column, OR\n"
                "- both 'production' and 'area' columns (to compute yield = production/area)."
            )
            return

    # Keep only rows that can draw a map
    df_map = df_map.dropna(subset=["country", "commodity", "year", value_col]).copy()
    if df_map.empty:
        st.warning("No valid rows found for the Yield Map.")
        return

    st.sidebar.markdown("---")
    st.sidebar.header("Yield Map Filters")

    valid_commodities = sorted(df_map["commodity"].unique().tolist())
    selected_commodity = st.sidebar.selectbox("Select Commodity", valid_commodities, key="yield_commodity")

    df_commodity = df_map[df_map["commodity"] == selected_commodity].copy()

    valid_years = sorted(df_commodity["year"].unique().tolist())
    if not valid_years:
        st.error(f"No valid years with yield data for commodity: {selected_commodity}")
        return

    selected_year = st.sidebar.slider(
        "Select Year",
        min_value=int(min(valid_years)),
        max_value=int(max(valid_years)),
        value=int(min(valid_years)),
        step=1,
        key="yield_year",
    )

    df_year = df_commodity[df_commodity["year"] == selected_year].copy()
    if df_year.empty:
        st.warning("No rows for that year/commodity combination.")
        return

    fig = px.choropleth(
        df_year,
        locations="country",
        locationmode="country names",
        color=value_col,
        color_continuous_scale="Oranges",
        hover_name="country",
        hover_data={
            "commodity": True,
            "year": True,
            value_col: ":.2f",
        },
        labels={value_col: "Yield"},
        title=f"Yield by Country – {selected_commodity} ({selected_year})",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Yield"),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show sample rows"):
        st.dataframe(df_year.head(50), use_container_width=True)



if __name__ == "__main__":
    DATA_PATH = "processed_data.csv"

    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load data from '{DATA_PATH}'.\n\n{e}")
        st.stop()

    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio(
        "Go to",
        [
            "Production Trends",
            "Trade and Throughput market Comparison",
            "Yield Map",
        ],
    )

    if page_selection == "Production Trends":
        show_production_page(df)
    elif page_selection == "Trade and Throughput market Comparison":
        show_trade_throughput_page(df)
    elif page_selection == "Yield Map":
        show_yield_map_page(df)

