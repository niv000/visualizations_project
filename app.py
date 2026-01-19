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
    st.caption(
        "Presents forecasted production over time for selected commodities and countries.\n"
        "It enables comparison of temporal trends between countries and highlights overall patterns "
        "using an optional global average."
    )

    if "production" not in df.columns:
        st.error("Column 'production' is missing in processed_data.csv, so this page can't be displayed.")
        return

    df_prod = df.dropna(subset=["production", "country", "commodity", "year"]).copy()
    if df_prod.empty:
        st.warning("No valid production rows found.")
        return

    st.sidebar.header("Filters")

    commodities = sorted(df_prod["commodity"].unique().tolist())

    # Default to all commodities (Average of all)
    default_comm = commodities if commodities else []

    selected_commodities = st.sidebar.multiselect(
        "Select commodities",
        commodities,
        default=default_comm,
        key="prod_commodities"
    )

    if not selected_commodities:
        st.warning("Please select at least one commodity.")
        return

    # Filter for all selected commodities
    df_c = df_prod[df_prod["commodity"].isin(selected_commodities)].copy()

    if df_c.empty:
        st.warning("No rows for selected commodities.")
        return

    # Countries filter (DEFAULT: Israel only, but user can add/remove freely)
    countries = sorted(df_c["country"].unique().tolist())
    default_countries = ["Israel"] if "Israel" in countries else (countries[:1] if countries else [])

    selected_countries = st.sidebar.multiselect(
        "Countries",
        options=countries,
        default=default_countries,
        key="prod_countries",
    )

    # Year range filter
    min_year = int(df_c["year"].min())
    max_year = int(df_c["year"].max())
    year_range = st.sidebar.slider(
        "Year range",
        min_year,
        max_year,
        (min_year, max_year),
        key="prod_years",
    )

    # Average checkbox (DEFAULT: off)
    show_avg = st.sidebar.checkbox("Show average of all countries", value=False, key="prod_avg")

    # Filter Data for selected countries + years
    df_subset = df_c[
        (df_c["country"].isin(selected_countries)) &
        (df_c["year"].between(year_range[0], year_range[1]))
        ].copy()

    if df_subset.empty:
        st.warning("No data matches the filters. Try selecting more countries or widening the year range.")
        return

    # Group by Country and Year, mean of production
    df_plot = df_subset.groupby(["country", "year"], as_index=False)["production"].mean()

    # Determine dynamic labels
    if len(selected_commodities) > 1:
        chart_title = "Average Predicted Production – Selected Commodities"
        y_label = "Average Production"
    else:
        chart_title = f"Predicted Production – {selected_commodities[0]}"
        y_label = "Predicted Production"

    # Main chart
    fig = px.line(
        df_plot,
        x="year",
        y="production",
        color="country",
        markers=True,
        labels={"year": "Year", "production": y_label, "country": "Country"},
        title=chart_title,
    )

    fig.update_traces(marker=dict(size=8), selector=dict(mode="lines+markers"))

    # Average line
    if show_avg:
        # Take all data for the selected years (across all countries)
        df_year_all = df_c[df_c["year"].between(year_range[0], year_range[1])]

        # Average commodities per country first
        country_avgs = df_year_all.groupby(["country", "year"], as_index=False)["production"].mean()

        # Global average across countries
        avg = country_avgs.groupby("year", as_index=False)["production"].mean()

        fig_avg = px.line(avg, x="year", y="production", markers=True)
        for tr in fig_avg.data:
            tr.name = "Average (all countries)"
            tr.showlegend = True
            tr.line.color = "black"
            tr.line.width = 6
            tr.marker.size = 10
            tr.marker.color = "black"
            fig.add_trace(tr)

    # X axis settings
    tick_vals = list(range(year_range[0], year_range[1] + 1, 1))
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=[str(y) for y in tick_vals],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )

    # Y axis settings
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        showticklabels=True,
        tickformat=",",
    )

    fig.update_layout(height=600, legend_title_text="Country")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show sample rows (Aggregated)"):
        st.dataframe(df_plot.head(50), use_container_width=True)


# Page 2: Trade / Throughput Comparison
def show_trade_throughput_page(df: pd.DataFrame):
    st.title("Compare Forecasts of Import, Export, Production, Consumption")
    st.caption(
        "Compares key indicators &  productio. it's imports, exports, and consumption across countries for a selected year.\n"
        "It supports side by side comparison of market structure and relative contribution by commodity."
    )

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

    # Default countries: Israel, United States, Russia (if present in the data)
    preferred_defaults = ["Israel", "United States", "Russia"]
    default_comp_countries = [c for c in preferred_defaults if c in comp_countries_list]

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


#Page 3: Yield Choropleth Map
def show_yield_map_page(df: pd.DataFrame):
    st.title("Global Yield: Relative Performance")
    st.caption("Displays agricultural yield by country for a selected commodity and year.\n It provides a global spatial overview, allowing identification of geographic patterns and regional differences")

    # Session state
    if "yield_map_countries" not in st.session_state:
        st.session_state.yield_map_countries = []

    df_map = df.copy()

    # Determine Yield column
    if "yield" in df_map.columns:
        if df_map["yield"].notna().sum() == 0 and "production" in df_map.columns and "area" in df_map.columns:
            df_map["yield"] = df_map["production"] / df_map["area"]
    elif "production" in df_map.columns and "area" in df_map.columns:
        df_map["yield"] = df_map["production"] / df_map["area"]
    else:
        st.error("Missing 'yield' column (or 'production' + 'area').")
        return

    if "production" not in df_map.columns:
        st.error("Missing 'production' column.")
        return

    df_map = df_map.dropna(subset=["yield", "country", "commodity", "year"]).copy()

    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.header("Yield Map Filters")

    selected_commodity = st.sidebar.selectbox(
        "Select Commodity",
        sorted(df_map["commodity"].unique()),
        key="yield_commodity",
    )

    df_commodity = df_map[df_map["commodity"] == selected_commodity]

    selected_year = st.sidebar.slider(
        "Select Year",
        int(df_commodity["year"].min()),
        int(df_commodity["year"].max()),
        int(df_commodity["year"].min()),
        key="yield_year",
    )

    df_year = df_commodity[df_commodity["year"] == selected_year]
    if df_year.empty:
        st.warning("No data for this selection.")
        return

    # Relative yield
    global_avg = df_year["yield"].mean()
    df_year["relative_yield"] = df_year["yield"] / global_avg

    p05 = df_year["relative_yield"].quantile(0.05)
    p95 = df_year["relative_yield"].quantile(0.95)
    max_dev = max(abs(p95 - 1), abs(1 - p05), 0.2)

    fig = px.choropleth(
        df_year,
        locations="country",
        locationmode="country names",
        color="relative_yield",
        range_color=[1 - max_dev, 1 + max_dev],
        color_continuous_scale=["#8B4513", "#FFFFE0", "#000080"],
        hover_name="country",
        hover_data={"yield": ":.2f", "production": ":.2f"},
        title=(
            f"Yield vs. Global Average – {selected_commodity} ({selected_year})"
            f"<br><sub>Global Avg: {global_avg:.2f} t/m^2 | 1.0 = Average</sub>"
        ),
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=80, b=0),
        coloraxis_colorbar=dict(title="Relative<br>Yield"),
    )

    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    # Extract selected countries (max 2)
    points = []
    if isinstance(selection, dict):
        points = selection.get("points", []) or selection.get("selection", {}).get("points", [])
    else:
        sel = getattr(selection, "selection", None)
        if isinstance(sel, dict):
            points = sel.get("points", [])

    clicked = []
    for p in points:
        c = p.get("location") or p.get("hovertext")
        if c and c not in clicked:
            clicked.append(c)

    if clicked:
        merged = []
        for c in st.session_state.yield_map_countries + clicked:
            if c not in merged:
                merged.append(c)
        st.session_state.yield_map_countries = merged[-2:]

    # Controls
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("Clear selection"):
            st.session_state.yield_map_countries = []
    with c2:
        st.write(
            "**Selected (up to 2):** "
            + (", ".join(st.session_state.yield_map_countries) if st.session_state.yield_map_countries else "None")
        )

    # Bar charts
    if st.session_state.yield_map_countries:
        df_sel = df_year[df_year["country"].isin(st.session_state.yield_map_countries)]

        df_agg = df_sel.groupby("country", as_index=False).agg(
            production=("production", "mean"),
            yield_val=("yield", "mean"),
        )

        order = {c: i for i, c in enumerate(st.session_state.yield_map_countries)}
        df_agg["__o"] = df_agg["country"].map(order)
        df_agg = df_agg.sort_values("__o")

        st.subheader(f"Country Comparison — {selected_commodity} ({selected_year})")

        col1, col2 = st.columns(2)

        # Production chart
        with col1:
            max_prod = df_agg["production"].max()

            fig_p = px.bar(
                df_agg,
                x="country",
                y="production",
                text="production",
                title="Production",
            )
            fig_p.update_traces(
                marker_color="#2E8B57",
                texttemplate="%{text:.2f}",
                textposition="outside",
                textfont=dict(size=16),
            )
            fig_p.update_layout(
                height=420,
                yaxis=dict(
                    range=[0, max_prod * 1.15],
                    title="Production",
                    title_font=dict(size=18, color="black"),
                    tickfont=dict(size=14, color="black"),
                ),
                xaxis=dict(
                    title="Country",
                    title_font=dict(size=18, color="black"),
                    tickfont=dict(size=20, color="black"),
                ),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        # Yield chart
        with col2:
            max_yield = df_agg["yield_val"].max()

            fig_y = px.bar(
                df_agg,
                x="country",
                y="yield_val",
                text="yield_val",
                title="Yield (t/m^2)",
            )
            fig_y.update_traces(
                marker_color="#D2691E",
                texttemplate="%{text:.2f}",
                textposition="outside",
                textfont=dict(size=16),
            )
            fig_y.update_layout(
                height=420,
                yaxis=dict(
                    range=[0, max_yield * 1.20],
                    title="Yield (t/m^2)",
                    title_font=dict(size=18, color="black"),
                    tickfont=dict(size=14, color="black"),
                ),
                xaxis=dict(
                    title="Country",
                    title_font=dict(size=18, color="black"),
                    tickfont=dict(size=20, color="black"),
                ),
            )
            st.plotly_chart(fig_y, use_container_width=True)

        st.caption("**t/m^2** = tons per hectare")

    else:
        st.info("Select up to 2 countries on the map to compare their Production and Yield.")

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





