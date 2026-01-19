import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        "Compares key indicators (Imports, Exports, Production, Consumption) across countries for a selected year.\n"
        "It supports side by side comparison of market structure and relative contribution by commodity.\n"
        "Below, select a country to focus on, and see how it performs relative to the global average."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Comparison Filters")

    possible_metrics = ["production", "consumption", "imports", "exports"]
    metrics = [m for m in possible_metrics if m in df.columns]

    if not metrics:
        st.error("None of the required metric columns exist in processed_data.csv.")
        return

    df_metrics = df.dropna(subset=metrics, how="all").copy()
    if df_metrics.empty:
        st.warning("No valid metric rows found.")
        return

    # Filters
    unique_commodities = sorted(df_metrics["commodity"].unique().tolist())
    comp_commodities = ["Average of All"] + unique_commodities

    selected_comp_commodity = st.sidebar.selectbox(
        "Select Commodity (Main Chart)",
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
    preferred_defaults = ["Israel", "United States", "Russia"]
    default_comp_countries = [c for c in preferred_defaults if c in comp_countries_list]

    selected_comp_countries = st.sidebar.multiselect(
        "Select Countries (Main Chart)",
        comp_countries_list,
        default=default_comp_countries,
        key="comp_country_select",
    )

    # Main chart
    st.subheader(f"1. Global Comparison: {selected_comp_commodity}")

    df_filtered = df_metrics[
        (df_metrics["year"] == selected_comp_year) &
        (df_metrics["country"].isin(selected_comp_countries))
        ].copy()

    if selected_comp_commodity == "Average of All":
        if not df_filtered.empty:
            df_comp = df_filtered.groupby("country", as_index=False)[metrics].mean()
        else:
            df_comp = df_filtered
        chart_title = f"Average Metrics by Country ({selected_comp_year})"
    else:
        df_comp = df_filtered[df_filtered["commodity"] == selected_comp_commodity]
        chart_title = f"{selected_comp_commodity} Metrics by Country ({selected_comp_year})"

    if not df_comp.empty:
        df_melted = df_comp.melt(
            id_vars=["country"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Value",
        )

        fig_comp = px.bar(
            df_melted,
            x="country",
            y="Value",
            color="Metric",
            barmode="group",
            title=chart_title,
            labels={"Value": "Quantity (1000 Tonnes)", "country": "Country"},
            height=450,
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No data for the selected countries/year in the main chart.")

    st.markdown("---")

    # Relative performance
    st.subheader("2. Country Deep-Dive: Deviation from Global Average")
    st.caption(
        "**Left Panel:** Absolute Quantity (Bar). **Right Panel:** Deviation from Global Average (Dot).\n"
        "In the right panel, the vertical dashed line represents the Global Average (0%)."
    )

    selected_drill_country = st.selectbox(
        "Select Country to Analyze",
        comp_countries_list,
        index=comp_countries_list.index("Israel") if "Israel" in comp_countries_list else 0
    )

    if selected_drill_country:
        df_drill_year = df_metrics[df_metrics["year"] == selected_comp_year].copy()

        # Selected country data
        df_target = df_drill_year[df_drill_year["country"] == selected_drill_country].copy()

        # Global averages for that year
        df_avgs = df_drill_year.groupby("commodity", as_index=False)[metrics].mean()

        if not df_target.empty and not df_avgs.empty:
            # Melt both for merging
            df_target_melt = df_target.melt(id_vars=["commodity"], value_vars=metrics, var_name="Metric",
                                            value_name="Value")
            df_avgs_melt = df_avgs.melt(id_vars=["commodity"], value_vars=metrics, var_name="Metric",
                                        value_name="Avg_Value")

            # Merge
            df_merged = pd.merge(df_target_melt, df_avgs_melt, on=["commodity", "Metric"], how="inner")

            # Calculate % Diff
            df_merged["Pct_Diff"] = df_merged.apply(
                lambda x: ((x["Value"] - x["Avg_Value"]) / x["Avg_Value"]) if x["Avg_Value"] > 0 else 0, axis=1
            )
            df_merged["Diff_Label"] = df_merged["Pct_Diff"].apply(lambda x: f"{x:+.0%}")

            # Sort commodities by total volume
            df_target["_total_vol"] = df_target[metrics].sum(axis=1)
            sorted_commodities = df_target.sort_values("_total_vol", ascending=True)["commodity"].tolist()

            fig_drill = make_subplots(
                rows=1, cols=2,
                shared_yaxes=True,
                column_widths=[0.3, 0.7],
                horizontal_spacing=0.03,
                subplot_titles=("Absolute Quantity (Tonnes)", "Deviation from Global Avg")
            )

            colors = {
                'production': '#1f77b4', 'consumption': '#89cff0',
                'imports': '#d62728', 'exports': '#ff9896'
            }

            for metric in metrics:
                subset = df_merged[df_merged["Metric"] == metric]

                # Left: Absolute bar
                fig_drill.add_trace(go.Bar(
                    y=subset["commodity"],
                    x=subset["Value"],
                    orientation='h',
                    name=metric,
                    marker_color=colors.get(metric, 'gray'),
                    legendgroup=metric,
                ), row=1, col=1)

                # Right: Normalized dot
                fig_drill.add_trace(go.Scatter(
                    y=subset["commodity"],
                    x=subset["Pct_Diff"],
                    mode='markers',
                    text=subset["Diff_Label"],
                    name=metric,
                    marker=dict(size=9, color=colors.get(metric, 'gray'), line=dict(width=1, color='DarkSlateGrey')),
                    legendgroup=metric,
                    showlegend=False,
                    hovertemplate=(
                            "<b>%{y}</b><br>" +
                            "Metric: " + metric + "<br>" +
                            "Value: %{customdata[0]:.0f}<br>" +
                            "Global Avg: %{customdata[1]:.0f}<br>" +
                            "Deviation: %{text}<extra></extra>"
                    ),
                    customdata=subset[['Value', 'Avg_Value']]
                ), row=1, col=2)

            # Add Vertical Line at 0 (Average)
            fig_drill.add_vline(
                x=0, line_width=2, line_dash="dash", line_color="#555555",
                annotation_text="Global Avg", annotation_position="top right",
                row=1, col=2
            )

            fig_drill.update_layout(
                height=max(600, len(sorted_commodities) * 25),
                title_text=f"Profile: {selected_drill_country} vs. Global Average ({selected_comp_year})",
                barmode='group',
                plot_bgcolor='white',
                yaxis=dict(categoryorder='array', categoryarray=sorted_commodities),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig_drill.update_xaxes(showgrid=True, gridcolor='#eeeeee', row=1, col=1)
            fig_drill.update_xaxes(
                tickformat=".0%",
                showgrid=True, gridcolor='#eeeeee', zeroline=False,
                row=1, col=2
            )
            fig_drill.update_yaxes(showgrid=True, gridcolor='#eeeeee', tickmode='linear')

            st.plotly_chart(fig_drill, use_container_width=True)

            with st.expander(f"View Raw Data for {selected_drill_country}"):
                st.dataframe(df_merged)

        else:
            st.warning(f"No data available for {selected_drill_country} in {selected_comp_year}.")


# Page 3: Yield Choropleth Map
def show_yield_map_page(df: pd.DataFrame):
    st.title("Global Yield: Relative Performance")
    st.caption(
        "Displays agricultural yield relative to the global average for that year.\n"
        "**Colors indicate deviation from the average:**\n"
        "- **Brown**: Below Global Average\n"
        "- **Light Yellow**: Near Global Average (1.0)\n"
        "- **Blue**: Above Global Average"
    )

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

    df_map = df_map.dropna(subset=["yield", "country", "commodity", "year"]).copy()

    st.sidebar.markdown("---")
    st.sidebar.header("Yield Map Filters")

    valid_commodities = sorted(df_map["commodity"].unique().tolist())
    selected_commodity = st.sidebar.selectbox("Select Commodity", valid_commodities, key="yield_commodity")

    df_commodity = df_map[df_map["commodity"] == selected_commodity].copy()

    valid_years = sorted(df_commodity["year"].unique().tolist())
    selected_year = st.sidebar.slider("Select Year", int(min(valid_years)), int(max(valid_years)),
                                      int(min(valid_years)), key="yield_year")

    df_year = df_commodity[df_commodity["year"] == selected_year].copy()
    if df_year.empty:
        st.warning("No data for this selection.")
        return

    # Calculate Relative Yield
    global_avg_yield = df_year["yield"].mean()
    df_year["relative_yield"] = df_year["yield"] / global_avg_yield

    # Symmetric Color Scale
    p05 = df_year["relative_yield"].quantile(0.05)
    p95 = df_year["relative_yield"].quantile(0.95)

    # This ensures 1.0 is always the center color (Light Yellow)
    max_dev = max(abs(p95 - 1), abs(1 - p05))
    max_dev = max(max_dev, 0.2)

    c_min = 1 - max_dev
    c_max = 1 + max_dev

    fig = px.choropleth(
        df_year,
        locations="country",
        locationmode="country names",
        color="relative_yield",
        # Custom Scale: Brown -> Light Yellow -> Blue
        color_continuous_scale=["#8B4513", "#FFFFE0", "#000080"],
        range_color=[c_min, c_max],
        hover_name="country",
        hover_data={
            "commodity": True,
            "year": True,
            "relative_yield": ":.2f",
            "yield": ":.2f"
        },
        labels={"relative_yield": "Rel. Yield"},
        title=f"Yield vs. Global Average – {selected_commodity} ({selected_year})<br><sub>Global Avg: {global_avg_yield:.2f} t/ha | 1.0 = Average</sub>",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=80, b=0),
        coloraxis_colorbar=dict(
            title="Relative<br>Yield",
            tickformat=".2f",
        )
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



