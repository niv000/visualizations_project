import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Forecast Production Trends", layout="wide")

st.title("Alternative 1: Forecast Production Trends")
st.caption("Prediction based data")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic validation
    required = {"country", "commodity", "year", "production"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean types
    df = df.dropna(subset=["country", "commodity", "year", "production"]).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df = df.dropna(subset=["year", "production"])
    df["year"] = df["year"].astype(int)

    return df

DATA_PATH = "processed_wide.csv"

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load data from '{DATA_PATH}'.\n\n{e}")
    st.stop()

st.sidebar.header("Filters")

commodities = sorted(df["commodity"].unique().tolist())
selected_commodity = st.sidebar.selectbox("Commodity", commodities)

df_c = df[df["commodity"] == selected_commodity].copy()

countries = sorted(df_c["country"].unique().tolist())
default_countries = countries[:6]
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=default_countries if len(countries) >= len(default_countries) else countries
)

min_year = int(df_c["year"].min())
max_year = int(df_c["year"].max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

show_avg = st.sidebar.checkbox("Show average of all countries", value=True) #check if needed...?

df_plot = df_c[
    (df_c["country"].isin(selected_countries)) &
    (df_c["year"].between(year_range[0], year_range[1]))
].copy()

if df_plot.empty:
    st.warning("No data matches to the filters, Try selecting more countries or widening the year range.")
    st.stop()

fig = px.line(
    df_plot,
    x="year",
    y="production",
    color="country",
    markers=True,
    labels={
        "year": "Year",
        "production": "Predicted production",
        "country": "Country"
    },
    title=f"Predicted production over time – {selected_commodity}"
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
st.plotly_chart(fig, width='stretch')

with st.expander("Show sample rows"):
    st.dataframe(df_plot.head(50), use_container_width=True)


