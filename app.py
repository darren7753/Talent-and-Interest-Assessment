import plotly.express as px
import streamlit as st
import pandas as pd
import json
from streamlit_plotly_events import plotly_events
from typing import List, Dict, Optional, Tuple

# Constants and configurations
BAKAT_COLUMNS = [f"bakat_{i}_ket" for i in range(1, 8)]
MINAT_THRESHOLD = 60
PAGE_TITLE = "ABM Report"
CHART_COLORS = {
    "Tidak Terukur": "#f7b787",
    "Kurang": "#fed76a",
    "Sedang": "#64ccc5",
    "Baik": "#176b87",
    "Minat": "#64ccc5",
    "Tidak Minat": "#f7b787"
}

def initialize_page():
    """Initialize Streamlit page configuration"""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.markdown("<h1 style='text-align: center;'>Analisis Bakat dan Minat</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")

@st.cache_data
def load_filter_options_from_json() -> Dict:
    """Load filter options from JSON file"""
    with open("filters.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_filtered_data(province: Optional[List[str]] = None, city: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and filter data based on province and city selections
    
    Args:
        province: List of selected provinces
        city: List of selected cities
    
    Returns:
        Filtered DataFrame
    """
    query = []
    if province:
        query.append(("nm_prop", "in", province))
    if city:
        query.append(("nm_rayon", "in", city))
    return pd.read_parquet("skor-abm-detail.parquet", filters=query)

def get_rename_mapping() -> Dict[str, str]:
    """Return mapping dictionary for renaming columns to display names"""
    return {
        # Bakat mapping
        **{f"bakat_{i}_ket": label for i, label in enumerate([
            "Kemampuan Spasial", "Kemampuan Verbal", "Penalaran", 
            "Kemampuan Klerikal", "Kemampuan Mekanika", 
            "Kemampuan Kuantitatif", "Kemampuan Bahasa"
        ], 1)},
        # Minat mapping
        **{f"minat_{i}_ket": label for i, label in enumerate([
            "Fasilitasi Sosial", "Pengelolaan", "Detail Bisnis", 
            "Pengelolaan Data", "Keteknikan", "Kerja Lapangan",
            "Kesenian", "Helping", "Sains Sosial", "Influence",
            "Sistem Bisnis", "Analisis Finansial", "Kerja Ilmiah",
            "Quality Control", "Kerja Manual", "Personal Service",
            "Keteknisian", "Layanan Dasar"
        ], 1)}
    }

def create_filters() -> Tuple[List[str], List[str], List[str]]:
    """Create and display filter widgets"""
    filters = load_filter_options_from_json()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_provinces = st.multiselect(
            "Pilih Provinsi (Wajib)", 
            options=sorted(filters["nm_prop"]),
            help="Pilih satu atau lebih provinsi untuk filter data"
        )

    province_filtered_df = None
    available_cities = []
    available_schools = []
    
    if selected_provinces:
        province_filtered_df = load_filtered_data(province=selected_provinces)
        available_cities = sorted(province_filtered_df["nm_rayon"].unique())

    with col2:
        selected_cities = st.multiselect(
            "Pilih Kabupaten/Kota (Opsional)",
            options=available_cities,
            help="Pilih satu atau lebih kota untuk filter data"
        )

    if selected_cities:
        city_filtered_df = load_filtered_data(province=selected_provinces, city=selected_cities)
        available_schools = sorted(city_filtered_df["nm_sek"].unique())

    with col3:
        selected_schools = st.multiselect(
            "Pilih Sekolah (Opsional)",
            options=available_schools,
            help="Pilih satu atau lebih sekolah untuk filter data"
        )

    return selected_provinces, selected_cities, selected_schools

def process_data(selected_provinces: List[str], selected_cities: List[str], 
                selected_schools: List[str]) -> pd.DataFrame:
    """Process and filter data based on selections"""
    if not selected_provinces:
        return pd.DataFrame()

    if selected_schools:
        df = load_filtered_data(province=selected_provinces, city=selected_cities)
        return df[df["nm_sek"].isin(selected_schools)]
    elif selected_cities:
        return load_filtered_data(province=selected_provinces, city=selected_cities)
    else:
        return load_filtered_data(province=selected_provinces)

def create_bakat_chart(df: pd.DataFrame, show_real_values: bool) -> px.bar:
    """Create stacked bar chart for bakat distribution"""
    # Prepare data
    melted_df = df.melt(value_vars=BAKAT_COLUMNS, var_name="bakat_type", value_name="bakat_ket")
    aggregated_df = melted_df.groupby(["bakat_type", "bakat_ket"]).size().reset_index(name="count")
    aggregated_df["percentage"] = (
        aggregated_df["count"] / aggregated_df.groupby("bakat_type")["count"].transform("sum")
    ) * 100

    # Set chart parameters
    y_axis = "count" if show_real_values else "percentage"
    y_label = "Jumlah" if show_real_values else "Persentase (%)"
    text_template = "%{text}" if show_real_values else "%{text:.2f}%"

    # Create chart
    fig = px.bar(
        aggregated_df,
        x="bakat_type",
        y=y_axis,
        color="bakat_ket",
        title="Distribusi Kategori Bakat",
        category_orders={"bakat_ket": ["Tidak Terukur", "Kurang", "Sedang", "Baik"]},
        labels={"bakat_type": "Bakat", y_axis: y_label, "bakat_ket": "Kategori Bakat"},
        color_discrete_map=CHART_COLORS
    )

    # Update layout and formatting
    rename_mapping = get_rename_mapping()
    fig.update_layout(
        barmode="stack",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        showlegend=True,
        uniformtext_minsize=14,
        uniformtext_mode="hide",
        margin=dict(t=30)
    )

    fig.update_xaxes(
        ticktext=[rename_mapping.get(val, val).replace(" ", "<br>") for val in aggregated_df["bakat_type"].unique()],
        tickvals=aggregated_df["bakat_type"].unique()
    )

    for trace in fig.data:
        trace.update(
            text=trace.y,
            textposition="inside",
            texttemplate=text_template,
            textfont=dict(size=16, color="#424242")
        )

    return fig

def create_minat_charts(df: pd.DataFrame, selected_bakat: str):
    """Create minat distribution charts for selected bakat category"""
    minat_cols = [col for col in df.columns if "minat_" in col and "ket" in col]
    minat_df = df[[selected_bakat] + minat_cols]
    
    cols = st.columns(2)
    for i, category in enumerate(["Baik", "Sedang", "Kurang", "Tidak Terukur"]):
        create_single_minat_chart(minat_df, selected_bakat, category, cols[i % 2])

def create_single_minat_chart(minat_df: pd.DataFrame, selected_bakat: str, 
                            category: str, col):
    """Create a single minat chart for a specific bakat category"""
    # Filter and prepare data
    category_filtered_df = minat_df[minat_df[selected_bakat] == category]
    minat_cols = [col for col in minat_df.columns if "minat_" in col and "ket" in col]
    
    minat_aggregated = category_filtered_df.melt(
        id_vars=[selected_bakat],
        value_vars=minat_cols,
        var_name="minat_type",
        value_name="minat_ket"
    ).groupby(["minat_type", "minat_ket"]).size().reset_index(name="count")

    minat_aggregated["percentage"] = (
        minat_aggregated["count"] / 
        minat_aggregated.groupby("minat_type")["count"].transform("sum")
    ) * 100

    # Get top 5 minat categories
    top_minat = (
        minat_aggregated[minat_aggregated["minat_ket"] == "Minat"]
        .nlargest(5, "count")
        .reset_index()
    )
    top_categories = top_minat["minat_type"].unique()
    filtered_df = minat_aggregated[minat_aggregated["minat_type"].isin(top_categories)]
    filtered_df = filtered_df.sort_values(by=["percentage"], ascending=True)

    # Create chart
    rename_mapping = get_rename_mapping()
    fig = px.bar(
        filtered_df,
        x="minat_type",
        y="percentage",
        color="minat_ket",
        title=f"Top 5 Minat untuk '{rename_mapping.get(selected_bakat)}' Kategori '{category}'",
        category_orders={"minat_ket": ["Tidak Minat", "Minat"]},
        color_discrete_map=CHART_COLORS,
        labels={"percentage": "Persentase (%)", "minat_type": "Minat", "minat_ket": "Kategori Minat"},
    )

    # Update layout and formatting
    fig.update_layout(
        barmode="stack",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        showlegend=True,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        margin=dict(t=30)
    )

    fig.update_traces(
        texttemplate="%{y:.2f}%",
        textposition="inside",
        textfont=dict(size=12, color="#424242")
    )

    fig.update_xaxes(
        ticktext=[rename_mapping.get(val, val).replace(" ", "<br>") 
                 for val in filtered_df["minat_type"].unique()],
        tickvals=filtered_df["minat_type"].unique()
    )

    with col:
        with st.container(border=True):
            plotly_events(fig, click_event=False)

def main():
    """Main application function"""
    initialize_page()
    
    # Create filters
    selected_provinces, selected_cities, selected_schools = create_filters()
    
    # Process data based on selections
    final_filtered_df = process_data(selected_provinces, selected_cities, selected_schools)
    
    if not final_filtered_df.empty:
        # Process minat columns
        minat_cols = final_filtered_df.columns[final_filtered_df.columns.str.contains("minat")]
        for minat_col in minat_cols:
            final_filtered_df[f"{minat_col}_ket"] = final_filtered_df[minat_col].apply(
                lambda x: "Minat" if x >= MINAT_THRESHOLD else "Tidak Minat"
            )

        # Display data table
        with st.expander("Klik di sini untuk melihat data"):
            st.dataframe(final_filtered_df, hide_index=True, use_container_width=True)

        # Create and display bakat chart
        with st.container(border=True):
            show_real_values = st.checkbox("Tampilkan Nilai Real", value=False)
            fig = create_bakat_chart(final_filtered_df, show_real_values)
            selected_bakat_dict = plotly_events(fig, click_event=True)

        # Display minat charts if bakat is selected
        st.info("Silakan klik salah satu bar di atas untuk melihat minat", icon="ℹ️")
        if selected_bakat_dict:
            create_minat_charts(final_filtered_df, selected_bakat_dict[0]["x"])
    else:
        st.info("Silakan pilih satu atau lebih provinsi", icon="ℹ️")

if __name__ == "__main__":
    main()