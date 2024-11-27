import plotly.express as px
import plotly.graph_objects as go
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
    "Tidak Minat": "#f7b787",
    "Minat Bidang Praktis": "rgba(255, 182, 193, 0.6)",
    "Minat Bidang Metodis": "rgba(144, 238, 144, 0.6)",
    "Minat Dasar": "rgba(173, 216, 230, 0.6)"
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

def create_top_minat_chart(df: pd.DataFrame, show_real_values: bool) -> px.bar:
    """Create stacked bar chart for top 5 minat distribution with specific top selection criteria."""
    # Prepare data
    minat_cols = [col for col in df.columns if "minat_" in col and "ket" in col]
    melted_df = df.melt(value_vars=minat_cols, var_name="minat_type", value_name="minat_ket")
    
    # Categorize minat types
    minat_categories = {
        "Minat Dasar": [f"minat_{i}_ket" for i in range(1, 9)],
        "Minat Bidang Metodis": [f"minat_{i}_ket" for i in range(9, 14)],
        "Minat Bidang Praktis": [f"minat_{i}_ket" for i in range(14, 19)],
    }
    melted_df["category"] = melted_df["minat_type"].map(
        lambda x: next((cat for cat, cols in minat_categories.items() if x in cols), None)
    )

    # Group by minat type and category, and calculate counts
    aggregated_df = melted_df.groupby(["minat_type", "category", "minat_ket"]).size().reset_index(name="count")
    minat_counts = aggregated_df[aggregated_df["minat_ket"] == "Minat"]

    # Get Top 3 Minat Dasar, Top 1 Minat Bidang Metodis, and Top 1 Minat Bidang Praktis
    top_minat_dasar = (
        minat_counts[minat_counts["category"] == "Minat Dasar"]
        .nlargest(3, "count")
        .sort_values("count", ascending=False)
    )
    top_minat_metodis = (
        minat_counts[minat_counts["category"] == "Minat Bidang Metodis"]
        .nlargest(1, "count")
    )
    top_minat_praktis = (
        minat_counts[minat_counts["category"] == "Minat Bidang Praktis"]
        .nlargest(1, "count")
    )
    
    # Combine top selections
    top_minat = pd.concat([top_minat_dasar, top_minat_metodis, top_minat_praktis])["minat_type"]

    # Ensure the order of minat types
    top_minat_ordered = list(top_minat_dasar["minat_type"]) + \
                        list(top_minat_metodis["minat_type"]) + \
                        list(top_minat_praktis["minat_type"])
    
    filtered_df = aggregated_df[aggregated_df["minat_type"].isin(top_minat)]
    filtered_df["minat_type"] = pd.Categorical(filtered_df["minat_type"], categories=top_minat_ordered, ordered=True)
    filtered_df = filtered_df.sort_values("minat_type")

    # Set chart parameters
    y_axis = "count" if show_real_values else "percentage"
    y_label = "Jumlah" if show_real_values else "Persentase (%)"
    text_template = "%{text}" if show_real_values else "%{text:.2f}%"

    # Add percentage calculation
    filtered_df["percentage"] = (
        filtered_df["count"] / filtered_df.groupby("minat_type")["count"].transform("sum")
    ) * 100

    # Create chart with separate color and opacity
    fig = go.Figure()

    # Add bars for Tidak Minat and Minat with opacity for first legend
    for minat_ket in ["Tidak Minat", "Minat"]:
        subset = filtered_df[filtered_df["minat_ket"] == minat_ket]
        fig.add_trace(go.Bar(
            x=subset["minat_type"],
            y=subset[y_axis],
            name=minat_ket,
            marker_color=CHART_COLORS[minat_ket],
            text=subset[y_axis],
            hovertext=subset[["minat_type", "minat_ket", y_axis]].apply(
                lambda row: f"{row[y_axis]}" if show_real_values else f"{row[y_axis]:.2f}%",
                axis=1
            ),  # Custom hover text
            hovertemplate="%{hovertext}",
            textposition="inside",
            texttemplate=text_template,
            textfont=dict(size=16),
            legendgroup="Tingkatan",
            legendgrouptitle_text="Tingkatan Minat"
        ))

    # Prepare category background and category legend traces
    max_y = filtered_df.groupby("minat_type")[y_axis].sum().max() * 1.1  # Add 10% padding
    x_vals = list(range(len(filtered_df["minat_type"].unique())))
    
    shapes = [
        # Vertical lines
        {
            "type": "line",
            "x0": x_vals[3] - 0.5,
            "x1": x_vals[3] - 0.5,
            "y0": 0,
            "y1": max_y,
            "line": {
                "color": "gray",
                "width": 2,
                "dash": "dash"
            }
        },
        {
            "type": "line",
            "x0": x_vals[4] - 0.5,
            "x1": x_vals[4] - 0.5,
            "y0": 0,
            "y1": max_y,
            "line": {
                "color": "gray",
                "width": 2,
                "dash": "dash"
            }
        }
    ]

    # Background shapes for categories
    background_shapes = [
        # Minat Dasar (light blue)
        dict(
            type="rect",
            x0=-0.5, x1=2.5,
            y0=0, y1=max_y,
            fillcolor="rgba(173, 216, 230, 0.2)",
            line=dict(width=0)
        ),
        # Minat Bidang Metodis (light green)
        dict(
            type="rect",
            x0=2.5, x1=3.5,
            y0=0, y1=max_y,
            fillcolor="rgba(144, 238, 144, 0.2)",
            line=dict(width=0)
        ),
        # Minat Bidang Praktis (light pink)
        dict(
            type="rect",
            x0=3.5, x1=4.5,
            y0=0, y1=max_y,
            fillcolor="rgba(255, 182, 193, 0.2)",
            line=dict(width=0)
        )
    ]

    for category, color in CHART_COLORS.items():
        if category in ["Minat Dasar", "Minat Bidang Metodis", "Minat Bidang Praktis"]:
            fig.add_trace(go.Bar(
                x=[None],
                y=[None],
                marker_color=color,
                name=category,
                showlegend=True,
                legendgroup="Kategori",
                legendgrouptitle_text="Kategori Minat"
            ))

    # Update layout
    rename_mapping = get_rename_mapping()
    fig.update_layout(
        barmode="stack",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        title="Top 5 Minat",
        yaxis_title=y_label,
        xaxis_title="Minat",
        showlegend=True,
        uniformtext_minsize=14,
        uniformtext_mode="hide",
        margin=dict(t=30),
        shapes=shapes + background_shapes
    )

    # Update x-axis ticks
    fig.update_xaxes(
        ticktext=[rename_mapping.get(val, val).replace(" ", "<br>") for val in filtered_df["minat_type"].unique()],
        tickvals=x_vals
    )

    return fig

def create_bakat_charts(df: pd.DataFrame, selected_minat: str):
    """Create side-by-side bakat distribution charts for selected minat category"""
    # Get the minat column name and rename mapping
    rename_mapping = get_rename_mapping()
    minat_name = rename_mapping.get(selected_minat, selected_minat)
    
    # Split data by minat category
    minat_mask = df[selected_minat] == "Minat"
    minat_df = df[minat_mask]
    tidak_minat_df = df[~minat_mask]
    
    # Get bakat columns
    bakat_cols = [col for col in df.columns if "bakat_" in col and "ket" in col]
    
    # Create side-by-side columns
    col1, col2 = st.columns(2)
    
    def create_single_bakat_chart(data: pd.DataFrame, title: str, container) -> None:
        if data.empty:
            return

        # Prepare data for plotting
        bakat_counts = []
        category_order = ["Baik", "Sedang", "Kurang", "Tidak Terukur"]

        for bakat_col in bakat_cols:
            # Get value counts for this bakat
            counts = data[bakat_col].value_counts()

            # Find the highest available category
            available_category = None
            for category in category_order:
                if category in counts.index:
                    available_category = category
                    break

            if available_category is not None:
                bakat_counts.append({
                    "bakat": bakat_col,
                    "count": counts[available_category],
                    "category": available_category
                })

        if not bakat_counts:
            return

        # Convert to DataFrame and sort by count
        df_counts = pd.DataFrame(bakat_counts)
        df_counts = df_counts.sort_values("count", ascending=False)  # For vertical bars

        # Create vertical bar chart
        fig = px.bar(
            df_counts,
            x="bakat",
            y="count",
            color="category",
            title=title,
            color_discrete_map=CHART_COLORS,
            labels={"category": "Kategori Bakat", "bakat": "Bakat", "count": "Jumlah"},
        )

        # Update layout
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.6,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor="white",
            uniformtext_minsize=14,
            uniformtext_mode="hide",
            xaxis_title="Bakat",
            yaxis_title="Jumlah",
        )

        # Update axes
        fig.update_xaxes(
            ticktext=[rename_mapping.get(val, val).replace(" ", "<br>") for val in df_counts["bakat"]],
            tickvals=df_counts["bakat"],
            showgrid=False
        )
        fig.update_yaxes(showgrid=False)

        # Update traces with conditional font color
        for trace in fig.data:
            trace.update(
                text=trace.y,
                textposition="auto",
                texttemplate="%{text:,}",
                textfont=dict(size=16),
                hovertemplate=(
                    "%{y:,}"
                )
            )

        # Display chart in container
        with container:
            with st.container(border=True):
                plotly_events(fig, click_event=False)
    
    # Create charts for both categories
    create_single_bakat_chart(
        minat_df, 
        f"Distribusi Bakat untuk {minat_name}\n(Kategori Minat)", 
        col1
    )
    create_single_bakat_chart(
        tidak_minat_df, 
        f"Distribusi Bakat untuk {minat_name}\n(Kategori Tidak Minat)", 
        col2
    )

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
            fig = create_top_minat_chart(final_filtered_df, show_real_values)
            selected_minat_dict = plotly_events(fig, click_event=True)

        # Display minat charts if bakat is selected
        st.info("Silakan klik salah satu bar di atas untuk melihat bakat yang bersangkutan", icon="ℹ️")

        if selected_minat_dict:
            selected_minat = selected_minat_dict[0]["x"]  # Get the minat column name
            create_bakat_charts(final_filtered_df, selected_minat)
    else:
        st.info("Silakan pilih satu atau lebih provinsi", icon="ℹ️")

if __name__ == "__main__":
    main()