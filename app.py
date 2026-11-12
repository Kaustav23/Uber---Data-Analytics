"""
app.py
------
2024 Uber NCR Ride-Sharing Business Intelligence Dashboard
Built with Python, Streamlit, Pandas, SciPy, and Plotly.
Visual design inspired by `Dasboard.jpg` and `Uber.pbix` (sleek dark-mode
Uber interface with taxi-amber, emerald, and slate accents).
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_processing import (
    compute_distance_value_statistics,
    compute_kpis,
    filter_dataframe,
    load_and_clean_data,
)

# ---------------------------------------------------------------------------
# Page Configuration & Uber Dark-Mode Theme Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Uber NCR Ride Analytics | 2024 BI Dashboard",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color Palette inspired by Dasboard.jpg (Uber Obsidian Black, Taxi Amber, Pure White)
UBER_COLORS: Dict[str, str] = {
    "bg_primary": "#0B0D13",
    "bg_card": "#141824",
    "bg_card_hover": "#1C2233",
    "border": "#262D40",
    "text_primary": "#F8FAFC",
    "text_muted": "#94A3B8",
    "accent_amber": "#F59E0B",
    "accent_gold": "#FBBF24",
    "accent_emerald": "#10B981",
    "accent_blue": "#3B82F6",
    "accent_purple": "#8B5CF6",
    "accent_rose": "#F43F5E",
    "accent_cyan": "#06B6D4",
}

PAYMENT_PALETTE: List[str] = [
    "#F59E0B",  # UPI - Uber Taxi Amber
    "#10B981",  # Cash - Emerald
    "#3B82F6",  # Uber Wallet - Royal Blue
    "#8B5CF6",  # Credit Card - Violet
    "#EC4899",  # Debit Card - Rose/Pink
]


def inject_custom_css() -> None:
    """Apply sleek, dark-mode-compatible CSS styling inspired by Dasboard.jpg."""
    st.markdown(
        f"""
        <style>
        /* Main Canvas & Typography */
        .stApp {{
            background-color: {UBER_COLORS["bg_primary"]};
            color: {UBER_COLORS["text_primary"]};
        }}
        section[data-testid="stSidebar"] {{
            background-color: #0E111A;
            border-right: 1px solid {UBER_COLORS["border"]};
        }}
        /* Hero Header Banner inspired by Dasboard.jpg */
        .uber-hero {{
            background: linear-gradient(135deg, #111420 0%, #191F30 60%, #241E14 100%);
            border: 1px solid {UBER_COLORS["border"]};
            border-left: 5px solid {UBER_COLORS["accent_amber"]};
            border-radius: 14px;
            padding: 1.35rem 1.75rem;
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }}
        .uber-brand-title {{
            font-size: 2.0rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #FFFFFF;
            margin: 0;
            line-height: 1.15;
        }}
        .uber-brand-subtitle {{
            font-size: 0.95rem;
            color: {UBER_COLORS["text_muted"]};
            margin-top: 0.3rem;
        }}
        .uber-pill {{
            display: inline-block;
            background: {UBER_COLORS["accent_amber"]};
            color: #0B0D13;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        /* Metric Card Styling */
        div[data-testid="stMetric"] {{
            background-color: {UBER_COLORS["bg_card"]};
            border: 1px solid {UBER_COLORS["border"]};
            border-radius: 12px;
            padding: 1rem 1.15rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }}
        div[data-testid="stMetricLabel"] > div {{
            color: {UBER_COLORS["text_muted"]} !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetricValue"] > div {{
            color: #FFFFFF !important;
            font-size: 1.65rem !important;
            font-weight: 800 !important;
        }}
        /* Section Header Banner */
        .section-title {{
            font-size: 1.18rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-top: 0.75rem;
            margin-bottom: 0.4rem;
            padding-left: 0.65rem;
            border-left: 4px solid {UBER_COLORS["accent_amber"]};
        }}
        /* Statistical Callout Banner */
        .stat-banner {{
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.14) 0%, rgba(59, 130, 246, 0.10) 100%);
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-radius: 10px;
            padding: 0.85rem 1.2rem;
            margin-bottom: 0.75rem;
            color: #F8FAFC;
            font-size: 0.94rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Data Loader
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning 2024 Uber NCR ride bookings...")
def get_cached_dataset(uploaded_file: Optional[object] = None) -> pd.DataFrame:
    """Load and cache the cleaned dataset for fast reactive filtering."""
    return load_and_clean_data(uploaded_file)


def format_inr(amount: float) -> str:
    """Format currency values cleanly in Indian Rupees (₹)."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr (₹{amount / 1e6:.2f}M)"
    if amount >= 1e6:
        return f"₹{amount / 1e6:.2f}M"
    if amount >= 1e3:
        return f"₹{amount:,.0f}"
    return f"₹{amount:.2f}"


# ---------------------------------------------------------------------------
# Modular Plotly Visualization Functions
# ---------------------------------------------------------------------------
def render_revenue_distribution(df: pd.DataFrame, completed_only: bool = False) -> go.Figure:
    """
    1. Revenue Distribution:
    Render an interactive Plotly donut chart showing revenue split by Payment Method
    (UPI, Cash, Credit Card, Uber Wallet, Debit Card).
    """
    source_df = df[df["Booking Status"] == "Completed"] if completed_only else df
    rev_df = (
        source_df.dropna(subset=["Payment Method", "Booking Value"])
        .groupby("Payment Method", as_index=False)
        .agg(
            Total_Revenue=("Booking Value", "sum"),
            Ride_Count=("Booking ID", "count"),
            Avg_Ticket=("Booking Value", "mean"),
        )
        .sort_values("Total_Revenue", ascending=False)
    )

    total_rev = rev_df["Total_Revenue"].sum()

    if rev_df.empty or total_rev == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Revenue Distribution by Payment Method (No Data for Current Filter)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
        )
        return fig

    fig = go.Figure(
        data=[
            go.Pie(
                labels=rev_df["Payment Method"],
                values=rev_df["Total_Revenue"],
                hole=0.58,
                marker=dict(
                    colors=PAYMENT_PALETTE[: len(rev_df)],
                    line=dict(color="#0B0D13", width=2.5),
                ),
                textinfo="label+percent",
                textposition="outside",
                customdata=np.stack(
                    [rev_df["Ride_Count"], rev_df["Avg_Ticket"]], axis=-1
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Revenue: ₹%{value:,.0f} (%{percent})<br>"
                    "Billed Rides: %{customdata[0]:,}<br>"
                    "Avg Booking Value: ₹%{customdata[1]:,.2f}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>Revenue Distribution by Payment Method</b>",
            font=dict(size=16, color="#F8FAFC"),
            x=0.02,
        ),
        annotations=[
            dict(
                text=(
                    f"<b>Total Billed</b><br>"
                    f"<span style='font-size:15px;color:#F59E0B'>₹{total_rev / 1e6:.2f}M</span>"
                ),
                x=0.5,
                y=0.5,
                font=dict(size=13, color="#F8FAFC"),
                showarrow=False,
            )
        ],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#E2E8F0"),
        ),
        margin=dict(l=20, r=20, t=55, b=60),
        height=430,
    )
    return fig


def render_fleet_performance(
    df: pd.DataFrame, vehicle_col: str = "Vehicle_Category"
) -> go.Figure:
    """
    2. Fleet Performance:
    Render a grouped bar chart comparing Total Bookings and Total Distance (km)
    across Vehicle Type.
    """
    fleet_df = (
        df.groupby(vehicle_col, as_index=False)
        .agg(
            Total_Bookings=("Booking ID", "count"),
            Completed_Rides=("Booking Status", lambda s: (s == "Completed").sum()),
            Total_Distance=("Ride_Distance_Filled", "sum"),
            Total_Revenue=("Booking_Value_Filled", "sum"),
            Avg_Distance=("Ride Distance", "mean"),
            Avg_Rating=("Customer Rating", "mean"),
        )
        .sort_values("Total_Bookings", ascending=False)
    )

    if fleet_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Fleet Performance across Vehicle Types (No Data)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
        )
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=fleet_df[vehicle_col],
            y=fleet_df["Total_Bookings"],
            name="Total Bookings (Count)",
            marker_color=UBER_COLORS["accent_amber"],
            offsetgroup=1,
            text=fleet_df["Total_Bookings"].apply(lambda v: f"{v / 1e3:.1f}K"),
            textposition="outside",
            customdata=np.stack(
                [
                    fleet_df["Completed_Rides"],
                    fleet_df["Total_Revenue"],
                    fleet_df["Avg_Rating"].fillna(0),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Total Bookings: %{y:,}<br>"
                "Completed Rides: %{customdata[0]:,}<br>"
                "Total Revenue: ₹%{customdata[1]:,.0f}<br>"
                "Avg Customer Rating: %{customdata[2]:.2f} ★"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=fleet_df[vehicle_col],
            y=fleet_df["Total_Distance"],
            name="Total Distance (km)",
            marker_color=UBER_COLORS["accent_blue"],
            offsetgroup=2,
            text=fleet_df["Total_Distance"].apply(lambda v: f"{v / 1e3:.0f}K km"),
            textposition="outside",
            customdata=fleet_df["Avg_Distance"].fillna(0),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Total Distance: %{y:,.1f} km<br>"
                "Avg Trip Distance: %{customdata:.2f} km"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text="<b>Fleet Performance: Total Bookings vs. Total Distance by Vehicle Type</b>",
            font=dict(size=16, color="#F8FAFC"),
            x=0.02,
        ),
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#E2E8F0"),
        ),
        margin=dict(l=20, r=20, t=55, b=65),
        height=430,
    )

    fig.update_xaxes(title_text="Vehicle Type", showgrid=False)
    fig.update_yaxes(
        title_text="Total Bookings (Count)",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.12)",
        range=[0, fleet_df["Total_Bookings"].max() * 1.18],
    )
    fig.update_yaxes(
        title_text="Total Ride Distance (km)",
        secondary_y=True,
        showgrid=False,
        range=[0, fleet_df["Total_Distance"].max() * 1.18],
    )
    return fig


def render_cancellation_diagnostics(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    3. Cancellation Diagnostics:
    Render horizontal bar charts breaking down the top reasons for
    Customer Cancellations vs. Driver Cancellations.
    """
    # Customer Cancellations Breakdown
    cust_df = (
        df[
            (df["Booking Status"] == "Cancelled by Customer")
            & (df["Reason for cancelling by Customer"].notna())
        ]
        .groupby("Reason for cancelling by Customer", as_index=False)
        .agg(Cancel_Count=("Booking ID", "count"))
        .sort_values("Cancel_Count", ascending=True)
    )
    cust_total = cust_df["Cancel_Count"].sum()
    if cust_total > 0:
        cust_df["Share_Pct"] = cust_df["Cancel_Count"] / cust_total * 100.0
    else:
        cust_df["Share_Pct"] = 0.0

    fig_cust = go.Figure(
        go.Bar(
            x=cust_df["Cancel_Count"],
            y=cust_df["Reason for cancelling by Customer"],
            orientation="h",
            marker=dict(
                color=cust_df["Cancel_Count"],
                colorscale=[[0, "#F59E0B"], [1, "#F43F5E"]],
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=cust_df.apply(
                lambda r: f"{int(r['Cancel_Count']):,} ({r['Share_Pct']:.1f}%)", axis=1
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Customer Cancellations: %{x:,}<br>"
                "Share of Customer Cancellations: %{customdata:.2f}%"
                "<extra></extra>"
            ),
            customdata=cust_df["Share_Pct"],
        )
    )
    fig_cust.update_layout(
        title=dict(
            text=f"<b>Customer Cancellation Reasons</b> (Total: {cust_total:,})",
            font=dict(size=15, color="#F8FAFC"),
            x=0.01,
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Cancelled Bookings",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            range=[0, (cust_df["Cancel_Count"].max() * 1.28) if not cust_df.empty else 10],
        ),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=10, r=35, t=50, b=40),
        height=360,
    )

    # Driver Cancellations Breakdown
    drv_df = (
        df[
            (df["Booking Status"] == "Cancelled by Driver")
            & (df["Driver Cancellation Reason"].notna())
        ]
        .groupby("Driver Cancellation Reason", as_index=False)
        .agg(Cancel_Count=("Booking ID", "count"))
        .sort_values("Cancel_Count", ascending=True)
    )
    drv_total = drv_df["Cancel_Count"].sum()
    if drv_total > 0:
        drv_df["Share_Pct"] = drv_df["Cancel_Count"] / drv_total * 100.0
    else:
        drv_df["Share_Pct"] = 0.0

    fig_drv = go.Figure(
        go.Bar(
            x=drv_df["Cancel_Count"],
            y=drv_df["Driver Cancellation Reason"],
            orientation="h",
            marker=dict(
                color=drv_df["Cancel_Count"],
                colorscale=[[0, "#3B82F6"], [1, "#8B5CF6"]],
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=drv_df.apply(
                lambda r: f"{int(r['Cancel_Count']):,} ({r['Share_Pct']:.1f}%)", axis=1
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Driver Cancellations: %{x:,}<br>"
                "Share of Driver Cancellations: %{customdata:.2f}%"
                "<extra></extra>"
            ),
            customdata=drv_df["Share_Pct"],
        )
    )
    fig_drv.update_layout(
        title=dict(
            text=f"<b>Driver Cancellation Reasons</b> (Total: {drv_total:,})",
            font=dict(size=15, color="#F8FAFC"),
            x=0.01,
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Cancelled Bookings",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            range=[0, (drv_df["Cancel_Count"].max() * 1.28) if not drv_df.empty else 10],
        ),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=10, r=35, t=50, b=40),
        height=360,
    )

    return fig_cust, fig_drv


def render_statistical_deep_dive(
    df: pd.DataFrame,
    stats_dict: Dict[str, float],
    max_scatter_points: int = 5000,
    completed_only: bool = False,
) -> go.Figure:
    """
    4. Statistical Deep Dive:
    Render a scatter plot with an OLS linear regression trendline analyzing the
    relationship between Ride Distance (km) and Booking Value (INR), paired with
    exact SciPy Pearson correlation metrics calculated across the full filtered dataset.
    """
    target_df = df[df["Booking Status"] == "Completed"] if completed_only else df
    valid = target_df.dropna(subset=["Ride Distance", "Booking Value"]).copy()

    if valid.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Ride Distance vs. Booking Value (No Valid Billed Rides in Selection)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
        )
        return fig

    # Sample for smooth WebGL scatter rendering if > max_scatter_points while
    # keeping the SciPy regression line & correlation exact on 100% of filtered rows.
    if len(valid) > max_scatter_points:
        plot_df = valid.sample(n=max_scatter_points, random_state=42)
        sample_note = f"Showing {max_scatter_points:,} sampled points (OLS & Pearson r computed on all {len(valid):,} rides)"
    else:
        plot_df = valid
        sample_note = f"Showing all {len(valid):,} billed rides"

    fig = px.scatter(
        plot_df,
        x="Ride Distance",
        y="Booking Value",
        color="Vehicle_Category",
        hover_data={
            "Booking ID": True,
            "Booking Status": True,
            "Vehicle Type": True,
            "Payment Method": True,
            "Ride Distance": ":.2f",
            "Booking Value": ":,.0f",
            "Vehicle_Category": False,
        },
        opacity=0.62,
        color_discrete_sequence=[
            "#F59E0B",
            "#3B82F6",
            "#10B981",
            "#8B5CF6",
            "#EC4899",
            "#06B6D4",
        ],
        labels={
            "Ride Distance": "Ride Distance (km)",
            "Booking Value": "Booking Value (₹)",
            "Vehicle_Category": "Vehicle Type",
        },
        render_mode="webgl",
    )

    # Overlay exact SciPy OLS Trendline across the full domain of Ride Distance
    x_min = float(valid["Ride Distance"].min())
    x_max = float(valid["Ride Distance"].max())
    x_line = np.linspace(x_min, x_max, 100)
    y_line = stats_dict["slope"] * x_line + stats_dict["intercept"]

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=(
                f"OLS Trendline (y = {stats_dict['slope']:.2f}x + "
                f"{stats_dict['intercept']:.1f}, r = {stats_dict['pearson_r']:.4f})"
            ),
            line=dict(color="#FFFFFF", width=3, dash="dash"),
            hovertemplate=(
                "<b>SciPy OLS Trendline</b><br>"
                "Ride Distance: %{x:.2f} km<br>"
                "Predicted Booking Value: ₹%{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Ride Distance (km) vs. Booking Value (₹) — Scatter & OLS Trendline</b> "
                f"<span style='font-size:12px;color:#94A3B8'>({sample_note})</span>"
            ),
            font=dict(size=16, color="#F8FAFC"),
            x=0.01,
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Ride Distance (km)",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
        ),
        yaxis=dict(
            title="Booking Value (₹ INR)",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#E2E8F0"),
        ),
        margin=dict(l=20, r=20, t=55, b=75),
        height=460,
    )
    return fig


def render_booking_status_trend(df: pd.DataFrame) -> go.Figure:
    """
    Bonus Contextual Visual (matching Power BI 'Overall' tab):
    Monthly Booking Volume & Completion Status Trend across 2024.
    """
    monthly = (
        df.groupby(["Month_Num", "Month_Name", "Booking Status"], as_index=False)
        .agg(Bookings=("Booking ID", "count"))
        .sort_values("Month_Num")
    )
    status_color_map = {
        "Completed": "#10B981",
        "Cancelled by Driver": "#F59E0B",
        "Cancelled by Customer": "#F43F5E",
        "No Driver Found": "#8B5CF6",
        "Incomplete": "#64748B",
    }
    fig = px.bar(
        monthly,
        x="Month_Name",
        y="Bookings",
        color="Booking Status",
        color_discrete_map=status_color_map,
        barmode="stack",
        labels={"Month_Name": "Month (2024)", "Bookings": "Total Bookings"},
    )
    fig.update_layout(
        title=dict(
            text="<b>Monthly Booking Volume & Status Mix (Jan – Dec 2024)</b>",
            font=dict(size=15, color="#F8FAFC"),
            x=0.01,
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=20, r=20, t=50, b=65),
        height=370,
    )
    return fig


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    inject_custom_css()

    # Load dataset (with sidebar uploader fallback if local CSV is moved)
    try:
        raw_df = get_cached_dataset()
    except FileNotFoundError:
        st.sidebar.warning("`ncr_ride_bookings.csv` not found in default paths. Please upload it below:")
        uploaded = st.sidebar.file_uploader("Upload ncr_ride_bookings.csv", type=["csv"])
        if uploaded is None:
            st.error("Please upload `ncr_ride_bookings.csv` to initialize the Uber BI Dashboard.")
            st.stop()
        raw_df = get_cached_dataset(uploaded)

    # -----------------------------------------------------------------------
    # Sidebar: Branding & Global Interactive Filters
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.6rem 0 1.0rem 0; border-bottom: 1px solid #262D40; margin-bottom: 1rem;">
                <div style="font-size: 2.1rem; font-weight: 900; letter-spacing: 0.08em; color: #FFFFFF;">UBER</div>
                <div style="font-size: 0.8rem; color: #F59E0B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">
                    NCR Ride-Sharing Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("🎛️ Global Filters")

        # 1. Date Range Slider
        min_date: date = raw_df["Booking_Date"].min()
        max_date: date = raw_df["Booking_Date"].max()
        selected_date_range = st.slider(
            "Date Range (2024)",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            help="Filter bookings across the 2024 calendar year (Jan 1, 2024 – Dec 30, 2024).",
        )

        # Vehicle Grouping Mode
        use_granular_vehicles = st.toggle(
            "Split eBike & Bike separately",
            value=False,
            help=(
                "When off, 'eBike' and 'Bike' are combined into 'eBike/Bike' (6 categories) "
                "to match the Power BI reference layout. Toggle on to inspect all 7 raw vehicle types."
            ),
        )
        vehicle_col = "Vehicle Type" if use_granular_vehicles else "Vehicle_Category"

        # 2. Multiselect for Vehicle Type
        all_vehicles = sorted(raw_df[vehicle_col].dropna().unique().tolist())
        selected_vehicles = st.multiselect(
            "Vehicle Type",
            options=all_vehicles,
            default=all_vehicles,
            help="Filter by Uber fleet category (Go Mini, Go Sedan, Auto, eBike/Bike, UberXL, Premier Sedan).",
        )

        # 3. Multiselect for Payment Method
        all_payments = ["UPI", "Cash", "Uber Wallet", "Credit Card", "Debit Card"]
        available_payments = [
            p for p in all_payments if p in raw_df["Payment Method"].dropna().unique()
        ]
        selected_payments = st.multiselect(
            "Payment Method",
            options=available_payments,
            default=available_payments,
            help="Filter billed rides by payment method (UPI, Cash, Uber Wallet, Credit Card, Debit Card).",
        )

        st.divider()
        st.subheader("⚙️ Metric & Filter Settings")

        include_unbilled = st.checkbox(
            "Retain unbilled cancelled rides",
            value=True,
            help=(
                "Cancelled and 'No Driver Found' rides do not have a billed payment method (NaN). "
                "Keep this enabled so Cancellation Diagnostics and Success/Cancellation rates remain populated."
            ),
        )

        rate_mode = st.radio(
            "Rate Calculation Basis",
            options=[
                "Power BI Baseline (Excl. Incomplete)",
                "All Bookings (Incl. Incomplete)",
            ],
            index=0,
            help=(
                "Power BI Baseline excludes the 9,000 'Incomplete' rides from status share, yielding "
                "the exact 65.96% Success Rate, 19.15% Driver Cancellation, and 7.45% Customer Cancellation. "
                "All Bookings uses all 150,000 rows (62.00% Success, 25.00% Total Customer+Driver Cancellation)."
            ),
        )

        completed_only_revenue = st.checkbox(
            "Count revenue on Completed rides only",
            value=False,
            help=(
                "In the raw dataset, both Completed (93,000) and Incomplete (9,000) rides carry Booking Value "
                "(₹51.85M total vs. ₹47.26M Completed only)."
            ),
        )

    # Apply global filters
    filtered_df = filter_dataframe(
        raw_df,
        date_range=selected_date_range,
        selected_vehicles=selected_vehicles,
        selected_payments=selected_payments,
        vehicle_col=vehicle_col,
        include_unbilled_rides=include_unbilled,
    )

    kpis = compute_kpis(filtered_df)

    # -----------------------------------------------------------------------
    # Hero Banner
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="uber-hero">
            <div>
                <div class="uber-brand-title">🚖 Uber NCR Ride-Sharing Analytics Dashboard</div>
                <div class="uber-brand-subtitle">
                    2024 Operational & Financial Intelligence • Showing <b>{kpis['total_bookings']:,}</b> records
                    (<b>{kpis['unique_bookings']:,}</b> unique Booking IDs) from <b>{selected_date_range[0]}</b> to <b>{selected_date_range[1]}</b>
                </div>
            </div>
            <div>
                <span class="uber-pill">2024 NCR Dataset • Live BI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # 4. Key Performance Indicators (KPIs) Section
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">Key Performance Indicators (KPIs)</div>', unsafe_allow_html=True)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric(
            label="Total Bookings vs. Completed Rides",
            value=f"{kpis['total_bookings']:,}",
            delta=f"{kpis['completed_rides']:,} Completed ({kpis['unique_bookings']:,} Unique IDs)",
            delta_color="normal",
            help=(
                f"Total Rows: {kpis['total_bookings']:,} | Unique Booking IDs: {kpis['unique_bookings']:,} "
                f"(Power BI card shows 148.77K) | Completed Rides: {kpis['completed_rides']:,} | "
                f"Incomplete Rides: {kpis['incomplete_rides']:,} | No Driver Found: {kpis['no_driver_found']:,}"
            ),
        )

    with kpi_col2:
        displayed_rev = kpis["completed_revenue"] if completed_only_revenue else kpis["total_revenue"]
        rev_sub_label = (
            f"Avg ₹{kpis['avg_booking_value']:,.0f}/ride • All Billed: ₹{kpis['total_revenue'] / 1e6:.2f}M"
            if completed_only_revenue
            else f"Completed: ₹{kpis['completed_revenue'] / 1e6:.2f}M • Avg ₹{kpis['avg_booking_value']:,.0f}/ride"
        )
        st.metric(
            label="Total Revenue (Booking Value)",
            value=format_inr(displayed_rev),
            delta=rev_sub_label,
            delta_color="normal",
            help=(
                f"Total Booking Value (Completed + Incomplete): ₹{kpis['total_revenue']:,.0f} "
                f"| Completed Rides Revenue: ₹{kpis['completed_revenue']:,.0f} "
                f"| Total Distance: {kpis['total_distance']:,.1f} km"
            ),
        )

    with kpi_col3:
        if rate_mode.startswith("Power BI Baseline"):
            succ_val = kpis["success_rate_baseline"]
            canc_sub = (
                f"Cancel: {kpis['driver_cancel_rate_baseline'] + kpis['customer_cancel_rate_baseline']:.2f}% "
                f"({kpis['driver_cancel_rate_baseline']:.2f}% Drv, {kpis['customer_cancel_rate_baseline']:.2f}% Cust) | "
                f"All-Row Cancel: {kpis['total_cancel_rate_all']:.2f}%"
            )
        else:
            succ_val = kpis["success_rate_all"]
            canc_sub = (
                f"Total Cancel: {kpis['total_cancel_rate_all']:.2f}% "
                f"({kpis['driver_cancel_rate_all']:.2f}% Drv, {kpis['customer_cancel_rate_all']:.2f}% Cust)"
            )
        st.metric(
            label="Overall Success Rate (%)",
            value=f"{succ_val:.2f}%",
            delta=canc_sub,
            delta_color="off",
            help=(
                "Baseline Verification:\n"
                f"• Excluding Incomplete (141K active bookings): {kpis['success_rate_baseline']:.2f}% Success, "
                f"{kpis['driver_cancel_rate_baseline']:.2f}% Driver Cancel, {kpis['customer_cancel_rate_baseline']:.2f}% Customer Cancel, "
                f"{kpis['no_driver_rate_baseline']:.2f}% No Driver Found.\n"
                f"• Across All 150K Bookings: {kpis['success_rate_all']:.2f}% Success, "
                f"{kpis['total_cancel_rate_all']:.2f}% Total Cancellation ({kpis['driver_cancel_rate_all']:.2f}% Driver + "
                f"{kpis['customer_cancel_rate_all']:.2f}% Customer)."
            ),
        )

    with kpi_col4:
        st.metric(
            label="Avg Customer & Driver Ratings",
            value=f"{kpis['avg_customer_rating']:.2f} ★ / {kpis['avg_driver_rating']:.2f} ★",
            delta=f"Avg VTAT: {kpis['avg_vtat']:.1f}m • Avg CTAT: {kpis['avg_ctat']:.1f}m",
            delta_color="normal",
            help=(
                f"Average Customer Rating: {kpis['avg_customer_rating']:.2f} / 5.00 | "
                f"Average Driver Rating: {kpis['avg_driver_rating']:.2f} / 5.00 | "
                f"Average Vehicle Turnaround Time (VTAT): {kpis['avg_vtat']:.2f} mins | "
                f"Average Customer Turnaround Time (CTAT): {kpis['avg_ctat']:.2f} mins"
            ),
        )

    # Baseline Verification Expander for Audit Transparency
    with st.expander("📊 Baseline Metric Verification & Power BI Reconciliation Summary", expanded=False):
        st.markdown(
            f"""
            | Metric Dimension | Power BI Active-Status Basis (Excl. 9,000 Incomplete) | Full Dataset Basis (All 150,000 Rows) | Current Filtered View |
            | :--- | :--- | :--- | :--- |
            | **Booking Volume** | 141,000 rows (139,912 unique IDs) | 150,000 rows (**148,767** unique IDs ≈ **148.77K**) | **{kpis['total_bookings']:,}** rows (**{kpis['unique_bookings']:,}** unique IDs) |
            | **Success Rate (Completed)** | **65.96%** (93,000 / 141,000) | **62.00%** (93,000 / 150,000) | **{kpis['success_rate_baseline']:.2f}%** (Excl. Inc.) / **{kpis['success_rate_all']:.2f}%** (All) |
            | **Total Cancellation Rate** | **26.60%** (37,500 / 141,000) | **25.00%** (37,500 / 150,000) | **{kpis['total_cancel_rate_all']:.2f}%** (All rows) |
            | **Driver Cancellation Rate** | **19.15%** (27,000 / 141,000) | **18.00%** (27,000 / 150,000) | **{kpis['driver_cancel_rate_baseline']:.2f}%** (Excl. Inc.) / **{kpis['driver_cancel_rate_all']:.2f}%** (All) |
            | **Customer Cancellation Rate** | **7.45%** (10,500 / 141,000) | **7.00%** (10,500 / 150,000) | **{kpis['customer_cancel_rate_baseline']:.2f}%** (Excl. Inc.) / **{kpis['customer_cancel_rate_all']:.2f}%** (All) |
            | **No Driver Found** | **7.45%** (10,500 / 141,000) | **7.00%** (10,500 / 150,000) | **{kpis['no_driver_rate_baseline']:.2f}%** (Excl. Inc.) |
            """
        )

    # -----------------------------------------------------------------------
    # 5A. Revenue Distribution & Fleet Performance
    # -----------------------------------------------------------------------
    st.markdown(
        '<div class="section-title">Revenue Distribution & Fleet Performance</div>',
        unsafe_allow_html=True,
    )
    chart_col1, chart_col2 = st.columns([1, 1.35])

    with chart_col1:
        fig_donut = render_revenue_distribution(
            filtered_df, completed_only=completed_only_revenue
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        fig_fleet = render_fleet_performance(filtered_df, vehicle_col=vehicle_col)
        st.plotly_chart(fig_fleet, use_container_width=True)

    # -----------------------------------------------------------------------
    # 5B. Cancellation Diagnostics (Customer vs. Driver Reasons)
    # -----------------------------------------------------------------------
    st.markdown(
        '<div class="section-title">Cancellation Diagnostics: Customer vs. Driver Breakdown</div>',
        unsafe_allow_html=True,
    )
    cancel_col1, cancel_col2 = st.columns(2)
    fig_cust_cancel, fig_drv_cancel = render_cancellation_diagnostics(filtered_df)

    with cancel_col1:
        st.plotly_chart(fig_cust_cancel, use_container_width=True)

    with cancel_col2:
        st.plotly_chart(fig_drv_cancel, use_container_width=True)

    # -----------------------------------------------------------------------
    # 5C. Statistical Deep Dive (Ride Distance vs. Booking Value + SciPy Pearson r)
    # -----------------------------------------------------------------------
    st.markdown(
        '<div class="section-title">Statistical Deep Dive: Ride Distance vs. Booking Value</div>',
        unsafe_allow_html=True,
    )

    stats_result = compute_distance_value_statistics(
        filtered_df, completed_only=completed_only_revenue
    )

    # Prominent SciPy Pearson Correlation Display above the Scatter Plot
    r_val = stats_result["pearson_r"]
    p_val = stats_result["p_value"]
    n_val = stats_result["n_samples"]
    r2_val = stats_result["r_squared"]
    corr_strength = (
        "near-zero / statistically independent"
        if abs(r_val) < 0.05
        else ("weak positive" if r_val > 0 else "weak negative")
    )

    stat_m1, stat_m2, stat_m3, stat_m4 = st.columns(4)
    stat_m1.metric("Pearson Correlation (r)", f"{r_val:.4f}", delta=corr_strength, delta_color="off")
    stat_m2.metric("Two-Tailed p-value (SciPy)", f"{p_val:.4e}", delta="α = 0.05 threshold", delta_color="off")
    stat_m3.metric(
        "OLS Regression Equation",
        f"y = {stats_result['slope']:.2f}x + {stats_result['intercept']:.1f}",
        delta=f"R² = {r2_val:.6f}",
        delta_color="off",
    )
    stat_m4.metric("Analyzed Billed Rides (n)", f"{n_val:,}", delta="Valid Distance & Value pairs", delta_color="off")

    st.markdown(
        f"""
        <div class="stat-banner">
            <b>🔬 SciPy Statistical Finding:</b> Across <b>{n_val:,}</b> billed rides, the Pearson correlation coefficient between
            <b>Ride Distance (km)</b> and <b>Booking Value (₹)</b> is <b>r = {r_val:.4f}</b> (<i>p</i> = {p_val:.4f}, R² = {r2_val:.6f}).
            This indicates that in the 2024 synthetic/operational NCR dataset, fare values are distributed largely independently of
            trip distance across vehicle classes (reflecting dynamic surge pricing, flat/minimum fares, and multi-tier bookings).
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_scatter = render_statistical_deep_dive(
        filtered_df,
        stats_dict=stats_result,
        max_scatter_points=5000,
        completed_only=completed_only_revenue,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------------
    # Additional Operational Context: Monthly Seasonality & Top Customers
    # -----------------------------------------------------------------------
    with st.expander("📅 View Monthly Seasonality & Top Customer Leaderboard (Power BI Tabs)", expanded=False):
        sub_col1, sub_col2 = st.columns([1.5, 1])
        with sub_col1:
            st.plotly_chart(render_booking_status_trend(filtered_df), use_container_width=True)
        with sub_col2:
            st.markdown("##### 🏆 Top 10 Customers by Total Booking Value")
            top_cust = (
                filtered_df.dropna(subset=["Booking Value"])
                .groupby("Customer ID", as_index=False)
                .agg(
                    Total_Spend=("Booking Value", "sum"),
                    Rides=("Booking ID", "count"),
                    Avg_Rating=("Customer Rating", "mean"),
                )
                .sort_values("Total_Spend", ascending=False)
                .head(10)
            )
            top_cust["Total_Spend"] = top_cust["Total_Spend"].apply(lambda v: f"₹{v:,.0f}")
            top_cust["Avg_Rating"] = top_cust["Avg_Rating"].apply(lambda v: f"{v:.2f} ★")
            st.dataframe(top_cust, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()