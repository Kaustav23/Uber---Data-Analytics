"""
data_processing.py
------------------
Data ingestion, cleaning, transformation, and statistical aggregation module
for the 2024 Uber NCR Ride-Sharing Analytics Dashboard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


# Default candidate locations for the dataset so the app runs seamlessly
# both locally (in VS Code / GitHub / Streamlit Cloud) and in the workspace.
DEFAULT_CSV_CANDIDATES: List[str] = [
    "ncr_ride_bookings.csv",
    "data/ncr_ride_bookings.csv",
    "./ncr_ride_bookings.csv",
    "/working_dir/c_59d1332a1c6aeec1/user_uploaded_files/ttl=63d/user_uploaded/ncr_ride_bookings.csv",
]

# Mapping raw Vehicle Type values to the 6-category grouped convention
# referenced in the business requirements and Power BI dashboard ("eBike/Bike").
VEHICLE_GROUP_MAP: Dict[str, str] = {
    "Go Mini": "Go Mini",
    "Go Sedan": "Go Sedan",
    "Auto": "Auto",
    "Bike": "eBike/Bike",
    "eBike": "eBike/Bike",
    "Uber XL": "UberXL",
    "UberXL": "UberXL",
    "Premier Sedan": "Premier Sedan",
}


def resolve_data_path(custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Resolve the path to `ncr_ride_bookings.csv` across common project locations."""
    if custom_path is not None:
        candidate = Path(custom_path)
        if candidate.is_file():
            return candidate

    for path_str in DEFAULT_CSV_CANDIDATES:
        candidate = Path(path_str)
        if candidate.is_file():
            return candidate

    # Fallback recursive search in current working directory
    cwd = Path.cwd()
    for match in cwd.rglob("ncr_ride_bookings.csv"):
        if match.is_file():
            return match

    return None


def load_and_clean_data(
    file_source: Optional[Union[str, Path, object]] = None,
) -> pd.DataFrame:
    """
    Ingest and clean the 2024 Uber NCR Ride Bookings dataset.

    Cleaning steps performed:
    1. Strip literal double quotes (`"..."`) from `Booking ID` and `Customer ID`.
    2. Parse `Date` and `Time` into proper datetime objects (`Booking_Datetime`,
       `Booking_Date`, `Hour`, `Day_of_Week`, `Month_Name`, `Month_Period`).
    3. Coerce numeric columns (`Booking Value`, `Ride Distance`, `Avg VTAT`,
       `Avg CTAT`, `Driver Ratings`, `Customer Rating`) to float64.
    4. Handle expected nulls in cancellation/incomplete reason columns by
       creating clean display columns (`Customer_Cancel_Reason_Clean`,
       `Driver_Cancel_Reason_Clean`, `Incomplete_Reason_Clean`) while preserving
       strict filtering for actual cancelled/incomplete bookings.
    5. Create standardized vehicle category (`Vehicle_Category`) combining
       `Bike` and `eBike` into `eBike/Bike` and normalizing `Uber XL` to `UberXL`.

    Parameters
    ----------
    file_source : str, Path, file-like object, or None
        Path or file buffer for `ncr_ride_bookings.csv`. If None, resolves automatically.

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched DataFrame ready for analytics and visualization.
    """
    if file_source is None or isinstance(file_source, (str, Path)):
        resolved = resolve_data_path(file_source)
        if resolved is None:
            raise FileNotFoundError(
                "Could not locate 'ncr_ride_bookings.csv'. Place it in the project root "
                "directory or upload it via the sidebar."
            )
        df = pd.read_csv(resolved, low_memory=False)
    else:
        df = pd.read_csv(file_source, low_memory=False)

    # 1. Clean string identifiers (remove literal surrounding quotes)
    for id_col in ["Booking ID", "Customer ID"]:
        if id_col in df.columns:
            df[id_col] = (
                df[id_col]
                .astype(str)
                .str.strip()
                .str.strip('"')
                .str.strip("'")
            )

    # Clean categorical text columns
    for cat_col in ["Booking Status", "Vehicle Type", "Pickup Location", "Drop Location"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).str.strip()

    # 2. Convert Date and Time into proper datetime objects
    df["Date_Parsed"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    if df["Date_Parsed"].isna().any():
        df["Date_Parsed"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Booking_Date"] = df["Date_Parsed"].dt.date

    # Parse Time (HH:MM:SS) and combine with Date
    time_str = df["Time"].astype(str).str.strip()
    df["Booking_Datetime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + time_str,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    df["Hour"] = df["Booking_Datetime"].dt.hour
    df["Day_of_Week"] = df["Date_Parsed"].dt.day_name()
    df["Month_Num"] = df["Date_Parsed"].dt.month
    df["Month_Name"] = df["Date_Parsed"].dt.strftime("%b %Y")

    # 3. Ensure numeric columns are properly typed
    numeric_cols = [
        "Avg VTAT",
        "Avg CTAT",
        "Cancelled Rides by Customer",
        "Cancelled Rides by Driver",
        "Incomplete Rides",
        "Booking Value",
        "Ride Distance",
        "Driver Ratings",
        "Customer Rating",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary flag columns (0 where null)
    for flag_col in [
        "Cancelled Rides by Customer",
        "Cancelled Rides by Driver",
        "Incomplete Rides",
    ]:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].fillna(0).astype(int)

    # Zero-filled helper columns for summation while keeping NaN in original columns
    # so mean/correlation calculations are not skewed by unbilled cancelled rides.
    df["Booking_Value_Filled"] = df["Booking Value"].fillna(0.0)
    df["Ride_Distance_Filled"] = df["Ride Distance"].fillna(0.0)

    # 4. Handle nulls in cancellation & incomplete reason columns appropriately
    df["Reason for cancelling by Customer"] = (
        df["Reason for cancelling by Customer"]
        .where(df["Reason for cancelling by Customer"].notna(), None)
    )
    df["Driver Cancellation Reason"] = (
        df["Driver Cancellation Reason"]
        .where(df["Driver Cancellation Reason"].notna(), None)
    )
    df["Incomplete Rides Reason"] = (
        df["Incomplete Rides Reason"]
        .where(df["Incomplete Rides Reason"].notna(), None)
    )

    df["Customer_Cancel_Reason_Clean"] = df["Reason for cancelling by Customer"].fillna(
        "Not Applicable (Not Cancelled by Customer)"
    )
    df["Driver_Cancel_Reason_Clean"] = df["Driver Cancellation Reason"].fillna(
        "Not Applicable (Not Cancelled by Driver)"
    )
    df["Incomplete_Reason_Clean"] = df["Incomplete Rides Reason"].fillna(
        "Not Applicable (Completed or Cancelled)"
    )

    # Handle Payment Method nulls (nulls occur on Cancelled / No Driver Found rides)
    df["Payment Method"] = df["Payment Method"].where(df["Payment Method"].notna(), None)
    df["Payment_Method_Clean"] = df["Payment Method"].fillna("Unbilled (Cancelled / No Driver)")

    # 5. Standardized Vehicle Category (6 grouped categories + raw 7 categories)
    df["Vehicle_Category"] = (
        df["Vehicle Type"].map(VEHICLE_GROUP_MAP).fillna(df["Vehicle Type"])
    )

    return df


def filter_dataframe(
    df: pd.DataFrame,
    date_range: Tuple,
    selected_vehicles: List[str],
    selected_payments: List[str],
    vehicle_col: str = "Vehicle_Category",
    include_unbilled_rides: bool = True,
) -> pd.DataFrame:
    """
    Apply global sidebar filters (Date Range, Vehicle Type, Payment Method) to the dataset.

    Note on `include_unbilled_rides`:
    In `ncr_ride_bookings.csv`, cancelled rides (`Cancelled by Customer`,
    `Cancelled by Driver`, `No Driver Found`) have `NaN` for `Payment Method`.
    When `include_unbilled_rides=True`, filtering by Payment Method filters
    billed rides by the selected payment methods while retaining unbilled
    cancelled rides so cancellation rates and cancellation diagnostics remain
    accurate. If the user deselects specific payment methods to inspect only
    those payment transactions, setting `include_unbilled_rides=False` restricts
    strictly to rows matching `selected_payments`.
    """
    mask = pd.Series(True, index=df.index)

    # Date filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range[0], date_range[1]
        mask &= (df["Booking_Date"] >= start_date) & (df["Booking_Date"] <= end_date)

    # Vehicle Type filter
    if selected_vehicles:
        mask &= df[vehicle_col].isin(selected_vehicles)
    else:
        mask &= False

    # Payment Method filter
    if selected_payments:
        if include_unbilled_rides:
            mask &= df["Payment Method"].isin(selected_payments) | df["Payment Method"].isna()
        else:
            mask &= df["Payment Method"].isin(selected_payments)
    else:
        if include_unbilled_rides:
            mask &= df["Payment Method"].isna()
        else:
            mask &= False

    return df.loc[mask].copy()


def compute_kpis(df: pd.DataFrame, deduplicate_booking_id: bool = False) -> Dict[str, float]:
    """
    Calculate dynamic top-level KPIs and baseline verification metrics.

    Supports both rate conventions present in the dataset & Power BI reference:
    - Power BI / Active-Dispatch Convention (excluding 'Incomplete' rides from status share):
      * Success Rate = 65.96% (93,000 / 141,000)
      * Driver Cancellation Rate = 19.15% (27,000 / 141,000)
      * Customer Cancellation Rate = 7.45% (10,500 / 141,000)
      * No Driver Found Rate = 7.45% (10,500 / 141,000)
    - All-Rows Convention (across all 150,000 rows):
      * Success Rate = 62.00% (93,000 / 150,000)
      * Total Cancellation Rate (Customer + Driver) = 25.00% (37,500 / 150,000)
        - Driver Cancellation Rate = 18.00% (27,000 / 150,000)
        - Customer Cancellation Rate = 7.00% (10,500 / 150,000)
    """
    working_df = df.drop_duplicates(subset=["Booking ID"]) if deduplicate_booking_id else df

    total_rows = len(working_df)
    unique_bookings = int(working_df["Booking ID"].nunique()) if total_rows > 0 else 0

    status_counts = working_df["Booking Status"].value_counts()
    completed_rides = int(status_counts.get("Completed", 0))
    cancelled_driver = int(status_counts.get("Cancelled by Driver", 0))
    cancelled_customer = int(status_counts.get("Cancelled by Customer", 0))
    no_driver_found = int(status_counts.get("No Driver Found", 0))
    incomplete_rides = int(status_counts.get("Incomplete", 0))
    total_cancelled = cancelled_driver + cancelled_customer

    # Denominator excluding Incomplete (matches Power BI Overall Status Pie Chart: 141,000 rows)
    denom_excl_incomplete = total_rows - incomplete_rides

    # Power BI / Baseline Success & Cancellation Rates (excluding Incomplete)
    success_rate_baseline = (
        (completed_rides / denom_excl_incomplete * 100.0) if denom_excl_incomplete > 0 else 0.0
    )
    driver_cancel_rate_baseline = (
        (cancelled_driver / denom_excl_incomplete * 100.0) if denom_excl_incomplete > 0 else 0.0
    )
    customer_cancel_rate_baseline = (
        (cancelled_customer / denom_excl_incomplete * 100.0) if denom_excl_incomplete > 0 else 0.0
    )
    no_driver_rate_baseline = (
        (no_driver_found / denom_excl_incomplete * 100.0) if denom_excl_incomplete > 0 else 0.0
    )

    # All-Bookings Rates (including Incomplete in denominator: 150,000 rows)
    success_rate_all = (completed_rides / total_rows * 100.0) if total_rows > 0 else 0.0
    total_cancel_rate_all = (total_cancelled / total_rows * 100.0) if total_rows > 0 else 0.0
    driver_cancel_rate_all = (cancelled_driver / total_rows * 100.0) if total_rows > 0 else 0.0
    customer_cancel_rate_all = (cancelled_customer / total_rows * 100.0) if total_rows > 0 else 0.0

    # Revenue & Distance metrics
    total_revenue = float(working_df["Booking Value"].sum(skipna=True))
    completed_df = working_df[working_df["Booking Status"] == "Completed"]
    completed_revenue = float(completed_df["Booking Value"].sum(skipna=True))
    avg_booking_value = float(working_df["Booking Value"].mean(skipna=True)) if total_rows > 0 else 0.0

    total_distance = float(working_df["Ride Distance"].sum(skipna=True))
    completed_distance = float(completed_df["Ride Distance"].sum(skipna=True))
    avg_distance = float(working_df["Ride Distance"].mean(skipna=True)) if total_rows > 0 else 0.0

    # Ratings & Turnaround Time metrics
    avg_customer_rating = (
        float(working_df["Customer Rating"].mean(skipna=True))
        if working_df["Customer Rating"].notna().any()
        else 0.0
    )
    avg_driver_rating = (
        float(working_df["Driver Ratings"].mean(skipna=True))
        if working_df["Driver Ratings"].notna().any()
        else 0.0
    )
    avg_vtat = (
        float(working_df["Avg VTAT"].mean(skipna=True))
        if working_df["Avg VTAT"].notna().any()
        else 0.0
    )
    avg_ctat = (
        float(working_df["Avg CTAT"].mean(skipna=True))
        if working_df["Avg CTAT"].notna().any()
        else 0.0
    )

    return {
        "total_bookings": total_rows,
        "unique_bookings": unique_bookings,
        "completed_rides": completed_rides,
        "cancelled_driver": cancelled_driver,
        "cancelled_customer": cancelled_customer,
        "total_cancelled": total_cancelled,
        "no_driver_found": no_driver_found,
        "incomplete_rides": incomplete_rides,
        "success_rate_baseline": success_rate_baseline,
        "driver_cancel_rate_baseline": driver_cancel_rate_baseline,
        "customer_cancel_rate_baseline": customer_cancel_rate_baseline,
        "no_driver_rate_baseline": no_driver_rate_baseline,
        "success_rate_all": success_rate_all,
        "total_cancel_rate_all": total_cancel_rate_all,
        "driver_cancel_rate_all": driver_cancel_rate_all,
        "customer_cancel_rate_all": customer_cancel_rate_all,
        "total_revenue": total_revenue,
        "completed_revenue": completed_revenue,
        "avg_booking_value": avg_booking_value,
        "total_distance": total_distance,
        "completed_distance": completed_distance,
        "avg_distance": avg_distance,
        "avg_customer_rating": avg_customer_rating,
        "avg_driver_rating": avg_driver_rating,
        "avg_vtat": avg_vtat,
        "avg_ctat": avg_ctat,
    }


def compute_distance_value_statistics(
    df: pd.DataFrame, completed_only: bool = False
) -> Dict[str, float]:
    """
    Use SciPy (`scipy.stats.pearsonr` and `scipy.stats.linregress`) to compute
    the Pearson correlation coefficient, p-value, and OLS linear trendline
    parameters between `Ride Distance (km)` and `Booking Value (INR)`.
    """
    target_df = df[df["Booking Status"] == "Completed"] if completed_only else df
    valid = target_df.dropna(subset=["Ride Distance", "Booking Value"])

    n_samples = len(valid)
    if n_samples < 3:
        return {
            "pearson_r": 0.0,
            "p_value": 1.0,
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "n_samples": n_samples,
        }

    x = valid["Ride Distance"].to_numpy(dtype=float)
    y = valid["Booking Value"].to_numpy(dtype=float)

    pearson_r, p_value = stats.pearsonr(x, y)
    reg = stats.linregress(x, y)

    return {
        "pearson_r": float(pearson_r),
        "p_value": float(p_value),
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "r_squared": float(pearson_r ** 2),
        "n_samples": int(n_samples),
    }