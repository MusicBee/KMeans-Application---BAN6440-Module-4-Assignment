import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = r"C:\Users\Kayode Ogunyemi\PycharmProjects\Module_4"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
STAT_DIR = os.path.join(OUTPUT_DIR, "statistical_analysis")

ANALYSIS_FILE = os.path.join(
    OUTPUT_DIR,
    "texas_50_station_analysis_2020_2024.csv"
)

STATION_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "texas_station_summary_2020_2024.csv"
)

MONTHLY_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "texas_monthly_summary_2020_2024.csv"
)

YEARLY_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "texas_yearly_summary_2020_2024.csv"
)

os.makedirs(STAT_DIR, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def save_plot(filename):
    filepath = os.path.join(STAT_DIR, filename)

    plt.tight_layout()
    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"  Saved: {filepath}")


def linear_trend(years, values):
    """
    Calculate a simple linear trend.

    Returns:
        slope, intercept
    """

    valid = pd.DataFrame({
        "year": years,
        "value": values
    }).dropna()

    if len(valid) < 2:
        return np.nan, np.nan

    slope, intercept = np.polyfit(
        valid["year"],
        valid["value"],
        1
    )

    return slope, intercept


def pct_change(first, last):

    if pd.isna(first) or first == 0:
        return np.nan

    return ((last - first) / first) * 100


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("TEXAS WEATHER STATISTICAL ANALYSIS")
    print("=" * 80)

    # =====================================================
    # 1. VALIDATE INPUT FILES
    # =====================================================

    print("\nChecking required input files...")

    required_files = [
        ANALYSIS_FILE,
        STATION_SUMMARY_FILE,
        MONTHLY_SUMMARY_FILE,
        YEARLY_SUMMARY_FILE
    ]

    for filepath in required_files:

        if not os.path.exists(filepath):

            raise FileNotFoundError(
                f"\nRequired file not found:\n{filepath}"
            )

        print(f"  Found: {os.path.basename(filepath)}")

    # =====================================================
    # 2. LOAD DATA
    # =====================================================

    print("\nLoading analysis datasets...")

    analysis = pd.read_csv(
        ANALYSIS_FILE
    )

    station_summary = pd.read_csv(
        STATION_SUMMARY_FILE
    )

    monthly_summary = pd.read_csv(
        MONTHLY_SUMMARY_FILE
    )

    yearly_summary = pd.read_csv(
        YEARLY_SUMMARY_FILE
    )

    analysis["date"] = pd.to_datetime(
        analysis["date"],
        errors="coerce"
    )

    print(
        f"Analysis records : {len(analysis):,}"
    )

    print(
        f"Stations          : "
        f"{analysis['station_id'].nunique()}"
    )

    print(
        f"Date range        : "
        f"{analysis['date'].min().date()} "
        f"to "
        f"{analysis['date'].max().date()}"
    )

    # =====================================================
    # 3. YEAR-OVER-YEAR ANALYSIS
    # =====================================================

    print("\n" + "=" * 80)
    print("YEAR-OVER-YEAR ANALYSIS")
    print("=" * 80)

    yearly = yearly_summary.copy()

    yearly = yearly.sort_values(
        "year"
    ).reset_index(drop=True)

    yearly["tmax_yoy_change"] = (
        yearly["tmax_mean"].diff()
    )

    yearly["tmin_yoy_change"] = (
        yearly["tmin_mean"].diff()
    )

    yearly["prcp_yoy_change"] = (
        yearly["prcp_total"].diff()
    )

    yearly["tmax_yoy_pct"] = (
        yearly["tmax_mean"].pct_change() * 100
    )

    yearly["tmin_yoy_pct"] = (
        yearly["tmin_mean"].pct_change() * 100
    )

    yearly["prcp_yoy_pct"] = (
        yearly["prcp_total"].pct_change() * 100
    )

    yearly["temperature_range"] = (
        yearly["tmax_mean"]
        - yearly["tmin_mean"]
    )

    yearly["temperature_range_change"] = (
        yearly["temperature_range"].diff()
    )

    yearly.to_csv(
        os.path.join(
            STAT_DIR,
            "year_over_year_analysis.csv"
        ),
        index=False
    )

    print(
        yearly[
            [
                "year",
                "tmax_mean",
                "tmax_yoy_change",
                "tmax_yoy_pct",
                "tmin_mean",
                "tmin_yoy_change",
                "tmin_yoy_pct",
                "prcp_total",
                "prcp_yoy_change",
                "prcp_yoy_pct"
            ]
        ].to_string(index=False)
    )

    # =====================================================
    # 4. 2020-2024 CHANGE
    # =====================================================

    print("\n" + "=" * 80)
    print("2020-2024 CHANGE")
    print("=" * 80)

    first_year = yearly.iloc[0]
    last_year = yearly.iloc[-1]

    tmax_change = (
        last_year["tmax_mean"]
        - first_year["tmax_mean"]
    )

    tmin_change = (
        last_year["tmin_mean"]
        - first_year["tmin_mean"]
    )

    prcp_change = (
        last_year["prcp_total"]
        - first_year["prcp_total"]
    )

    tmax_change_pct = pct_change(
        first_year["tmax_mean"],
        last_year["tmax_mean"]
    )

    tmin_change_pct = pct_change(
        first_year["tmin_mean"],
        last_year["tmin_mean"]
    )

    prcp_change_pct = pct_change(
        first_year["prcp_total"],
        last_year["prcp_total"]
    )

    period_change = pd.DataFrame(
        [
            [
                "TMAX",
                first_year["tmax_mean"],
                last_year["tmax_mean"],
                tmax_change,
                tmax_change_pct
            ],
            [
                "TMIN",
                first_year["tmin_mean"],
                last_year["tmin_mean"],
                tmin_change,
                tmin_change_pct
            ],
            [
                "PRCP",
                first_year["prcp_total"],
                last_year["prcp_total"],
                prcp_change,
                prcp_change_pct
            ]
        ],
        columns=[
            "metric",
            "2020_value",
            "2024_value",
            "absolute_change",
            "percentage_change"
        ]
    )

    period_change.to_csv(
        os.path.join(
            STAT_DIR,
            "2020_2024_change.csv"
        ),
        index=False
    )

    print(
        period_change.to_string(
            index=False
        )
    )

    # =====================================================
    # 5. LINEAR TREND ANALYSIS
    # =====================================================

    print("\n" + "=" * 80)
    print("LINEAR TREND ANALYSIS")
    print("=" * 80)

    tmax_slope, tmax_intercept = linear_trend(
        yearly["year"],
        yearly["tmax_mean"]
    )

    tmin_slope, tmin_intercept = linear_trend(
        yearly["year"],
        yearly["tmin_mean"]
    )

    prcp_slope, prcp_intercept = linear_trend(
        yearly["year"],
        yearly["prcp_total"]
    )

    trend_results = pd.DataFrame(
        [
            [
                "Average TMAX",
                tmax_slope,
                "degrees C per year"
            ],
            [
                "Average TMIN",
                tmin_slope,
                "degrees C per year"
            ],
            [
                "Total PRCP",
                prcp_slope,
                "mm per year"
            ]
        ],
        columns=[
            "metric",
            "linear_trend_slope",
            "unit_per_year"
        ]
    )

    trend_results.to_csv(
        os.path.join(
            STAT_DIR,
            "linear_trends.csv"
        ),
        index=False
    )

    print(
        trend_results.to_string(
            index=False
        )
    )

    # =====================================================
    # 6. SEASONAL ANALYSIS
    # =====================================================

    print("\n" + "=" * 80)
    print("SEASONAL ANALYSIS")
    print("=" * 80)

    seasonal = (
        monthly_summary
        .groupby(
            [
                "month",
                "month_name"
            ],
            as_index=False
        )
        .agg(
            average_tmax=(
                "tmax_mean",
                "mean"
            ),
            average_tmin=(
                "tmin_mean",
                "mean"
            ),
            average_prcp=(
                "prcp_mean",
                "mean"
            ),
            total_prcp=(
                "prcp_total",
                "sum"
            )
        )
        .sort_values(
            "month"
        )
    )

    seasonal["temperature_range"] = (
        seasonal["average_tmax"]
        - seasonal["average_tmin"]
    )

    seasonal.to_csv(
        os.path.join(
            STAT_DIR,
            "seasonal_analysis.csv"
        ),
        index=False
    )

    hottest_month = seasonal.loc[
        seasonal["average_tmax"].idxmax()
    ]

    coldest_month = seasonal.loc[
        seasonal["average_tmin"].idxmin()
    ]

    wettest_month = seasonal.loc[
        seasonal["average_prcp"].idxmax()
    ]

    driest_month = seasonal.loc[
        seasonal["average_prcp"].idxmin()
    ]

    print(
        f"Hottest month : "
        f"{hottest_month['month_name']} "
        f"({hottest_month['average_tmax']:.2f} °C)"
    )

    print(
        f"Coldest month : "
        f"{coldest_month['month_name']} "
        f"({coldest_month['average_tmin']:.2f} °C)"
    )

    print(
        f"Wettest month : "
        f"{wettest_month['month_name']} "
        f"({wettest_month['average_prcp']:.2f} mm/day)"
    )

    print(
        f"Driest month  : "
        f"{driest_month['month_name']} "
        f"({driest_month['average_prcp']:.2f} mm/day)"
    )





    # 7. STATION VARIABILITY

    print("\n" + "=" * 80)
    print("STATION VARIABILITY")
    print("=" * 80)

    station = station_summary.copy()

    station["temperature_range"] = (
        station["tmax_mean"]
        - station["tmin_mean"]
    )

    station["tmax_rank"] = (
        station["tmax_mean"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    station["tmin_rank"] = (
        station["tmin_mean"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    station["prcp_rank"] = (
        station["prcp_total"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    station["tmax_zscore"] = (
        (
            station["tmax_mean"]
            - station["tmax_mean"].mean()
        )
        / station["tmax_mean"].std()
    )

    station["tmin_zscore"] = (
        (
            station["tmin_mean"]
            - station["tmin_mean"].mean()
        )
        / station["tmin_mean"].std()
    )

    station["prcp_zscore"] = (
        (
            station["prcp_total"]
            - station["prcp_total"].mean()
        )
        / station["prcp_total"].std()
    )

    station.to_csv(
        os.path.join(
            STAT_DIR,
            "station_variability_analysis.csv"
        ),
        index=False
    )

    print("\nTMAX station range:")

    print(
        f"  Minimum: "
        f"{station['tmax_mean'].min():.2f} °C"
    )

    print(
        f"  Maximum: "
        f"{station['tmax_mean'].max():.2f} °C"
    )

    print(
        f"  Range: "
        f"{station['tmax_mean'].max() - station['tmax_mean'].min():.2f} °C"
    )

    print("\nTMIN station range:")

    print(
        f"  Minimum: "
        f"{station['tmin_mean'].min():.2f} °C"
    )

    print(
        f"  Maximum: "
        f"{station['tmin_mean'].max():.2f} °C"
    )

    print(
        f"  Range: "
        f"{station['tmin_mean'].max() - station['tmin_mean'].min():.2f} °C"
    )

    print("\nPrecipitation station range:")

    print(
        f"  Minimum: "
        f"{station['prcp_total'].min():.2f} mm"
    )

    print(
        f"  Maximum: "
        f"{station['prcp_total'].max():.2f} mm"
    )

    print(
        f"  Range: "
        f"{station['prcp_total'].max() - station['prcp_total'].min():.2f} mm"
    )

    # =====================================================
    # 8. STATION TEMPERATURE RANGE CHART
    # =====================================================

    plt.figure(
        figsize=(13, 8)
    )

    station_sorted = station.sort_values(
        "tmax_mean"
    )

    plt.barh(
        station_sorted["station_name"],
        station_sorted["tmax_mean"]
        - station_sorted["tmin_mean"]
    )

    plt.title(
        "Average Temperature Range by Station"
    )

    plt.xlabel(
        "Average TMAX - TMIN (°C)"
    )

    plt.ylabel(
        "Station"
    )

    save_plot(
        "01_station_temperature_range.png"
    )

    # =====================================================
    # 9. YEAR-OVER-YEAR TMAX CHANGE
    # =====================================================

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        yearly["year"].astype(str),
        yearly["tmax_yoy_change"]
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Year-over-Year Change in Average TMAX"
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Change in TMAX (°C)"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "02_yoy_tmax_change.png"
    )

    # =====================================================
    # 10. YEAR-OVER-YEAR PRECIPITATION CHANGE
    # =====================================================

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        yearly["year"].astype(str),
        yearly["prcp_yoy_pct"]
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Year-over-Year Change in Total Precipitation"
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Change in Precipitation (%)"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "03_yoy_precipitation_change.png"
    )

    # =====================================================
    # 11. SEASONAL TEMPERATURE PROFILE
    # =====================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        seasonal["month_name"],
        seasonal["average_tmax"],
        marker="o",
        linewidth=2,
        label="TMAX"
    )

    plt.plot(
        seasonal["month_name"],
        seasonal["average_tmin"],
        marker="o",
        linewidth=2,
        label="TMIN"
    )

    plt.fill_between(
        range(len(seasonal)),
        seasonal["average_tmin"],
        seasonal["average_tmax"],
        alpha=0.15
    )

    plt.title(
        "Seasonal Temperature Profile"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.xticks(
        range(len(seasonal)),
        seasonal["month_name"],
        rotation=45
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    save_plot(
        "04_seasonal_temperature_profile.png"
    )

    # =====================================================
    # 12. SEASONAL PRECIPITATION PROFILE
    # =====================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        seasonal["month_name"],
        seasonal["average_prcp"]
    )

    plt.title(
        "Seasonal Precipitation Profile"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Average Daily Precipitation (mm)"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "05_seasonal_precipitation_profile.png"
    )

    # =====================================================
    # 13. TMAX/TMIN CORRELATION BY STATION
    # =====================================================

    print("\n" + "=" * 80)
    print("STATION TEMPERATURE CORRELATIONS")
    print("=" * 80)

    station_correlations = []

    for station_id, group in analysis.groupby(
        "station_id"
    ):

        valid = group[
            [
                "tmax",
                "tmin"
            ]
        ].dropna()

        if len(valid) >= 2:

            correlation = (
                valid["tmax"]
                .corr(valid["tmin"])
            )

        else:

            correlation = np.nan

        station_name = (
            group["station_name"]
            .iloc[0]
        )

        station_correlations.append(
            [
                station_id,
                station_name,
                len(valid),
                correlation
            ]
        )

    station_correlations = pd.DataFrame(
        station_correlations,
        columns=[
            "station_id",
            "station_name",
            "valid_observations",
            "tmax_tmin_correlation"
        ]
    )

    station_correlations.to_csv(
        os.path.join(
            STAT_DIR,
            "station_temperature_correlations.csv"
        ),
        index=False
    )

    print(
        station_correlations[
            [
                "station_name",
                "tmax_tmin_correlation"
            ]
        ]
        .sort_values(
            "tmax_tmin_correlation",
            ascending=False
        )
        .to_string(index=False)
    )

    # =====================================================
    # 14. EXTREME EVENT COUNTS
    # =====================================================

    print("\n" + "=" * 80)
    print("EXTREME EVENT ANALYSIS")
    print("=" * 80)

    extreme_events = []

    # Very hot days
    very_hot_days = (
        analysis["tmax"] > 40
    ).sum()

    # Cold nights
    cold_nights = (
        analysis["tmin"] < 0
    ).sum()

    # Wet days
    wet_days = (
        analysis["prcp"] >= 1
    ).sum()

    # Heavy rainfall
    heavy_rain_days = (
        analysis["prcp"] >= 25
    ).sum()

    # Extreme rainfall
    extreme_rain_days = (
        analysis["prcp"] >= 50
    ).sum()

    extreme_events.extend(
        [
            [
                "Very hot days",
                "tmax",
                40,
                ">",
                very_hot_days
            ],
            [
                "Cold nights",
                "tmin",
                0,
                "<",
                cold_nights
            ],
            [
                "Wet days",
                "prcp",
                1,
                ">=",
                wet_days
            ],
            [
                "Heavy rainfall days",
                "prcp",
                25,
                ">=",
                heavy_rain_days
            ],
            [
                "Extreme rainfall days",
                "prcp",
                50,
                ">=",
                extreme_rain_days
            ]
        ]
    )

    extreme_events_df = pd.DataFrame(
        extreme_events,
        columns=[
            "event",
            "variable",
            "threshold",
            "operator",
            "count"
        ]
    )

    extreme_events_df.to_csv(
        os.path.join(
            STAT_DIR,
            "extreme_event_counts.csv"
        ),
        index=False
    )

    print(
        extreme_events_df.to_string(
            index=False
        )
    )

    # =====================================================
    # 15. OVERALL TMAX/TMIN CORRELATION
    # =====================================================

    print("\n" + "=" * 80)
    print("OVERALL TEMPERATURE CORRELATION")
    print("=" * 80)

    valid_temperature = analysis[
        [
            "tmax",
            "tmin"
        ]
    ].dropna()

    overall_correlation = (
        valid_temperature["tmax"]
        .corr(valid_temperature["tmin"])
    )

    print(
        f"TMAX/TMIN correlation: "
        f"{overall_correlation:.4f}"
    )

    # =====================================================
    # 16. STATISTICAL FINDINGS
    # =====================================================

    print("\n" + "=" * 80)
    print("STATISTICAL FINDINGS")
    print("=" * 80)

    hottest_year = yearly.loc[
        yearly["tmax_mean"].idxmax()
    ]

    coldest_year = yearly.loc[
        yearly["tmin_mean"].idxmin()
    ]

    wettest_year = yearly.loc[
        yearly["prcp_total"].idxmax()
    ]

    driest_year = yearly.loc[
        yearly["prcp_total"].idxmin()
    ]

    hottest_station = station.loc[
        station["tmax_mean"].idxmax()
    ]

    coldest_station = station.loc[
        station["tmin_mean"].idxmin()
    ]

    wettest_station = station.loc[
        station["prcp_total"].idxmax()
    ]

    driest_station = station.loc[
        station["prcp_total"].idxmin()
    ]

    findings = [
        [
            "Hottest sample year by average TMAX",
            hottest_year["year"]
        ],
        [
            "Highest average TMAX",
            hottest_year["tmax_mean"]
        ],
        [
            "Lowest sample year by average TMIN",
            coldest_year["year"]
        ],
        [
            "Lowest average TMIN",
            coldest_year["tmin_mean"]
        ],
        [
            "Wettest sample year",
            wettest_year["year"]
        ],
        [
            "Highest aggregate precipitation",
            wettest_year["prcp_total"]
        ],
        [
            "Driest sample year",
            driest_year["year"]
        ],
        [
            "Lowest aggregate precipitation",
            driest_year["prcp_total"]
        ],
        [
            "Highest average TMAX station",
            hottest_station["station_name"]
        ],
        [
            "Highest average TMAX station value",
            hottest_station["tmax_mean"]
        ],
        [
            "Lowest average TMIN station",
            coldest_station["station_name"]
        ],
        [
            "Lowest average TMIN station value",
            coldest_station["tmin_mean"]
        ],
        [
            "Wettest station",
            wettest_station["station_name"]
        ],
        [
            "Wettest station precipitation",
            wettest_station["prcp_total"]
        ],
        [
            "Driest station",
            driest_station["station_name"]
        ],
        [
            "Driest station precipitation",
            driest_station["prcp_total"]
        ],
        [
            "Overall TMAX/TMIN correlation",
            overall_correlation
        ],
        [
            "TMAX linear trend",
            tmax_slope
        ],
        [
            "TMIN linear trend",
            tmin_slope
        ],
        [
            "PRCP linear trend",
            prcp_slope
        ],
        [
            "Hottest month",
            hottest_month["month_name"]
        ],
        [
            "Coldest month",
            coldest_month["month_name"]
        ],
        [
            "Wettest month",
            wettest_month["month_name"]
        ],
        [
            "Driest month",
            driest_month["month_name"]
        ],
        [
            "Very hot days (>40°C)",
            very_hot_days
        ],
        [
            "Cold nights (<0°C)",
            cold_nights
        ],
        [
            "Wet days (>=1 mm)",
            wet_days
        ],
        [
            "Heavy rainfall days (>=25 mm)",
            heavy_rain_days
        ],
        [
            "Extreme rainfall days (>=50 mm)",
            extreme_rain_days
        ]
    ]

    findings_df = pd.DataFrame(
        findings,
        columns=[
            "finding",
            "value"
        ]
    )

    findings_df.to_csv(
        os.path.join(
            STAT_DIR,
            "statistical_findings.csv"
        ),
        index=False
    )

    print(
        findings_df.to_string(
            index=False
        )
    )

    # =====================================================
    # 17. FINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETED")
    print("=" * 80)

    print(
        f"\nOutput directory:\n{STAT_DIR}"
    )

    print("\nGenerated files:")

    for filename in sorted(
        os.listdir(STAT_DIR)
    ):

        print(
            f"  - {filename}"
        )

    print("=" * 80)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()