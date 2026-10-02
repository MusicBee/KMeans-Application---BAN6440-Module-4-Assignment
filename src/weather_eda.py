import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = r"C:\Users\Kayode Ogunyemi\PycharmProjects\Module_4"

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs"
)

EDA_DIR = os.path.join(
    OUTPUT_DIR,
    "eda"
)

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

EDA_SUMMARY_FILE = os.path.join(
    EDA_DIR,
    "eda_summary.csv"
)

STATION_EXTREMES_FILE = os.path.join(
    EDA_DIR,
    "station_extremes.csv"
)

TEMPERATURE_EXTREMES_FILE = os.path.join(
    EDA_DIR,
    "temperature_extremes.csv"
)

PRECIPITATION_EXTREMES_FILE = os.path.join(
    EDA_DIR,
    "precipitation_extremes.csv"
)


# =========================================================
# CREATE EDA DIRECTORY
# =========================================================

os.makedirs(
    EDA_DIR,
    exist_ok=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def save_plot(filename):
    """
    Save the current matplotlib figure.
    """

    filepath = os.path.join(
        EDA_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  Saved: {filepath}"
    )


def print_section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =========================================================
# MAIN
# =========================================================

def main():

    print_section(
        "TEXAS WEATHER EXPLORATORY DATA ANALYSIS"
    )

    # =====================================================
    # 1. VALIDATE INPUT FILES
    # =====================================================

    required_files = [
        ANALYSIS_FILE,
        STATION_SUMMARY_FILE,
        MONTHLY_SUMMARY_FILE,
        YEARLY_SUMMARY_FILE
    ]

    for filepath in required_files:

        if not os.path.exists(filepath):

            raise FileNotFoundError(
                f"Required file not found:\n{filepath}"
            )

    print(
        "All required analysis files found."
    )

    # =====================================================
    # 2. LOAD DATA
    # =====================================================

    print_section(
        "LOADING DATA"
    )

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
        f"Analysis records : "
        f"{len(analysis):,}"
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
    # 3. DESCRIPTIVE STATISTICS
    # =====================================================

    print_section(
        "DESCRIPTIVE STATISTICS"
    )

    weather_statistics = (
        analysis[
            [
                "tmax",
                "tmin",
                "prcp"
            ]
        ]
        .describe()
        .T
    )

    weather_statistics[
        "missing"
    ] = (
        analysis[
            [
                "tmax",
                "tmin",
                "prcp"
            ]
        ]
        .isna()
        .sum()
    )

    weather_statistics[
        "missing_pct"
    ] = (
        weather_statistics["missing"]
        /
        len(analysis)
        *
        100
    )

    print(
        weather_statistics.to_string()
    )

    weather_statistics.to_csv(
        os.path.join(
            EDA_DIR,
            "weather_descriptive_statistics.csv"
        )
    )

    # =====================================================
    # 4. ANNUAL TEMPERATURE TREND
    # =====================================================

    print_section(
        "ANNUAL TEMPERATURE TREND"
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        yearly_summary["year"],
        yearly_summary["tmax_mean"],
        marker="o",
        linewidth=2,
        label="Average TMAX"
    )

    plt.plot(
        yearly_summary["year"],
        yearly_summary["tmin_mean"],
        marker="o",
        linewidth=2,
        label="Average TMIN"
    )

    plt.title(
        "Average Annual Temperature - Texas 50-Station Sample"
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.xticks(
        yearly_summary["year"]
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    save_plot(
        "01_annual_temperature_trend.png"
    )

    # =====================================================
    # 5. ANNUAL PRECIPITATION
    # =====================================================

    print_section(
        "ANNUAL PRECIPITATION"
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        yearly_summary["year"].astype(str),
        yearly_summary["prcp_total"]
    )

    plt.title(
        "Annual Precipitation - Texas 50-Station Sample"
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Total Precipitation (mm)"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    save_plot(
        "02_annual_precipitation.png"
    )

    # =====================================================
    # 6. MONTHLY TEMPERATURE PATTERN
    # =====================================================

    print_section(
        "MONTHLY TEMPERATURE PATTERN"
    )

    monthly_climate = (
        monthly_summary
        .groupby(
            [
                "month",
                "month_name"
            ],
            as_index=False
        )
        .agg(
            avg_tmax=(
                "tmax_mean",
                "mean"
            ),
            avg_tmin=(
                "tmin_mean",
                "mean"
            ),
            avg_prcp=(
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

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        monthly_climate["month_name"],
        monthly_climate["avg_tmax"],
        marker="o",
        linewidth=2,
        label="Average TMAX"
    )

    plt.plot(
        monthly_climate["month_name"],
        monthly_climate["avg_tmin"],
        marker="o",
        linewidth=2,
        label="Average TMIN"
    )

    plt.title(
        "Average Monthly Temperature Pattern"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    save_plot(
        "03_monthly_temperature_pattern.png"
    )

    # =====================================================
    # 7. MONTHLY PRECIPITATION
    # =====================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        monthly_climate["month_name"],
        monthly_climate["avg_prcp"]
    )

    plt.title(
        "Average Monthly Precipitation Pattern"
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
        "04_monthly_precipitation_pattern.png"
    )

    # =====================================================
    # 8. STATION AVERAGE TMAX
    # =====================================================

    station_sorted = (
        station_summary
        .sort_values(
            "tmax_mean",
            ascending=False
        )
    )

    plt.figure(
        figsize=(13, 8)
    )

    plt.barh(
        station_sorted[
            "station_name"
        ],
        station_sorted[
            "tmax_mean"
        ]
    )

    plt.title(
        "Average TMAX by Station"
    )

    plt.xlabel(
        "Average TMAX (°C)"
    )

    plt.ylabel(
        "Station"
    )

    plt.gca().invert_yaxis()

    save_plot(
        "05_station_average_tmax.png"
    )

    # =====================================================
    # 9. STATION AVERAGE TMIN
    # =====================================================

    station_sorted_tmin = (
        station_summary
        .sort_values(
            "tmin_mean",
            ascending=False
        )
    )

    plt.figure(
        figsize=(13, 8)
    )

    plt.barh(
        station_sorted_tmin[
            "station_name"
        ],
        station_sorted_tmin[
            "tmin_mean"
        ]
    )

    plt.title(
        "Average TMIN by Station"
    )

    plt.xlabel(
        "Average TMIN (°C)"
    )

    plt.ylabel(
        "Station"
    )

    plt.gca().invert_yaxis()

    save_plot(
        "06_station_average_tmin.png"
    )

    # =====================================================
    # 10. STATION PRECIPITATION
    # =====================================================

    station_sorted_prcp = (
        station_summary
        .sort_values(
            "prcp_total",
            ascending=False
        )
    )

    plt.figure(
        figsize=(13, 8)
    )

    plt.barh(
        station_sorted_prcp[
            "station_name"
        ],
        station_sorted_prcp[
            "prcp_total"
        ]
    )

    plt.title(
        "Total Precipitation by Station - 2020 to 2024"
    )

    plt.xlabel(
        "Total Precipitation (mm)"
    )

    plt.ylabel(
        "Station"
    )

    plt.gca().invert_yaxis()

    save_plot(
        "07_station_total_precipitation.png"
    )

    # =====================================================
    # 11. TMAX VS TMIN
    # =====================================================

    print_section(
        "TEMPERATURE RELATIONSHIP"
    )

    correlation_data = (
        analysis[
            [
                "tmax",
                "tmin"
            ]
        ]
        .dropna()
    )

    temperature_correlation = (
        correlation_data[
            "tmax"
        ]
        .corr(
            correlation_data[
                "tmin"
            ]
        )
    )

    print(
        f"TMAX/TMIN correlation: "
        f"{temperature_correlation:.4f}"
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        correlation_data["tmin"],
        correlation_data["tmax"],
        alpha=0.15,
        s=8
    )

    plt.title(
        "Relationship Between Daily TMIN and TMAX"
    )

    plt.xlabel(
        "TMIN (°C)"
    )

    plt.ylabel(
        "TMAX (°C)"
    )

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "08_tmin_vs_tmax.png"
    )

    # =====================================================
    # 12. TEMPERATURE DISTRIBUTION
    # =====================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        analysis["tmax"].dropna(),
        bins=40,
        alpha=0.7,
        label="TMAX"
    )

    plt.hist(
        analysis["tmin"].dropna(),
        bins=40,
        alpha=0.7,
        label="TMIN"
    )

    plt.title(
        "Distribution of Daily Temperatures"
    )

    plt.xlabel(
        "Temperature (°C)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "09_temperature_distribution.png"
    )

    # =====================================================
    # 13. PRECIPITATION DISTRIBUTION
    # =====================================================

    plt.figure(
        figsize=(10, 6)
    )

    precipitation = (
        analysis["prcp"]
        .dropna()
    )

    precipitation = (
        precipitation[
            precipitation >= 0
        ]
    )

    plt.hist(
        precipitation,
        bins=50,
        alpha=0.8
    )

    plt.title(
        "Distribution of Daily Precipitation"
    )

    plt.xlabel(
        "Precipitation (mm)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "10_precipitation_distribution.png"
    )

    # =====================================================
    # 14. EXTREME TEMPERATURES
    # =====================================================

    print_section(
        "EXTREME TEMPERATURE ANALYSIS"
    )

    hottest_days = (
        analysis
        .nlargest(
            20,
            "tmax"
        )[
            [
                "station_id",
                "station_name",
                "date",
                "tmax",
                "tmin"
            ]
        ]
    )

    coldest_days = (
        analysis
        .nsmallest(
            20,
            "tmin"
        )[
            [
                "station_id",
                "station_name",
                "date",
                "tmax",
                "tmin"
            ]
        ]
    )

    temperature_extremes = pd.concat(
        [
            hottest_days.assign(
                extreme_type="Highest TMAX"
            ),
            coldest_days.assign(
                extreme_type="Lowest TMIN"
            )
        ],
        ignore_index=True
    )

    temperature_extremes.to_csv(
        TEMPERATURE_EXTREMES_FILE,
        index=False
    )

    print(
        "\nHighest recorded TMAX:"
    )

    print(
        hottest_days.head(10).to_string(
            index=False
        )
    )

    print(
        "\nLowest recorded TMIN:"
    )

    print(
        coldest_days.head(10).to_string(
            index=False
        )
    )

    # =====================================================
    # 15. EXTREME PRECIPITATION
    # =====================================================

    print_section(
        "EXTREME PRECIPITATION"
    )

    wettest_days = (
        analysis
        .nlargest(
            20,
            "prcp"
        )[
            [
                "station_id",
                "station_name",
                "date",
                "prcp"
            ]
        ]
    )

    wettest_days.to_csv(
        PRECIPITATION_EXTREMES_FILE,
        index=False
    )

    print(
        "Highest precipitation observations:"
    )

    print(
        wettest_days.head(10).to_string(
            index=False
        )
    )

    # =====================================================
    # 16. STATION CLIMATE EXTREMES
    # =====================================================

    station_extremes = station_summary[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "tmax_mean",
            "tmin_mean",
            "prcp_total",
            "prcp_max",
            "temperature_range_mean"
        ]
    ].copy()

    station_extremes[
        "tmax_rank"
    ] = (
        station_extremes[
            "tmax_mean"
        ]
        .rank(
            ascending=False,
            method="min"
        )
    )

    station_extremes[
        "tmin_rank"
    ] = (
        station_extremes[
            "tmin_mean"
        ]
        .rank(
            ascending=False,
            method="min"
        )
    )

    station_extremes[
        "prcp_rank"
    ] = (
        station_extremes[
            "prcp_total"
        ]
        .rank(
            ascending=False,
            method="min"
        )
    )

    station_extremes.to_csv(
        STATION_EXTREMES_FILE,
        index=False
    )

    # =====================================================
    # 17. GEOGRAPHIC STATION DISTRIBUTION
    # =====================================================

    print_section(
        "GEOGRAPHIC STATION DISTRIBUTION"
    )

    plt.figure(
        figsize=(10, 8)
    )

    scatter = plt.scatter(
        station_summary["longitude"],
        station_summary["latitude"],
        s=80,
        alpha=0.8
    )

    for _, row in station_summary.iterrows():

        plt.annotate(
            row["station_id"],
            (
                row["longitude"],
                row["latitude"]
            ),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points"
        )

    plt.title(
        "Geographic Distribution of Selected Texas Stations"
    )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "11_station_geographic_distribution.png"
    )

    # =====================================================
    # 18. ANNUAL TEMPERATURE RANGE
    # =====================================================

    yearly_summary[
        "temperature_range"
    ] = (
        yearly_summary["tmax_mean"]
        -
        yearly_summary["tmin_mean"]
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        yearly_summary["year"],
        yearly_summary["temperature_range"],
        marker="o",
        linewidth=2
    )

    plt.title(
        "Annual Average Temperature Range"
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "TMAX - TMIN (°C)"
    )

    plt.xticks(
        yearly_summary["year"]
    )

    plt.grid(
        alpha=0.3
    )

    save_plot(
        "12_annual_temperature_range.png"
    )

    # =====================================================
    # 19. YEAR-TO-YEAR CHANGE
    # =====================================================

    yearly_change = yearly_summary[
        [
            "year",
            "tmax_mean",
            "tmin_mean",
            "prcp_total"
        ]
    ].copy()

    yearly_change[
        "tmax_change"
    ] = (
        yearly_change[
            "tmax_mean"
        ].diff()
    )

    yearly_change[
        "tmin_change"
    ] = (
        yearly_change[
            "tmin_mean"
        ].diff()
    )

    yearly_change[
        "prcp_change"
    ] = (
        yearly_change[
            "prcp_total"
        ].diff()
    )

    yearly_change[
        "prcp_change_pct"
    ] = (
        yearly_change[
            "prcp_total"
        ]
        .pct_change()
        *
        100
    )

    yearly_change.to_csv(
        os.path.join(
            EDA_DIR,
            "yearly_changes.csv"
        ),
        index=False
    )

    # =====================================================
    # 20. MONTHLY CLIMATE SUMMARY
    # =====================================================

    monthly_climate.to_csv(
        os.path.join(
            EDA_DIR,
            "monthly_climate_summary.csv"
        ),
        index=False
    )

    # =====================================================
    # 21. CREATE EDA SUMMARY
    # =====================================================

    hottest_year = (
        yearly_summary.loc[
            yearly_summary["tmax_mean"].idxmax(),
            "year"
        ]
    )

    coldest_year = (
        yearly_summary.loc[
            yearly_summary["tmin_mean"].idxmin(),
            "year"
        ]
    )

    wettest_year = (
        yearly_summary.loc[
            yearly_summary["prcp_total"].idxmax(),
            "year"
        ]
    )

    driest_year = (
        yearly_summary.loc[
            yearly_summary["prcp_total"].idxmin(),
            "year"
        ]
    )

    hottest_station = (
        station_summary.loc[
            station_summary["tmax_mean"].idxmax(),
            "station_name"
        ]
    )

    coldest_station = (
        station_summary.loc[
            station_summary["tmin_mean"].idxmin(),
            "station_name"
        ]
    )

    wettest_station = (
        station_summary.loc[
            station_summary["prcp_total"].idxmax(),
            "station_name"
        ]
    )

    driest_station = (
        station_summary.loc[
            station_summary["prcp_total"].idxmin(),
            "station_name"
        ]
    )

    summary_rows = [

        (
            "Number of stations",
            analysis["station_id"].nunique()
        ),

        (
            "Number of observations",
            len(analysis)
        ),

        (
            "Study start",
            analysis["date"].min().date()
        ),

        (
            "Study end",
            analysis["date"].max().date()
        ),

        (
            "Average TMAX",
            analysis["tmax"].mean()
        ),

        (
            "Average TMIN",
            analysis["tmin"].mean()
        ),

        (
            "Average PRCP",
            analysis["prcp"].mean()
        ),

        (
            "TMAX/TMIN correlation",
            temperature_correlation
        ),

        (
            "Hottest average TMAX year",
            hottest_year
        ),

        (
            "Lowest average TMIN year",
            coldest_year
        ),

        (
            "Wettest year",
            wettest_year
        ),

        (
            "Driest year",
            driest_year
        ),

        (
            "Highest average TMAX station",
            hottest_station
        ),

        (
            "Lowest average TMIN station",
            coldest_station
        ),

        (
            "Highest total precipitation station",
            wettest_station
        ),

        (
            "Lowest total precipitation station",
            driest_station
        )
    ]

    eda_summary = pd.DataFrame(
        summary_rows,
        columns=[
            "metric",
            "value"
        ]
    )

    eda_summary.to_csv(
        EDA_SUMMARY_FILE,
        index=False
    )

    # =====================================================
    # 22. PRINT KEY FINDINGS
    # =====================================================

    print_section(
        "KEY EDA RESULTS"
    )

    print(
        f"Hottest average TMAX year : "
        f"{hottest_year}"
    )

    print(
        f"Lowest average TMIN year  : "
        f"{coldest_year}"
    )

    print(
        f"Wettest year               : "
        f"{wettest_year}"
    )

    print(
        f"Driest year                : "
        f"{driest_year}"
    )

    print(
        f"Highest average TMAX station: "
        f"{hottest_station}"
    )

    print(
        f"Lowest average TMIN station : "
        f"{coldest_station}"
    )

    print(
        f"Wettest station             : "
        f"{wettest_station}"
    )

    print(
        f"Driest station              : "
        f"{driest_station}"
    )

    print(
        f"TMAX/TMIN correlation       : "
        f"{temperature_correlation:.4f}"
    )

    # =====================================================
    # 23. OUTPUT FILES
    # =====================================================

    print_section(
        "EDA COMPLETED SUCCESSFULLY"
    )

    print(
        f"EDA output directory:\n"
        f"{EDA_DIR}"
    )

    print()
    print(
        "Generated visualizations:"
    )

    png_files = sorted(
        [
            filename
            for filename in os.listdir(
                EDA_DIR
            )
            if filename.lower().endswith(".png")
        ]
    )

    for filename in png_files:

        print(
            f"  - {filename}"
        )

    print()
    print(
        "Generated analysis files:"
    )

    csv_files = sorted(
        [
            filename
            for filename in os.listdir(
                EDA_DIR
            )
            if filename.lower().endswith(".csv")
        ]
    )

    for filename in csv_files:

        print(
            f"  - {filename}"
        )

    print()
    print("=" * 80)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()