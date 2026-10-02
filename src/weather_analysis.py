import os
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = r"C:\Users\Kayode Ogunyemi\PycharmProjects\Module_4"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

FINAL_STATIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "final_stations.csv"
)

RAW_DATA_FILE = os.path.join(
    OUTPUT_DIR,
    "texas_station_data_2020_2024.csv"
)

ANALYSIS_DATA_FILE = os.path.join(
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

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

EXPECTED_STATIONS = 50
EXPECTED_DAYS = 1827

REQUIRED_ELEMENTS = [
    "TMAX",
    "TMIN",
    "PRCP"
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def validate_file(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required file does not exist:\n{file_path}"
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
        "TEXAS 50-STATION WEATHER ANALYSIS"
    )

    # =====================================================
    # 1. VALIDATE INPUT FILES
    # =====================================================

    print("Validating input files...")

    validate_file(
        FINAL_STATIONS_FILE
    )

    validate_file(
        RAW_DATA_FILE
    )

    print(
        "Input files found successfully."
    )

    # =====================================================
    # 2. LOAD FINAL STATIONS
    # =====================================================

    print_section(
        "LOADING FINAL STATION SELECTION"
    )

    stations = pd.read_csv(
        FINAL_STATIONS_FILE
    )

    stations.columns = [
        str(column).strip().lower()
        for column in stations.columns
    ]

    required_station_columns = [
        "station_id",
        "station_name",
        "latitude",
        "longitude"
    ]

    missing_columns = [
        column
        for column in required_station_columns
        if column not in stations.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in final_stations.csv: "
            f"{missing_columns}"
        )

    stations["station_id"] = (
        stations["station_id"]
        .astype(str)
        .str.strip()
    )

    selected_station_ids = set(
        stations["station_id"]
    )

    print(
        f"Stations in final_stations.csv: "
        f"{len(stations):,}"
    )

    print(
        f"Unique selected stations: "
        f"{len(selected_station_ids):,}"
    )

    if len(selected_station_ids) != EXPECTED_STATIONS:

        raise ValueError(
            f"Expected {EXPECTED_STATIONS} unique stations "
            f"but found {len(selected_station_ids)}."
        )

    # =====================================================
    # 3. LOAD RAW DATA
    # =====================================================

    print_section(
        "LOADING RAW WEATHER DATA"
    )

    raw = pd.read_csv(
        RAW_DATA_FILE
    )

    raw.columns = [
        str(column).strip().lower()
        for column in raw.columns
    ]

    print(
        f"Raw observations loaded: "
        f"{len(raw):,}"
    )

    print()
    print("Raw columns:")

    print(
        ", ".join(raw.columns)
    )

    # =====================================================
    # 4. VALIDATE GHCN LONG FORMAT
    # =====================================================

    required_raw_columns = [
        "station_id",
        "date",
        "element",
        "value"
    ]

    missing_raw_columns = [
        column
        for column in required_raw_columns
        if column not in raw.columns
    ]

    if missing_raw_columns:

        raise ValueError(
            "Missing required raw data columns: "
            f"{missing_raw_columns}"
        )

    print()
    print(
        "Detected GHCN long-format weather data."
    )

    # =====================================================
    # 5. STANDARDIZE RAW DATA
    # =====================================================

    raw["station_id"] = (
        raw["station_id"]
        .astype(str)
        .str.strip()
    )

    raw["date"] = pd.to_datetime(
        raw["date"],
        errors="coerce"
    )

    raw["element"] = (
        raw["element"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    raw["value"] = pd.to_numeric(
        raw["value"],
        errors="coerce"
    )

    # =====================================================
    # 6. FILTER TO SELECTED STATIONS
    # =====================================================

    print_section(
        "FILTERING WEATHER OBSERVATIONS"
    )

    analysis = raw[
        raw["station_id"].isin(
            selected_station_ids
        )
    ].copy()

    print(
        f"Observations after station filter: "
        f"{len(analysis):,}"
    )

    # =====================================================
    # 7. FILTER STUDY PERIOD
    # =====================================================

    analysis = analysis[
        (analysis["date"] >= START_DATE)
        &
        (analysis["date"] <= END_DATE)
    ].copy()

    print(
        f"Observations after date filter: "
        f"{len(analysis):,}"
    )

    # =====================================================
    # 8. FILTER REQUIRED ELEMENTS
    # =====================================================

    analysis = analysis[
        analysis["element"].isin(
            REQUIRED_ELEMENTS
        )
    ].copy()

    print(
        f"Observations after element filter: "
        f"{len(analysis):,}"
    )

    print()
    print("Element distribution:")

    print(
        analysis["element"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # =====================================================
    # 9. HANDLE GHCN VALUES
    # =====================================================
    #
    # GHCN daily values are normally stored in tenths
    # of the reporting unit:
    #
    # TMAX/TMIN -> tenths of degrees Celsius
    # PRCP      -> tenths of millimetres
    #
    # Therefore divide by 10.
    #
    # =====================================================

    analysis["value"] = (
        analysis["value"] / 10.0
    )

    # =====================================================
    # 10. REMOVE DUPLICATE STATION/DATE/ELEMENT RECORDS
    # =====================================================

    duplicate_count = (
        analysis
        .duplicated(
            subset=[
                "station_id",
                "date",
                "element"
            ],
            keep=False
        )
        .sum()
    )

    print()
    print(
        f"Duplicate station/date/element records: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:

        print(
            "Keeping the first occurrence "
            "for each station/date/element."
        )

        analysis = (
            analysis
            .drop_duplicates(
                subset=[
                    "station_id",
                    "date",
                    "element"
                ],
                keep="first"
            )
        )

    # =====================================================
    # 11. PIVOT LONG DATA TO WIDE FORMAT
    # =====================================================

    print_section(
        "TRANSFORMING GHCN DATA TO ANALYTICAL FORMAT"
    )

    analysis = (
        analysis
        .pivot(
            index=[
                "station_id",
                "date"
            ],
            columns="element",
            values="value"
        )
        .reset_index()
    )

    analysis.columns.name = None

    # Ensure all required columns exist
    for element in REQUIRED_ELEMENTS:

        if element not in analysis.columns:

            analysis[element] = pd.NA

    analysis = analysis[
        [
            "station_id",
            "date",
            "TMAX",
            "TMIN",
            "PRCP"
        ]
    ]

    # Rename weather columns to lowercase
    analysis = analysis.rename(
        columns={
            "TMAX": "tmax",
            "TMIN": "tmin",
            "PRCP": "prcp"
        }
    )

    print(
        f"Daily station records after pivot: "
        f"{len(analysis):,}"
    )

    # =====================================================
    # 12. ATTACH STATION METADATA
    # =====================================================

    print(
        "\nAttaching station metadata..."
    )

    station_metadata = stations[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude"
        ]
    ].drop_duplicates(
        subset=["station_id"]
    )

    analysis = analysis.merge(
        station_metadata,
        on="station_id",
        how="left",
        validate="many_to_one"
    )

    # =====================================================
    # 13. ADD CALENDAR ATTRIBUTES
    # =====================================================

    analysis["year"] = (
        analysis["date"].dt.year
    )

    analysis["month"] = (
        analysis["date"].dt.month
    )

    analysis["month_name"] = (
        analysis["date"].dt.month_name()
    )

    analysis["day"] = (
        analysis["date"].dt.day
    )

    analysis["day_of_year"] = (
        analysis["date"].dt.dayofyear
    )

    analysis["day_of_week"] = (
        analysis["date"].dt.day_name()
    )

    # =====================================================
    # 14. REORDER COLUMNS
    # =====================================================

    analysis = analysis[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "date",
            "year",
            "month",
            "month_name",
            "day",
            "day_of_year",
            "day_of_week",
            "tmax",
            "tmin",
            "prcp"
        ]
    ]

    analysis = analysis.sort_values(
        [
            "station_id",
            "date"
        ]
    )

    # =====================================================
    # 15. SAVE ANALYSIS DATASET
    # =====================================================

    analysis.to_csv(
        ANALYSIS_DATA_FILE,
        index=False
    )

    print()
    print(
        f"Analysis dataset saved:\n"
        f"{ANALYSIS_DATA_FILE}"
    )

    # =====================================================
    # 16. VALIDATE STATION COVERAGE
    # =====================================================

    print_section(
        "ANALYSIS VALIDATION"
    )

    actual_station_count = (
        analysis["station_id"]
        .nunique()
    )

    print(
        f"Expected stations       : "
        f"{EXPECTED_STATIONS}"
    )

    print(
        f"Stations represented    : "
        f"{actual_station_count}"
    )

    missing_stations = (
        selected_station_ids
        -
        set(
            analysis["station_id"]
        )
    )

    if missing_stations:

        print()
        print(
            "Stations with no observations:"
        )

        for station_id in sorted(
            missing_stations
        ):
            print(
                f"  {station_id}"
            )

    else:

        print(
            "All 50 selected stations "
            "have observations."
        )

    # =====================================================
    # 17. STATION-LEVEL SUMMARY
    # =====================================================

    print_section(
        "CREATING STATION-LEVEL SUMMARY"
    )

    station_summary = (
        analysis
        .groupby(
            [
                "station_id",
                "station_name",
                "latitude",
                "longitude"
            ],
            dropna=False
        )
        .agg(
            observation_count=(
                "date",
                "count"
            ),

            first_date=(
                "date",
                "min"
            ),

            last_date=(
                "date",
                "max"
            ),

            tmax_valid_days=(
                "tmax",
                "count"
            ),

            tmax_mean=(
                "tmax",
                "mean"
            ),

            tmax_min=(
                "tmax",
                "min"
            ),

            tmax_max=(
                "tmax",
                "max"
            ),

            tmin_valid_days=(
                "tmin",
                "count"
            ),

            tmin_mean=(
                "tmin",
                "mean"
            ),

            tmin_min=(
                "tmin",
                "min"
            ),

            tmin_max=(
                "tmin",
                "max"
            ),

            prcp_valid_days=(
                "prcp",
                "count"
            ),

            prcp_mean=(
                "prcp",
                "mean"
            ),

            prcp_total=(
                "prcp",
                "sum"
            ),

            prcp_max=(
                "prcp",
                "max"
            )
        )
        .reset_index()
    )

    station_summary[
        "temperature_range_mean"
    ] = (
        station_summary["tmax_mean"]
        -
        station_summary["tmin_mean"]
    )

    station_summary[
        "tmax_completeness_pct"
    ] = (
        station_summary["tmax_valid_days"]
        /
        EXPECTED_DAYS
        *
        100
    )

    station_summary[
        "tmin_completeness_pct"
    ] = (
        station_summary["tmin_valid_days"]
        /
        EXPECTED_DAYS
        *
        100
    )

    station_summary[
        "prcp_completeness_pct"
    ] = (
        station_summary["prcp_valid_days"]
        /
        EXPECTED_DAYS
        *
        100
    )

    station_summary.to_csv(
        STATION_SUMMARY_FILE,
        index=False
    )

    print(
        f"Station summary saved:\n"
        f"{STATION_SUMMARY_FILE}"
    )

    # =====================================================
    # 18. MONTHLY SUMMARY
    # =====================================================

    print_section(
        "CREATING MONTHLY SUMMARY"
    )

    monthly_summary = (
        analysis
        .groupby(
            [
                "year",
                "month",
                "month_name"
            ],
            dropna=False
        )
        .agg(

            station_count=(
                "station_id",
                "nunique"
            ),

            observation_count=(
                "date",
                "count"
            ),

            tmax_mean=(
                "tmax",
                "mean"
            ),

            tmax_min=(
                "tmax",
                "min"
            ),

            tmax_max=(
                "tmax",
                "max"
            ),

            tmin_mean=(
                "tmin",
                "mean"
            ),

            tmin_min=(
                "tmin",
                "min"
            ),

            tmin_max=(
                "tmin",
                "max"
            ),

            prcp_mean=(
                "prcp",
                "mean"
            ),

            prcp_total=(
                "prcp",
                "sum"
            ),

            prcp_max=(
                "prcp",
                "max"
            )
        )
        .reset_index()
    )

    monthly_summary = monthly_summary.sort_values(
        [
            "year",
            "month"
        ]
    )

    monthly_summary.to_csv(
        MONTHLY_SUMMARY_FILE,
        index=False
    )

    print(
        f"Monthly summary saved:\n"
        f"{MONTHLY_SUMMARY_FILE}"
    )

    # =====================================================
    # 19. YEARLY SUMMARY
    # =====================================================

    print_section(
        "CREATING YEARLY SUMMARY"
    )

    yearly_summary = (
        analysis
        .groupby(
            "year",
            dropna=False
        )
        .agg(

            station_count=(
                "station_id",
                "nunique"
            ),

            observation_count=(
                "date",
                "count"
            ),

            tmax_mean=(
                "tmax",
                "mean"
            ),

            tmax_min=(
                "tmax",
                "min"
            ),

            tmax_max=(
                "tmax",
                "max"
            ),

            tmin_mean=(
                "tmin",
                "mean"
            ),

            tmin_min=(
                "tmin",
                "min"
            ),

            tmin_max=(
                "tmin",
                "max"
            ),

            prcp_mean=(
                "prcp",
                "mean"
            ),

            prcp_total=(
                "prcp",
                "sum"
            ),

            prcp_max=(
                "prcp",
                "max"
            )
        )
        .reset_index()
    )

    yearly_summary.to_csv(
        YEARLY_SUMMARY_FILE,
        index=False
    )

    print(
        f"Yearly summary saved:\n"
        f"{YEARLY_SUMMARY_FILE}"
    )

    # =====================================================
    # 20. MISSING VALUE CHECK
    # =====================================================

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    print(
        f"TMAX missing values : "
        f"{analysis['tmax'].isna().sum():,}"
    )

    print(
        f"TMIN missing values : "
        f"{analysis['tmin'].isna().sum():,}"
    )

    print(
        f"PRCP missing values : "
        f"{analysis['prcp'].isna().sum():,}"
    )

    # =====================================================
    # 21. COMPLETENESS CHECK
    # =====================================================

    print_section(
        "STATION COMPLETENESS"
    )

    print(
        f"Minimum TMAX completeness : "
        f"{station_summary['tmax_completeness_pct'].min():.2f}%"
    )

    print(
        f"Minimum TMIN completeness : "
        f"{station_summary['tmin_completeness_pct'].min():.2f}%"
    )

    print(
        f"Minimum PRCP completeness : "
        f"{station_summary['prcp_completeness_pct'].min():.2f}%"
    )

    print(
        f"Average TMAX completeness : "
        f"{station_summary['tmax_completeness_pct'].mean():.2f}%"
    )

    print(
        f"Average TMIN completeness : "
        f"{station_summary['tmin_completeness_pct'].mean():.2f}%"
    )

    print(
        f"Average PRCP completeness : "
        f"{station_summary['prcp_completeness_pct'].mean():.2f}%"
    )

    # =====================================================
    # 22. YEARLY RESULTS
    # =====================================================

    print_section(
        "YEARLY WEATHER SUMMARY"
    )

    print(
        yearly_summary.to_string(
            index=False
        )
    )

    # =====================================================
    # 23. FINAL OUTPUT
    # =====================================================

    print_section(
        "ANALYSIS COMPLETED SUCCESSFULLY"
    )

    print(
        f"Final analysis observations: "
        f"{len(analysis):,}"
    )

    print(
        f"Stations analysed: "
        f"{analysis['station_id'].nunique()}"
    )

    print(
        f"Study period: "
        f"{START_DATE} to {END_DATE}"
    )

    print()
    print("Generated files:")

    print(
        f"1. {ANALYSIS_DATA_FILE}"
    )

    print(
        f"2. {STATION_SUMMARY_FILE}"
    )

    print(
        f"3. {MONTHLY_SUMMARY_FILE}"
    )

    print(
        f"4. {YEARLY_SUMMARY_FILE}"
    )

    print()
    print("=" * 80)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()