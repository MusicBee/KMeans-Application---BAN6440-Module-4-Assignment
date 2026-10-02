from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests


# Import Configurations
from config import (START_DATE,END_DATE,ELEMENTS,COMPLETENESS_THRESHOLD,STATION_DATA_URL)



# PROJECT PATHS - I declared all my *.csv dataset outputs
PROJECT_PATH = Path(r"C:\Users\Kayode Ogunyemi\PycharmProjects\Module_4")
OUTPUT_DIR = PROJECT_PATH / "outputs"
CANDIDATE_FILE = (OUTPUT_DIR / "candidate_stations.csv")
RAW_DATA_FILE = (OUTPUT_DIR / "texas_station_data_2020_2024.csv")
QUALITY_FILE = (OUTPUT_DIR / "station_data_quality.csv")
QUALIFIED_FILE = (OUTPUT_DIR / "qualified_stations.csv")
LOW_COMPLETENESS_FILE = (OUTPUT_DIR / "low_completeness_stations.csv")
PARTIAL_DATA_FILE = (OUTPUT_DIR / "partial_data_stations.csv")
NO_DATA_FILE = (OUTPUT_DIR / "no_data_stations.csv")
DOWNLOAD_ERRORS_FILE = (OUTPUT_DIR / "station_download_errors.csv")



# DOWNLOAD SETTINGS - Setting Worker and Timeout
MAX_WORKERS = 12
REQUEST_TIMEOUT = 60



# STUDY PERIOD
start_date = pd.Timestamp(START_DATE)
end_date = pd.Timestamp(END_DATE)
expected_days = len(pd.date_range(start=start_date,end=end_date,freq="D"))



# START - Initiate Data Quality Checks
print("=" * 70)
print("TEXAS STATION DATA QUALITY ASSESSMENT")
print("=" * 70)

print(f"\nStudy period: {START_DATE} to {END_DATE}")
print(f"Expected days: {expected_days}")
print(f"Required elements: {', '.join(ELEMENTS)}")
print(f"Completeness threshold: "f"{COMPLETENESS_THRESHOLD:.0%}")
print(f"Download workers: {MAX_WORKERS}")





# LOAD CANDIDATE STATIONS
print("\nLoading candidate stations...")

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Candidate station file not found: "
        f"{CANDIDATE_FILE}"
    )

candidate_stations = pd.read_csv(CANDIDATE_FILE)
station_ids = (
    candidate_stations["station_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

print(f"Candidate Texas stations loaded: "f"{len(station_ids)}")


# PARSE GHCN .DLY FILE
def parse_station_file(
    text,
    station_id
):

    records = []

    for line in text.splitlines():

        if len(line) < 21:
            continue

        try:

            year = int(line[11:15])
            month = int(line[15:17])

        except ValueError:
            continue

        element = line[17:21].strip()

        if element not in ELEMENTS:
            continue

        for day in range(1, 32):

            offset = 21 + (
                (day - 1) * 8
            )

            value_block = (
                line[offset:offset + 8]
            )

            if len(value_block) < 8:
                continue

            try:

                observation_date = pd.Timestamp(
                    year=year,
                    month=month,
                    day=day
                )

            except ValueError:
                continue

            if not (
                start_date
                <= observation_date
                <= end_date
            ):
                continue

            try:
                value = int(
                    value_block[0:5]
                )

            except ValueError:
                value = None

            if value == -9999:
                value = None

            records.append(
                {
                    "station_id": station_id,
                    "date": observation_date,
                    "element": element,
                    "value": value,
                    "mflag": value_block[5],
                    "qflag": value_block[6],
                    "sflag": value_block[7]
                }
            )

    return records







# DOWNLOAD ONE STATION
def download_one_station(
    station_id
):

    url = STATION_DATA_URL.format(
        station_id=station_id
    )

    try:

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        records = parse_station_file(
            response.text,
            station_id
        )

        return {
            "station_id": station_id,
            "records": records,
            "error": None
        }

    except Exception as error:

        return {
            "station_id": station_id,
            "records": [],
            "error": str(error)
        }





# PARALLEL DOWNLOAD - Initiate Multiple Download
print("\nDownloading station data...")
print("This may take several minutes for "f"{len(station_ids)} stations.")

all_records = []
download_errors = []

completed = 0

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(
            download_one_station,
            station_id
        ): station_id
        for station_id in station_ids
    }

    for future in as_completed(futures):

        result = future.result()
        completed += 1
        station_id = result["station_id"]

        if result["error"]:

            download_errors.append(
                {
                    "station_id": station_id,
                    "error": result["error"]
                }
            )

        else:

            all_records.extend(
                result["records"]
            )

        # Progress every 25 stations
        if (
            completed % 25 == 0
            or completed == len(station_ids)
        ):

            print(
                f"Progress: "
                f"{completed:,}/"
                f"{len(station_ids):,} stations "
                f"({completed / len(station_ids) * 100:.1f}%)"
            )









# DOWNLOAD SUMMARY - For Data Completeness, This summary was put in place to understand the summary of this download
print("\nDownload completed.")
print(f"Successful station requests: "f"{len(station_ids) - len(download_errors):,}")
print(f"Failed station requests: "f"{len(download_errors):,}")
print(f"Total downloaded observations: "f"{len(all_records):,}")





# SAVE DOWNLOAD ERRORS
if download_errors:

    pd.DataFrame(
        download_errors
    ).to_csv(
        DOWNLOAD_ERRORS_FILE,
        index=False
    )

    print(
        f"Download errors saved to: "
        f"{DOWNLOAD_ERRORS_FILE}"
    )





# CREATE RAW DATAFRAME
if all_records:
    station_data = pd.DataFrame(all_records)

else:
    station_data = pd.DataFrame(columns=["station_id","date","element","value","mflag","qflag","sflag"])


# DATA TYPES
if not station_data.empty:
    station_data["date"] = pd.to_datetime(station_data["date"])

    station_data["value"] = pd.to_numeric(station_data["value"],errors="coerce")




# SAVE RAW DATA
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
station_data.to_csv(RAW_DATA_FILE,index=False)
print(f"\nRaw data saved to:"f"\n{RAW_DATA_FILE}")






# QUALITY ASSESSMENT
print("\nCalculating station completeness...")
quality_records = []

for station_id in station_ids:

    station_subset = station_data[station_data["station_id"] == station_id]
    station_record = {"station_id": station_id}
    total_valid_observations = 0
    total_quality_flags = 0

    element_completeness = []
    element_valid_counts = []


    # EACH REQUIRED ELEMENT
    for element in ELEMENTS:

        element_data = station_subset[
            station_subset["element"] == element
        ].copy()

        valid_data = element_data[
            element_data["value"].notna()
        ]

        # Count unique dates with valid observations
        valid_observations = (
            valid_data["date"]
            .nunique()
        )

        valid_observations = min(
            valid_observations,
            expected_days
        )

        missing_observations = (
            expected_days
            - valid_observations
        )

        completeness = (
            valid_observations
            / expected_days
        )

        quality_flags = (
            element_data["qflag"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        total_valid_observations += (
            valid_observations
        )

        total_quality_flags += (
            quality_flags
        )

        element_completeness.append(
            completeness
        )

        element_valid_counts.append(
            valid_observations
        )

        station_record[
            f"{element}_valid_observations"
        ] = valid_observations

        station_record[
            f"{element}_missing_observations"
        ] = missing_observations

        station_record[
            f"{element}_completeness_pct"
        ] = round(
            completeness * 100,
            2
        )

        station_record[
            f"{element}_quality_flags"
        ] = int(
            quality_flags
        )



    # OVERALL COMPLETENESS
    station_record[
        "overall_completeness_pct"
    ] = round(
        (
            sum(element_completeness)
            / len(element_completeness)
        ) * 100,
        2
    )

    station_record[
        "minimum_element_completeness_pct"
    ] = round(
        min(element_completeness) * 100,
        2
    )

    station_record[
        "total_valid_observations"
    ] = total_valid_observations

    station_record[
        "total_quality_flags"
    ] = int(
        total_quality_flags
    )



    # DATA CLASSIFICATION
    if total_valid_observations == 0:

        status = (
            "NO_DATA_IN_STUDY_PERIOD"
        )

    elif any(
        count == 0
        for count in element_valid_counts
    ):

        status = "PARTIAL_DATA"

    elif all(
        completeness
        >= COMPLETENESS_THRESHOLD
        for completeness
        in element_completeness
    ):

        status = "QUALIFIED"

    else:

        status = "LOW_COMPLETENESS"


    station_record["status"] = status

    quality_records.append(
        station_record
    )



# QUALITY DATAFRAME
quality_df = pd.DataFrame(quality_records)



# MERGE STATION METADATA
metadata_columns = ["station_id","station_name","latitude","longitude","elevation","state"]
available_metadata_columns = [column for column in metadata_columns if column in candidate_stations.columns]

quality_df = quality_df.merge(
    candidate_stations[
        available_metadata_columns
    ],
    on="station_id",
    how="left"
)



# SORT
status_order = {
    "QUALIFIED": 1,
    "LOW_COMPLETENESS": 2,
    "PARTIAL_DATA": 3,
    "NO_DATA_IN_STUDY_PERIOD": 4
}

quality_df["status_order"] = (
    quality_df["status"]
    .map(status_order)
)

quality_df = (
    quality_df
    .sort_values(
        [
            "status_order",
            "minimum_element_completeness_pct",
            "overall_completeness_pct"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .drop(
        columns=["status_order"]
    )
)


# SAVE QUALITY REPORT
quality_df.to_csv(
    QUALITY_FILE,
    index=False
)

print(
    f"\nQuality report saved to:"
    f"\n{QUALITY_FILE}"
)




# STATUS FILES
qualified_df = quality_df[
    quality_df["status"] == "QUALIFIED"
].copy()

low_completeness_df = quality_df[
    quality_df["status"] == "LOW_COMPLETENESS"
].copy()

partial_data_df = quality_df[
    quality_df["status"] == "PARTIAL_DATA"
].copy()

no_data_df = quality_df[
    quality_df["status"] ==
    "NO_DATA_IN_STUDY_PERIOD"
].copy()


qualified_df.to_csv(
    QUALIFIED_FILE,
    index=False
)

low_completeness_df.to_csv(
    LOW_COMPLETENESS_FILE,
    index=False
)

partial_data_df.to_csv(
    PARTIAL_DATA_FILE,
    index=False
)

no_data_df.to_csv(
    NO_DATA_FILE,
    index=False
)




# SUMMARY
print("\n" + "=" * 70)
print("TEXAS DATA QUALITY SUMMARY")
print("=" * 70)

print(f"\nTotal candidate stations: "f"{len(quality_df)}")
print(f"Qualified stations: "f"{len(qualified_df)}")
print(f"Low completeness stations: "f"{len(low_completeness_df)}")
print(f"Partial-data stations: "f"{len(partial_data_df)}")
print(f"No-data stations: "f"{len(no_data_df)}")


# STATUS BREAKDOWN
print("\nStatus breakdown:")

for status, count in (
    quality_df["status"]
    .value_counts()
    .items()
):

    print(
        f"  {status}: {count}"
    )


# TOP STATIONS
print("\n" + "=" * 70)
print("TOP 25 STATIONS BY DATA COMPLETENESS")
print("=" * 70)

top_columns = ["station_id","station_name","TMAX_completeness_pct","TMIN_completeness_pct","PRCP_completeness_pct","minimum_element_completeness_pct","overall_completeness_pct","status"]

print(
    quality_df[
        top_columns
    ]
    .head(25)
    .to_string(
        index=False
    )
)



# OUTPUT FILES
print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"\nRaw data:"
    f"\n  {RAW_DATA_FILE}"
)

print(
    f"\nQuality report:"
    f"\n  {QUALITY_FILE}"
)

print(
    f"\nQualified stations:"
    f"\n  {QUALIFIED_FILE}"
)

print(
    f"\nLow completeness:"
    f"\n  {LOW_COMPLETENESS_FILE}"
)

print(
    f"\nPartial data:"
    f"\n  {PARTIAL_DATA_FILE}"
)

print(
    f"\nNo data:"
    f"\n  {NO_DATA_FILE}"
)

if download_errors:

    print(
        f"\nDownload errors:"
        f"\n  {DOWNLOAD_ERRORS_FILE}"
    )


print(
    "\nTexas data quality assessment completed successfully."
)