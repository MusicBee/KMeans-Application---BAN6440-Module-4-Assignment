# IMPORT REQUIRED LIBRARIES
# datetime is used to convert and validate the analysis period.
# requests is used to download NOAA GHCN-D files from the web.
# pandas is used to store, transform, clean, and analyze the
# downloaded weather observations.
from datetime import datetime
import requests
import pandas as pd



# IMPORT PROJECT CONFIGURATION - In this block, i will import the dataset configuration from config.py.
# Keeping URLs, dates, geographic scope, and required elements
# in a separate configuration file makes the application easier
# to maintain and prevents duplication across scripts.
from .config import (START_DATE,END_DATE,COUNTRY_CODE,ELEMENTS,STATIONS_URL,INVENTORY_URL,STATION_DATA_URL)



# DOWNLOAD TEXT FROM NOAA GHCN-D - Sample Download a text file from a specified NOAA GHCN-D URL.
# A timeout is used to prevent the application from waiting indefinitely for an unavailable or slow connection.
# I configured raise_for_status() ensures that HTTP errors are detected rather than silently processing an unsuccessful response.
def download_text(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text



# LOAD GHCN-D STATION METADATA - This block is used to download and parse the GHCN-D station metadata file.
# The metadata contains information such as: Station ID,Latitude,Longitude,Elevation,State,Station name
def load_station_metadata():

    # Download the station metadata text file from NOAA.
    text = download_text(STATIONS_URL)
    records = []

    # Process the fixed-width station metadata file line by line.
    for line in text.splitlines():

        # Ignore stations outside the configured country.
        if not line.startswith(COUNTRY_CODE):
            continue

        # Extract station information using the fixed-width
        # column positions defined by the GHCN-D station format.
        records.append({
            "station_id": line[0:11].strip(),
            "latitude": float(line[12:20]),
            "longitude": float(line[21:30]),
            "elevation": float(line[31:37]),
            "state": line[38:40].strip(),
            "station_name": line[41:71].strip()
        })

    # Return the station metadata as a pandas DataFrame.
    return pd.DataFrame(records)



# LOAD GHCN-D STATION INVENTORY - Download and parse the GHCN-D inventory file to identify which weather elements are available
# for each station and the years for which observations exist.
def load_inventory():

    # Download the inventory file from NOAA.
    text = download_text(INVENTORY_URL)

    records = []

    # Process the fixed-width inventory file line by line.
    for line in text.splitlines():

        # Extract the station identifier.
        station_id = line[0:11].strip()

        # Ignore stations outside the configured country.
        if not station_id.startswith(COUNTRY_CODE):
            continue

        # Extract the weather element from the inventory record.
        element = line[31:35].strip()

        # Ignore weather elements that are not required
        # for this project.
        if element not in ELEMENTS:
            continue

        # Store the station, weather element, and coverage period.
        records.append({
            "station_id": station_id,
            "element": element,
            "start_year": int(line[36:40]),
            "end_year": int(line[41:45])
        })

    # Return the filtered inventory as a DataFrame.
    return pd.DataFrame(records)



# IDENTIFY CANDIDATE STATIONS - this function is used to identify stations that contain all required weather elements.
# A station must have TMAX, TMIN, and PRCP available in the inventory before it can proceed to the detailed data-quality assessment.
def identify_candidate_stations(inventory):

    # Convert the required element list into a set so that
    # set inclusion can be used to check station coverage.
    required_elements = set(ELEMENTS)

    # Group inventory records by station and create a set
    # containing the weather elements available at each station.
    station_elements = (
        inventory
        .groupby("station_id")["element"]
        .apply(set)
        .reset_index()
    )

    # Retain only stations whose available elements include
    # every required element.
    candidates = station_elements[
        station_elements["element"].apply(
            lambda x: required_elements.issubset(x)
        )
    ]

    # Return the qualifying station IDs as a Python list.
    return candidates["station_id"].tolist()






# PARSE INDIVIDUAL GHCN-D STATION FILE - Parse a GHCN-D .dly station file and convert its fixed-width records into structured daily observations.
# The function also restricts observations to the project's configured study period and required weather elements.
def parse_station_file(text):

    records = []

    # Convert the configured start and end dates from strings
    # into Python date objects for reliable date comparison.
    start_date = datetime.strptime(
        START_DATE,
        "%Y-%m-%d"
    ).date()

    end_date = datetime.strptime(
        END_DATE,
        "%Y-%m-%d"
    ).date()

    # Process each fixed-width record in the station file.
    for line in text.splitlines():

        # Ignore malformed or incomplete records.
        if len(line) < 21:
            continue

        # Extract the station ID, year, month, and weather element.
        station_id = line[0:11].strip()
        year = int(line[11:15])
        month = int(line[15:17])
        element = line[17:21].strip()

        # Ignore weather elements outside the project scope.
        if element not in ELEMENTS:
            continue

        # GHCN-D stores up to 31 daily observations per record.
        for day in range(1, 32):

            # Calculate the starting position of the daily
            # observation within the fixed-width record.
            offset = 21 + ((day - 1) * 8)

            value_block = line[offset:offset + 8]

            # Ignore incomplete daily observation blocks.
            if len(value_block) < 8:
                continue

            # Construct a valid calendar date. Invalid dates,
            # such as February 30, are skipped.
            try:
                observation_date = datetime(
                    year,
                    month,
                    day
                ).date()
            except ValueError:
                continue

            # Restrict observations to the configured study period.
            if not (
                start_date <= observation_date <= end_date
            ):
                continue

            # Extract the five-character observation value.
            value = int(value_block[0:5])

            # GHCN-D uses -9999 to represent a missing value.
            if value == -9999:
                value = None

            # Store the observation together with its measurement
            # and quality-control flags.
            records.append({
                "station_id": station_id,
                "date": observation_date,
                "element": element,
                "value": value,
                "mflag": value_block[5],
                "qflag": value_block[6],
                "sflag": value_block[7]
            })

    # Return all parsed observations as a DataFrame.
    return pd.DataFrame(records)






# DOWNLOAD DAILY DATA FOR SELECTED STATIONS - I included this function downloads and parse the daily GHCN-D data for each station
# supplied to the function. Stations that cannot be downloaded are reported and skipped so that one failed station does not terminate the entire acquisition process.
def download_station_data(station_ids):

    results = []

    # Process each candidate station individually.
    for station_id in station_ids:

        # Build the station-specific GHCN-D download URL.
        url = STATION_DATA_URL.format(
            station_id=station_id
        )

        # Download the station file.
        response = requests.get(
            url,
            timeout=60
        )

        # Handle unsuccessful HTTP responses.
        if response.status_code != 200:

            print(
                f"Unable to download {station_id}: "
                f"HTTP {response.status_code}"
            )

            continue

        # Parse the downloaded station file.
        station_data = parse_station_file(
            response.text
        )

        # Store non-empty station datasets for later combination.
        if not station_data.empty:
            results.append(station_data)

    # Return an empty DataFrame if no station data was retrieved.
    if not results:
        return pd.DataFrame()

    # Combine all successfully downloaded station datasets
    # into a single DataFrame.
    return pd.concat(
        results,
        ignore_index=True
    )



# CLEAN AND QUALITY-CHECK WEATHER OBSERVATIONS - To clean the downloaded observations before analysis.
# ===========
# The cleaning process:
# 1. I converted dates and values to appropriate data types.
# 2. Handled GHCN-D missing-value codes.
# 3. Removed duplicate station/date/element records.
# 4. Removed observations that contain quality-control flags.
# 5. Converted GHCN-D measurements into standard units.

def clean_observations(df):

    # Create a copy so that the original DataFrame is not modified.
    df = df.copy()

    # Convert observation dates to pandas datetime format.
    df["date"] = pd.to_datetime(df["date"])

    # Convert observation values to numeric values.
    # Invalid values are converted to NaN.
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    # Replace the GHCN-D missing-value code with a missing value.
    df.loc[df["value"] == -9999, "value"] = None

    # Remove duplicate observations for the same station,
    # date, and weather element.
    df = df.drop_duplicates(
        subset=[
            "station_id",
            "date",
            "element"
        ]
    )

    # Retain observations that do not contain a GHCN quality flag.
    df = df[
        df["qflag"].fillna("").eq("")
    ]

    # Convert GHCN-D temperature values from tenths of degrees
    # Celsius to degrees Celsius.
    df.loc[
        df["element"].isin(["TMAX", "TMIN"]),
        "value"
    ] /= 10

    # Convert GHCN-D precipitation values from tenths of millimeters
    # to millimeters.
    df.loc[
        df["element"] == "PRCP",
        "value"
    ] /= 10

    # Return the cleaned and standardized observations.
    return df






# CREATE STATION-LEVEL CLIMATE FEATURES - I transformed daily weather observations into station-level summary features for exploratory analysis and clustering.
# The resulting features describe temperature level,temperature variability, precipitation characteristics,and the typical temperature range at each station.
def create_station_features(df):

    # Create a copy so that the source DataFrame remains unchanged.
    df = df.copy()

    # Transform the long-format weather observations into a
    # station-day wide format with separate TMAX, TMIN, and PRCP
    # columns.
    wide = (
        df
        .pivot_table(
            index=["station_id", "date"],
            columns="element",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    # Calculate the daily temperature range as the difference
    # between maximum and minimum temperature.
    wide["temperature_range"] = (
        wide["TMAX"] - wide["TMIN"]
    )

    # Aggregate daily observations into station-level features.
    features = (
        wide
        .groupby("station_id")
        .agg(
            avg_tmax=("TMAX", "mean"),
            avg_tmin=("TMIN", "mean"),
            std_tmax=("TMAX", "std"),
            std_tmin=("TMIN", "std"),
            avg_prcp=("PRCP", "mean"),
            total_prcp=("PRCP", "sum"),
            rainy_days=(
                "PRCP",
                lambda x: (x > 0).sum()
            ),
            avg_temperature_range=(
                "temperature_range",
                "mean"
            )
        )
        .reset_index()
    )

    # Return the station-level feature dataset.
    return features