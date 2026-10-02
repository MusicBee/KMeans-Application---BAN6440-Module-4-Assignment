from pathlib import Path



# I have included this to resolve the absolute path of the project root directory.
# moves two levels upward to the main project directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the locations used to store raw data, processed data
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"



# DATASET ANALYSIS PERIOD - Define the five-year period used for the weather analysis which covers daily observations from January 1, 2020 through December 31, 2024.
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"



# GEOGRAPHIC SCOPE - Restrict the dataset to weather stations located in Texas,United States. The GHCN-D station and inventory identifiers
# use the ISO country code and state abbreviation.
COUNTRY_CODE = "US"
STATE_CODE = "TX"




# REQUIRED WEATHER ELEMENTS - This was used to define the weather variables required for this project: These variables form the core weather measurements used
# # throughout the data preparation and analysis stages.
# TMAX = daily maximum temperature
# TMIN = daily minimum temperature
# PRCP = daily precipitation

ELEMENTS = ["TMAX", "TMIN", "PRCP"]



# DATA COMPLETENESS THRESHOLD - I defined the minimum acceptable data completeness level for weather stations. Only stations meeting or exceeding 80% completeness for the required weather elements are considered
COMPLETENESS_THRESHOLD = 0.80



# NOAA GHCN-D STATION METADATA - I configured the URL containing the NOAA GHCN-D station metadata to provides information such as station ID,
# station name, latitude, longitude, elevation, country,and state.

STATIONS_URL = ("https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt")



# NOAA GHCN-D STATION INVENTORY - Configured URL containing the GHCN-D station inventory.The inventory identifies which weather elements are available
# at each station and the period over which observations exist. It is used to identify stations that support TMAX, TMIN, and PRCP during the required study period.
INVENTORY_URL = ("https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt")



# NOAA GHCN-D DAILY STATION DATA - Configured the URL to download the daily GHCN-D data for an individual weather station. {station_id} is replaced dynamically by the station identifier
# during the data-acquisition process which the .dly files contain daily observations for weather elements such as TMAX, TMIN, and PRCP.
STATION_DATA_URL = ("https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly")