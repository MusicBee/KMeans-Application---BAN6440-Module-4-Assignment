from pathlib import Path

from data_loader import (
    load_station_metadata,
    load_inventory,
    identify_candidate_stations
)

from config import STATE_CODE


# PROJECT PATH
PROJECT_PATH = Path(r"C:\Users\Kayode Ogunyemi\PycharmProjects\Module_4")
OUTPUT_DIR = PROJECT_PATH / "outputs"
OUTPUT_FILE = (OUTPUT_DIR /"candidate_stations.csv")



# START
print("=" * 70)
print("TEXAS WEATHER STATION DISCOVERY")
print("=" * 70)


# LOAD STATION METADATA
print("\nLoading station metadata...")

try:
    stations = load_station_metadata()

except Exception as error:
    print(
        f"Error loading station metadata: {error}"
    )
    raise


print(
    f"Total United States Stations: "
    f"{len(stations)}"
)


# FILTER TO TEXAS
texas_stations = stations[
    stations["state"].str.upper() == STATE_CODE.upper()
].copy()

print(
    f"Texas stations found: "
    f"{len(texas_stations)}"
)




# LOAD INVENTORY
print("\nLoading station inventory...")

try:
    inventory = load_inventory()

except Exception as error:
    print(
        f"Error loading inventory: {error}"
    )
    raise


print(
    f"Total United States inventory records: "
    f"{len(inventory)}"
)



# IDENTIFY ALL CANDIDATE STATIONS
print("\nIdentifying candidate stations...")
candidate_station_ids = (identify_candidate_stations(inventory))
print(f"United States candidate station IDs: "f"{len(candidate_station_ids)}")



# FILTER CANDIDATES TO TEXAS
texas_station_ids = set(
    texas_stations["station_id"]
)

candidate_station_ids = set(
    candidate_station_ids
)

texas_candidate_ids = (
    candidate_station_ids
    .intersection(texas_station_ids)
)





# MATCH METADATA
candidate_stations = texas_stations[
    texas_stations["station_id"].isin(
        texas_candidate_ids
    )
].copy()


candidate_stations = (
    candidate_stations
    .sort_values(
        [
            "station_name",
            "station_id"
        ]
    )
    .reset_index(drop=True)
)



# CREATE OUTPUT DIRECTORY
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)



# SAVE CANDIDATE STATIONS
candidate_stations.to_csv(OUTPUT_FILE,index=False)
print(f"\nCandidate stations saved successfully:"f"\n{OUTPUT_FILE}")



# SUMMARY
print("\n" + "=" * 70)
print("STATION DISCOVERY SUMMARY")
print("=" * 70)

print(f"Total United States Stations:       "f"{len(stations)}")
print(
    f"Texas stations:                     "
    f"{len(texas_stations)}")

print(f"United States inventory records:    "f"{len(inventory)}")
print(f"United States candidate station IDs:"f" {len(candidate_station_ids)}")
print(f"Texas candidate station IDs:        "f"{len(texas_candidate_ids)}")
print(f"Matched Texas candidate stations:   "f"{len(candidate_stations)}")
print(f"Output file:                         "f"{OUTPUT_FILE}")
print("\nStation discovery completed.")