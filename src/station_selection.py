import os
import math
import pandas as pd

PROJECT_DIR = r"C:\Users\Kayode Ogunyemi\PycharmProjects\Module_4"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

INPUT_FILE = os.path.join(OUTPUT_DIR, "station_data_quality.csv")
FINAL_FILE = os.path.join(OUTPUT_DIR, "final_stations.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "station_selection_report.csv")

TARGET_STATIONS = 50


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two coordinates in km."""
    radius_km = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius_km * math.asin(math.sqrt(a))


if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

print("Loading station quality results...")

quality_df = pd.read_csv(INPUT_FILE)

required_columns = [
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "status",
    "TMAX_completeness_pct",
    "TMIN_completeness_pct",
    "PRCP_completeness_pct",
    "overall_completeness_pct",
]

missing_columns = [
    col for col in required_columns
    if col not in quality_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print(f"Total stations in quality file: {len(quality_df):,}")




# 1. Keep only fully qualified stations
qualified = quality_df[
    quality_df["status"].astype(str).str.upper() == "QUALIFIED"
].copy()

print(f"Qualified stations available: {len(qualified):,}")

if len(qualified) < TARGET_STATIONS:
    raise ValueError(
        f"Only {len(qualified)} qualified stations are available. "
        f"{TARGET_STATIONS} are required."
    )




# 2. Convert numeric fields
numeric_columns = [
    "latitude",
    "longitude",
    "TMAX_completeness_pct",
    "TMIN_completeness_pct",
    "PRCP_completeness_pct",
    "overall_completeness_pct",
]

for column in numeric_columns:
    qualified[column] = pd.to_numeric(
        qualified[column],
        errors="coerce"
    )

qualified = qualified.dropna(
    subset=numeric_columns
).copy()




# 3. Calculate minimum completeness across all 3 elements
qualified["minimum_element_completeness_pct"] = qualified[
    [
        "TMAX_completeness_pct",
        "TMIN_completeness_pct",
        "PRCP_completeness_pct",
    ]
].min(axis=1)




# 4. Calculate geographic centre of qualified stations
center_lat = qualified["latitude"].mean()
center_lon = qualified["longitude"].mean()

print()
print("Qualified station network centre:")
print(f"  Latitude : {center_lat:.4f}")
print(f"  Longitude: {center_lon:.4f}")

qualified["distance_to_network_center_km"] = qualified.apply(
    lambda row: haversine(
        center_lat,
        center_lon,
        row["latitude"],
        row["longitude"],
    ),
    axis=1,
)






# 5. Select initial central station
start_idx = (
    qualified.sort_values(
        by=[
            "distance_to_network_center_km",
            "minimum_element_completeness_pct",
            "overall_completeness_pct",
            "station_id",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )
    .index[0]
)

selected_indices = [start_idx]

qualified["min_distance_to_selected_km"] = float("inf")





# 6. Greedy geographic farthest-point selection
print()
print("Selecting geographically distributed stations...")
for selection_number in range(2, TARGET_STATIONS + 1):

    selected_coordinates = qualified.loc[
        selected_indices,
        ["latitude", "longitude"],
    ].values

    candidate_indices = [
        idx
        for idx in qualified.index
        if idx not in selected_indices
    ]

    for idx in candidate_indices:

        candidate_lat = qualified.at[idx, "latitude"]
        candidate_lon = qualified.at[idx, "longitude"]

        minimum_distance = min(
            haversine(
                candidate_lat,
                candidate_lon,
                selected_lat,
                selected_lon,
            )
            for selected_lat, selected_lon in selected_coordinates
        )

        qualified.at[
            idx,
            "min_distance_to_selected_km"
        ] = minimum_distance

    next_idx = (
        qualified.loc[candidate_indices]
        .sort_values(
            by=[
                "min_distance_to_selected_km",
                "minimum_element_completeness_pct",
                "overall_completeness_pct",
                "station_id",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .index[0]
    )

    selected_indices.append(next_idx)

    if selection_number % 10 == 0:
        print(
            f"  Selected {selection_number}/{TARGET_STATIONS}"
        )









# 7. Build final station dataset
selected = qualified.loc[selected_indices].copy()

# Calculate distance to nearest other selected station
nearest_distances = []

for idx, row in selected.iterrows():

    other_stations = selected.drop(index=idx)

    distances = other_stations.apply(
        lambda other: haversine(
            row["latitude"],
            row["longitude"],
            other["latitude"],
            other["longitude"],
        ),
        axis=1,
    )

    nearest_distances.append(distances.min())

selected["nearest_selected_station_km"] = nearest_distances

selected["selection_rank"] = range(
    1,
    len(selected) + 1
)

selected["selection_reason"] = (
    "QUALIFIED station selected using deterministic "
    "geographic farthest-point sampling"
)






# 8. Select output columns
output_columns = ["selection_rank","station_id","station_name","latitude","longitude","elevation","TMAX_completeness_pct","TMIN_completeness_pct","PRCP_completeness_pct","minimum_element_completeness_pct","overall_completeness_pct","nearest_selected_station_km",
    "selection_reason","status",]

output_columns = [col for col in output_columns if col in selected.columns]

final_stations = (
    selected[
        output_columns
    ]
    .sort_values("selection_rank")
    .copy()
)





# 9. Save final station list
final_stations.to_csv(FINAL_FILE,index=False)




# 10. Create selection summary
report_rows = [
    (
        "Total qualified stations available",
        len(qualified),
    ),
    (
        "Target final sample",
        TARGET_STATIONS,
    ),
    (
        "Stations selected",
        len(final_stations),
    ),
    (
        "Average TMAX completeness (%)",
        round(
            final_stations[
                "TMAX_completeness_pct"
            ].mean(),
            2,
        ),
    ),
    (
        "Average TMIN completeness (%)",
        round(
            final_stations[
                "TMIN_completeness_pct"
            ].mean(),
            2,
        ),
    ),
    (
        "Average PRCP completeness (%)",
        round(
            final_stations[
                "PRCP_completeness_pct"
            ].mean(),
            2,
        ),
    ),
    (
        "Average overall completeness (%)",
        round(
            final_stations[
                "overall_completeness_pct"
            ].mean(),
            2,
        ),
    ),
    (
        "Minimum selected TMAX completeness (%)",
        round(
            final_stations[
                "TMAX_completeness_pct"
            ].min(),
            2,
        ),
    ),
    (
        "Minimum selected TMIN completeness (%)",
        round(
            final_stations[
                "TMIN_completeness_pct"
            ].min(),
            2,
        ),
    ),
    (
        "Minimum selected PRCP completeness (%)",
        round(
            final_stations[
                "PRCP_completeness_pct"
            ].min(),
            2,
        ),
    ),
    (
        "Minimum nearest-station distance (km)",
        round(
            final_stations[
                "nearest_selected_station_km"
            ].min(),
            2,
        ),
    ),
    (
        "Maximum nearest-station distance (km)",
        round(
            final_stations[
                "nearest_selected_station_km"
            ].max(),
            2,
        ),
    ),
]

report_df = pd.DataFrame(
    report_rows,
    columns=["metric", "value"],
)

report_df.to_csv(
    REPORT_FILE,
    index=False
)





# 11. Print final summary
print()
print("=" * 70)
print("TEXAS FINAL STATION SELECTION")
print("=" * 70)

print(
    f"Qualified stations available : "
    f"{len(qualified):,}"
)

print(
    f"Target stations              : "
    f"{TARGET_STATIONS}"
)

print(
    f"Stations selected            : "
    f"{len(final_stations)}"
)

print()
print("Completeness summary:")

print(
    f"  Average TMAX               : "
    f"{final_stations['TMAX_completeness_pct'].mean():.2f}%"
)

print(
    f"  Average TMIN               : "
    f"{final_stations['TMIN_completeness_pct'].mean():.2f}%"
)

print(
    f"  Average PRCP               : "
    f"{final_stations['PRCP_completeness_pct'].mean():.2f}%"
)

print(
    f"  Average Overall            : "
    f"{final_stations['overall_completeness_pct'].mean():.2f}%"
)

print()
print("Minimum selected completeness:")

print(
    f"  TMAX                       : "
    f"{final_stations['TMAX_completeness_pct'].min():.2f}%"
)

print(
    f"  TMIN                       : "
    f"{final_stations['TMIN_completeness_pct'].min():.2f}%"
)

print(
    f"  PRCP                       : "
    f"{final_stations['PRCP_completeness_pct'].min():.2f}%"
)

print()
print("Output files:")
print(f"  Final stations : {FINAL_FILE}")
print(f"  Report         : {REPORT_FILE}")

print("=" * 70)