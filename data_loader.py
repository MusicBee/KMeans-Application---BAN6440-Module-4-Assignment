from datetime import datetime
import requests
import pandas as pd

## Import Config.Py Variables
from config import (START_DATE,END_DATE,COUNTRY_CODE,ELEMENTS,STATIONS_URL,INVENTORY_URL,STATION_DATA_URL)


def download_text(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


## Downloads the station metadata and extract Nigerian Stations Only
def load_station_metadata():

    ## Download Text from AWS
    text = download_text(STATIONS_URL)
    records = []

    for line in text.splitlines():
        if not line.startswith(COUNTRY_CODE):
            continue

        # Station Metadata
        records.append({"station_id": line[0:11].strip(),"latitude": float(line[12:20]),"longitude": float(line[21:30]), "elevation": float(line[31:37]),"state": line[38:40].strip(),"station_name": line[41:71].strip()})

    return pd.DataFrame(records)





## Downloads the Station Inventory
def load_inventory():

    text = download_text(INVENTORY_URL)
    records = []

    for line in text.splitlines():
        station_id = line[0:11].strip()
        if not station_id.startswith(COUNTRY_CODE):
            continue

        element = line[31:35].strip()
        if element not in ELEMENTS:
            continue

        # Station Inventory
        records.append({"station_id": station_id,"element": element,"start_year": int(line[36:40]),"end_year": int(line[41:45])})

    return pd.DataFrame(records)






def identify_candidate_stations(inventory):

    required_elements = set(ELEMENTS)

    station_elements = (
        inventory
        .groupby("station_id")["element"]
        .apply(set)
        .reset_index()
    )

    candidates = station_elements[
        station_elements["element"].apply(
            lambda x: required_elements.issubset(x)
        )
    ]

    return candidates["station_id"].tolist()







def parse_station_file(text):

    records = []

    ## Dataset Period
    start_date = datetime.strptime(START_DATE,"%Y-%m-%d").date()
    end_date = datetime.strptime(END_DATE,"%Y-%m-%d").date()

    for line in text.splitlines():

        if len(line) < 21:
            continue

        station_id = line[0:11].strip()
        year = int(line[11:15])
        month = int(line[15:17])
        element = line[17:21].strip()

        if element not in ELEMENTS:
            continue

        for day in range(1, 32):

            offset = 21 + ((day - 1) * 8)
            value_block = line[offset:offset + 8]

            if len(value_block) < 8:
                continue

            try:
                observation_date = datetime(year,month,day).date()
            except ValueError:
                continue

            if not (start_date <= observation_date <= end_date):
                continue

            value = int(value_block[0:5])

            if value == -9999:
                value = None

            records.append({"station_id": station_id,"date": observation_date,"element": element,"value": value,"mflag": value_block[5],"qflag": value_block[6],"sflag": value_block[7]})

    return pd.DataFrame(records)








def download_station_data(station_ids):

    results = []
    for station_id in station_ids:
        url = STATION_DATA_URL.format(station_id=station_id)
        response = requests.get(url,timeout=60)

        if response.status_code != 200:
            print(f"Unable to download {station_id}: "f"HTTP {response.status_code}")
            continue

        station_data = parse_station_file(response.text)

        if not station_data.empty:
            results.append(station_data)

    if not results:
        return pd.DataFrame()

    return pd.concat(results,ignore_index=True)

