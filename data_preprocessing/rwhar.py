import pandas as pd
from io import BytesIO
import zipfile


def check_rwhar_zip(path) -> str:
    # verify that the path is to the zip containing csv and not another zip of csv

    if any(".zip" in filename for filename in zipfile.ZipFile(path, "r").namelist()):
        # There are multiple zips in some cases
        with zipfile.ZipFile(path, "r") as temp:
            path = BytesIO(temp.read(
                max(temp.namelist())))

    return path


def rwhar_load_csv(path) -> dict:
    path = check_rwhar_zip(path)
    pos_tables = {}
    with zipfile.ZipFile(path, 'r') as Zip:
        zip_files = Zip.namelist()

        for csv in zip_files:
            if 'csv' in csv:
                position = csv[csv.rfind('_') + 1:csv.rfind('.')]
                sensor = csv[:3]
                prefix = sensor.lower() + '_'
                table = pd.read_csv(Zip.open(csv))

                table.rename(columns={
                    "attr_x": prefix + "x",
                    "attr_y": prefix + "y",
                    "attr_z": prefix + "z",
                    "attr_time": "timestamp"
                }, inplace=True)
                table.drop(columns='id', inplace=True)
                pos_tables[position] = table

    return pos_tables


def rwhar_load_activity(path) -> pd.DataFrame:
    csv_per_pos = rwhar_load_csv(path)
    data = pd.DataFrame()

    for pos in csv_per_pos.keys():
        acc_tab = csv_per_pos[pos]

        acc_tab = pd.DataFrame(acc_tab)
        acc_tab["position"] = pos

        data = data.append(acc_tab)

    return data
