import pandas as pd
import xarray as xr
from pathlib import Path


def load_electricity(data_path: str = "../data"):
    """
    source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
    """
    df = pd.read_csv(f"{data_path}/LD2011_2014.txt", sep=";", decimal=",", index_col=0)

    df.index = pd.to_datetime(df.index)

    return df

def _load_entsoe(file: Path):
    df = pd.read_csv(file)
    df = df.drop(columns="Area")
    df = df.rename(columns={
        "Wind Offshore  - Actual Aggregated [MW]": "offshore", 
        "Wind Onshore  - Actual Aggregated [MW]": "onshore",
        "MTU": "time"
        })
    df["offshore"] = df.offshore.replace("-", None).astype("float")
    df["onshore"] = df.onshore.replace("-", None).astype("float")
    df["time"] = pd.to_datetime(df.time.str.split(" -").str[0], dayfirst=True)
    df = df.set_index("time")

    return df

def load_entsoe(data_path: str = "../data"):

    files = list(Path(f"{data_path}/entsoe").glob("*"))
    dfs = [_load_entsoe(f) for f in files]
    df = pd.concat(dfs)

    return df




def load_merra(data_path: str = "../data", longitude: float = 13.125, lattitude: float = 53.0):
    p = Path(f"{data_path}/raw/merra2")
    fs = list(p.glob(f"merra2_{longitude}_{lattitude}_*.nc"))
    ds = xr.open_mfdataset(fs, engine='netcdf4')
    df = ds.to_dataframe()
    df = df.reset_index().set_index("time")
    return df