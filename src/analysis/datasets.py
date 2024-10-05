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




from pathlib import Path
import xarray as xr
import pandas as pd
from datetime import datetime

def load_merra(data_path: str = "../data", longitude: float = 13.125, lattitude: float = 53.0, 
               start_date: str = None, end_date: str = None):
    # Define path
    p = Path(f"{data_path}/raw/merra2")
    
    # List all files matching the pattern
    fs = list(p.glob(f"merra2_{longitude}_{lattitude}_*.nc"))
    
    # Convert start_date and end_date to datetime objects if provided
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Function to extract the date from filename
    def extract_date(file_name):
        date_str = file_name.stem.split('_')[3:6]  # Get the date part (YYYYMMDD)
        date_str = "_".join(date_str)
        return datetime.strptime(date_str, '%Y_%m_%d')
    
    # Filter files by date
    if start_date or end_date:
        fs = [f for f in fs if (not start_date or extract_date(f) >= start_date) and 
                             (not end_date or extract_date(f) <= end_date)]
    
    # Load the dataset
    ds = xr.open_mfdataset(fs, engine='netcdf4')
    df = ds.to_dataframe()
    
    # Reset index and rename columns
    df = df.reset_index().set_index("time")
    df = df.rename(columns={
        'HLML': 'layer_height (m)',
        'PS': 'pressure (Pa)',
        'QLML': 'specific_humidity (1)',
        'SPEEDLML': 'wind_speed (m/s)',
        'TLML': 'air_temperature (K)',
    })

    return df