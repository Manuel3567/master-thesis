import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

# Dataset 1
def load_electricity(data_path: str = "../data"):
    """
    source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
    """
    df = pd.read_csv(f"{data_path}/LD2011_2014.txt", sep=";", decimal=",", index_col=0)

    df.index = pd.to_datetime(df.index)

    return df


# Dataset 2

def load_entsoe(data_path: str = "../data"):

    files = list(Path(f"{data_path}/entsoe").glob("*"))
    dfs = [_load_entsoe(f) for f in files]
    df = pd.concat(dfs)

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



from pathlib import Path
import xarray as xr
import pandas as pd
from datetime import datetime

# Dataset 3

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


# Load dataset 4
#def load_turbine_data():
#    turbine_data = load_turbine_electricity_data()
#    turbine_events = load_turbine_events()
#    merged_data = turbine_data.merge(turbine_events, how='left', left_index=True, right_index=True)
#    merged_data['Status'] = merged_data.Status.fillna("Running").astype("category")
#    
#    return merged_data


#def load_turbine_events(data_path: str = "../data"):
#    p = Path(f"{data_path}/zenodo_turbine_data/raw/2016/Status_Penmanshiel_09_2016-06-24_-_2017-01-01_1049.csv")
#    turbine_events = pd.read_csv(p,
#        skiprows=9,
#        quotechar='"',
#        encoding='utf-8-sig')
#    
#    turbine_events = turbine_events.rename(columns={'Timestamp start': 'start', 'Timestamp end': 'end'})
#    turbine_events = turbine_events.loc[:, ["Status", 'start', 'end', 'Message', 'Service contract category','IEC category']]
#    turbine_events['start'] = pd.to_datetime(turbine_events['start'])
#    turbine_events = turbine_events[turbine_events['end'] != "-"]
#    turbine_events['end'] = pd.to_datetime(turbine_events['end'])
#    turbine_events['start'] = turbine_events['start'].dt.floor('10min')
#    turbine_events['end'] = turbine_events['end'].dt.ceil('10min')
#    turbine_events['intervals'] = turbine_events.apply(lambda row: pd.date_range(row['start'], row['end'], freq='10min', inclusive='left'), axis=1)
#    turbine_events = turbine_events.explode('intervals')
#    turbine_events = turbine_events.drop(columns=['start', 'end'])
#    turbine_events = turbine_events.set_index('intervals')

#    turbine_events['Events'] = turbine_events['Status'] + " - " + turbine_events['Message'] + " - " + turbine_events['Service contract category'].fillna('') + " - " + turbine_events['IEC category'].fillna('')

    # Group by interval and concatenate events
#    df_grouped = turbine_events.groupby(turbine_events.index)[["Status", "Events"]].agg({'Events': ' | '.join, "Status": lambda s: "Stop" if (s == "Stop").any() else None})

#    return df_grouped


#def load_turbine_electricity_data(data_path: str = "../data"):
#    try:
#        p = Path(f"{data_path}/zenodo_turbine_data/raw/2016/Turbine_Data_Penmanshiel_09_2016-06-24_-_2017-01-01_1049.csv")
#        turbine_09_data = pd.read_csv(p,
#            skiprows=9,
#            quotechar='"',
#            encoding='utf-8-sig')
#        turbine_09_data = turbine_09_data.rename(columns={"# Date and time": "time"})
#        turbine_09_data['time'] = pd.to_datetime(turbine_09_data['time'])
#        turbine_09_data = turbine_09_data.set_index("time")

#        return turbine_09_data
    
#    except Exception as e:
#        print(f"Error loading turbine electricity data: {e}")
#        return None  # Return None explicitly on error
    

def load_turbine_data_dynamic(year: int):
    turbine_data = load_turbine_electricity_data_dynamic(year)
    turbine_events = load_turbine_events_dynamic(year)
    merged_data = turbine_data.merge(turbine_events, how='left', left_index=True, right_index=True)
    merged_data['Status'] = merged_data.Status.fillna("Running").astype("category")
    
    return merged_data


def load_turbine_events_raw(year: int, data_path: str = "../data"):
        """
        Loads turbine events data for the specified year.
        """
        try:
            # Adjust file start date if the year is 2016
            start_date = "2016-06-24" if year == 2016 else f"{year}-01-01"
            
            file_path = Path(f"{data_path}/zenodo_turbine_data/raw/{year}/Status_Penmanshiel_09_{start_date}_-_{year + 1}-01-01_1049.csv")
            
            # Read the CSV file
            turbine_events = pd.read_csv(
                file_path,
                skiprows=9,
                quotechar='"',
                encoding='utf-8-sig'
            )
            return turbine_events

        except FileNotFoundError:
            print(f"Error: File for year {year} not found.")
            return None  # Explicitly return None on file error
    

def load_turbine_events_dynamic(year: int, data_path: str = "../data"):
        """
        Loads turbine events data for the specified year.
        """
        try:
            # Adjust file start date if the year is 2016
            start_date = "2016-06-24" if year == 2016 else f"{year}-01-01"
            
            file_path = Path(f"{data_path}/zenodo_turbine_data/raw/{year}/Status_Penmanshiel_09_{start_date}_-_{year + 1}-01-01_1049.csv")
            
            # Read the CSV file
            turbine_events = pd.read_csv(
                file_path,
                skiprows=9,
                quotechar='"',
                encoding='utf-8-sig'
            )
    
            turbine_events = turbine_events.rename(columns={'Timestamp start': 'start', 'Timestamp end': 'end'})
            turbine_events = turbine_events.loc[:, ["Status", 'start', 'end', 'Message', 'Service contract category','IEC category']]
            turbine_events['start'] = pd.to_datetime(turbine_events['start'])
            turbine_events = turbine_events[turbine_events['end'] != "-"]
            turbine_events['end'] = pd.to_datetime(turbine_events['end'])
            turbine_events['start'] = turbine_events['start'].dt.floor('10min')
            turbine_events['end'] = turbine_events['end'].dt.ceil('10min')
            turbine_events['intervals'] = turbine_events.apply(lambda row: pd.date_range(row['start'], row['end'], freq='10min', inclusive='left'), axis=1)
            turbine_events = turbine_events.explode('intervals')
            turbine_events = turbine_events.drop(columns=['start', 'end'])
            turbine_events = turbine_events.set_index('intervals')

            turbine_events['Events'] = turbine_events['Status'] + " - " + turbine_events['Message'] + " - " + turbine_events['Service contract category'].fillna('') + " - " + turbine_events['IEC category'].fillna('')

            # Group by interval and concatenate events
            df_grouped = turbine_events.groupby(turbine_events.index)[["Status", "Events"]].agg({'Events': ' | '.join, "Status": lambda s: "Stop" if (s == "Stop").any() else None})

            return df_grouped
        
        except FileNotFoundError:
            print(f"Error: File for year {year} not found.")
            return None  # Explicitly return None on file error

def load_turbine_electricity_data_dynamic(year: int, data_path: str = "../data"):
    """
    Loads turbine electricity data for the specified year.
    """
    try:
        # Determine start and end dates based on the year
        if year == 2016:
            start_code = "2016-06-24"
            end_code = f"{year + 1}-01-01"
        elif year < 2021:
            start_code = f"{year}-01-01"
            end_code = f"{year + 1}-01-01"
        elif year == 2021:
            start_code = "2021-01-01"
            end_code = "2021-07-01"
        else:
            raise ValueError("Data for years after 2021 is not available.")

        # Construct the file path based on the year and date codes
        file_path = Path(f"{data_path}/zenodo_turbine_data/raw/{year}/Turbine_Data_Penmanshiel_09_{start_code}_-_{end_code}_1049.csv")

        print(f"Attempting to load file: {file_path}")

        # Read the CSV file
        turbine_09_data = pd.read_csv(
            file_path,
            skiprows=9,
            quotechar='"',
            encoding='utf-8-sig'
        )

        # Rename columns and set the 'time' column as the index
        turbine_09_data = turbine_09_data.rename(columns={"# Date and time": "time"})
        turbine_09_data['time'] = pd.to_datetime(turbine_09_data['time'])
        turbine_09_data = turbine_09_data.set_index("time")

        return turbine_09_data
    
    except FileNotFoundError:
        print(f"Error: File for year {year} not found.")
        return None
    except Exception as e:
        print(f"Error loading turbine electricity data: {e}")
        return None


def load_weather_forecast(data_path: str = "../data"):

    try:
        file_path = Path(f"{data_path}/weather_forecast/raw/open-meteo-53.00N13.12E89m.csv")

        weather_forecast = pd.read_csv(
                file_path,
                skiprows=3,
                quotechar='"',
                encoding='utf-8-sig'
            )
        return weather_forecast
        
    except FileNotFoundError:
        print(f"Error: File not found.")
        return None



