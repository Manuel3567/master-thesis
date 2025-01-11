import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


# 2 Types of datasets: Aggregated, Turbine level

#-----------------------------------------------------------------------------------------
# 1. Aggregated has the following datasets, each data is loaded via its own method:
    # 1. ENTSO-E (Aggregated power output of turbines)
    # 2. eeg 50Hertz-Anlagenstammdaten --> find the installed capacity of the grid providers
    # 3. plz in Germany
    # 4. merged file of 2. and 3. to find  --> maps the grid providers to a location (plz)
    # 5. Wind speed data at the location of the 10 biggest wind parks identified
#-----------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------
# 2. Turbine level dataset has the following datasets, each data is loaded via its own method:
    # 1. SCADA data (power output & wind speed at the exact location of the turbine)
    # 2. Turbine events
    # 3. merged file of 1. and 2. to map events to turbine
    # 4 3-months weather forecasts (d1 = day ahead forecasts, d0 = latest update forecasts) --> idea: analyse this dataset and add noise to measured data at the turbine 
#-----------------------------------------------------------------------------------------





# 1.1 ENTSO-E (Aggregated power output of turbines)

def load_entsoe(data_path: str = "../data"):

    files = list(Path(f"{data_path}/entsoe").glob("*"))
    dfs = [_load_entsoe(f) for f in files]
    df = pd.concat(dfs)

    return df


def _load_entsoe(file: Path):
    cleaned_lines = []
    
    with open(file, "r") as f:
        for line in f:
            wrong_header = '"Area,""MTU"",""Wind Offshore - Actual Aggregated [MW]"",""Wind Onshore - Actual Aggregated [MW]"""\n'
            correct_header = '"Area","MTU","Wind Offshore  - Actual Aggregated [MW]","Wind Onshore  - Actual Aggregated [MW]"\n'
            # Fix the first column's missing quote
            if line == wrong_header:
                line = correct_header
            
            line = line.replace("CTA|DE(50Hertz),", 'CTA|DE(50Hertz)",', 1)  # Add missing quote to rows
            line = line.replace('"""', '"')
            line = line.replace('""', '"')


            # Append cleaned line
            cleaned_lines.append(line)
    
    # Write the cleaned data to a temporary buffer
    from io import StringIO
    cleaned_csv = StringIO("".join(cleaned_lines))
    
    # Load the cleaned CSV into a DataFrame
    df = pd.read_csv(cleaned_csv, delimiter=",", quotechar='"')
    df.columns = [col.strip() for col in df.columns]  # Normalize column names
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


# 1.2 eeg 50Hertz-Anlagenstammdaten

def load_and_process_eeg_data(eeg_file_path='../data/50Hertz_Transmission_GmbH_EEG-Zahlungen_Stammdaten_2023.csv'):
    """
    Loads and processes the EEG data.

    Parameters:
        eeg_file_path (str): Path to the EEG payments CSV file.

    Returns:
        pd.DataFrame: Processed EEG data.
    """
    dtype_dict = {'PLZ': 'object'}  # Treat PLZ as an object initially

    # Load EEG data
    eeg_data = pd.read_csv(
        eeg_file_path,
        encoding='ISO-8859-1',
        engine='python',
        sep=";",
        dtype=dtype_dict
    )

    # Drop NaN and convert 'PLZ' to int
    eeg_data = eeg_data.dropna(subset=['PLZ'])
    eeg_data['PLZ'] = eeg_data['PLZ'].astype('int')

    # Rename column PLZ to plz
    eeg_data.rename(columns={'PLZ': 'plz'}, inplace=True)

    # Convert "Installierte_Leistung" to numeric
    eeg_data['Installierte_Leistung'] = pd.to_numeric(
        eeg_data['Installierte_Leistung'], errors='coerce'
    )

    return eeg_data


# 1.3 plz in Germany

def load_and_process_plz_data(plz_file_path='../data/georef-germany-postleitzahl.csv'):
    """
    Loads and processes the Germany postal code data.

    Parameters:
        plz_file_path (str): Path to the Germany postal code CSV file.

    Returns:
        pd.DataFrame: Processed postal code data.
    """
    dtype_dict = {'Postleitzahl / Post code': 'int'}

    # Load postal code data
    plz_data = pd.read_csv(
        plz_file_path,
        skip_blank_lines=False,
        encoding='utf-8-sig',
        sep=";",
        dtype=dtype_dict
    )

    # Drop duplicates and clean up columns
    plz_data = plz_data.drop_duplicates(subset=['Postleitzahl / Post code'], keep='first')
    plz_data = plz_data.dropna(subset=['Postleitzahl / Post code'])
    plz_data.rename(columns={'Postleitzahl / Post code': 'plz'}, inplace=True)

    return plz_data


# 1.4 Merged file of 2. and 3. to find  --> maps the grid providers to a location (plz)

def load_wind_turbine_coordinates_from_zips():
    df = merge_eeg_plz_data()

    return df

def load_10_biggest_wind_turbines_by_installed_capacity(n):
    df = merge_eeg_plz_data().head(n)
    
    return df

def get_coordinates_of_geographic_mean_and_10_biggest_wind_turbines_by_installed_capacity():
    """
    Computes the geographic mean (weighted by installed capacity) and retrieves
    the coordinates of the 10 largest wind turbines by installed capacity.

    Returns:
        tuple: (latitudes, longitudes) as comma-separated strings.
    """
    aggregated_df = merge_eeg_plz_data()
    df = load_10_biggest_wind_turbines_by_installed_capacity(n=10)

    # Calculate weighted averages
    weighted_longitude = (aggregated_df['longitude'] * aggregated_df['installed_capacity_sum']).sum() / aggregated_df['installed_capacity_sum'].sum()
    weighted_latitude = (aggregated_df['latitude'] * aggregated_df['installed_capacity_sum']).sum() / aggregated_df['installed_capacity_sum'].sum()

    # Create latitude and longitude strings
    longitudes = f"{weighted_longitude}," + ",".join(df['longitude'].astype(str))
    latitudes = f"{weighted_latitude}," + ",".join(df['latitude'].astype(str))

    return latitudes, longitudes



def merge_eeg_plz_data():
    """
    Merges the EEG data and postal code data, and performs aggregation.

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """

    eeg_data = load_and_process_eeg_data()
    plz_data = load_and_process_plz_data()

    # Merge datasets
    merged_df = eeg_data.merge(plz_data, on='plz', how='left')

    # Split 'geo_point_2d' into latitude and longitude
    merged_df[['latitude', 'longitude']] = merged_df['geo_point_2d'].str.split(',', expand=True)

    # Convert latitude and longitude to numeric
    merged_df['latitude'] = pd.to_numeric(merged_df['latitude'], errors='coerce')
    merged_df['longitude'] = pd.to_numeric(merged_df['longitude'], errors='coerce')

    # Filter for Energietraeger (category wind) == 7
    filtered_df = merged_df[merged_df['Energietraeger'] == 7]
    filtered_df = filtered_df[['plz', 'Installierte_Leistung', 'EEG-Anlagenschluessel', 'longitude', 'latitude']]

    # Group and aggregate data
    aggregated_df = filtered_df.groupby('plz').agg(
        installed_capacity_sum=('Installierte_Leistung', 'sum'),
        eeg_count=('EEG-Anlagenschluessel', 'nunique'),
        longitude=('longitude', 'first'),
        latitude=('latitude', 'first')
    ).reset_index()

    # Remove rows where installed_capacity_sum is zero
    aggregated_df = aggregated_df[aggregated_df['installed_capacity_sum'] != 0]

    # Sort by installed_capacity_sum
    aggregated_df = aggregated_df.sort_values(by='installed_capacity_sum', ascending=False)

    aggregated_df = aggregated_df.sort_values(by='installed_capacity_sum', ascending=False)

    aggregated_df['cumulative_capacity'] = aggregated_df['installed_capacity_sum'].cumsum()

    total_installed_capacity = aggregated_df['installed_capacity_sum'].sum()
    aggregated_df['cumulative_percentage'] = (aggregated_df['cumulative_capacity'] / total_installed_capacity) * 100

    return aggregated_df

    


# 1.5 Wind speed data at the location of the 10 biggest wind parks identified

def load_wind_park_data(file_path="../data/top_10_biggest_wind_parks/top_10_biggest_wind_parks_50hertz.json"):
    try:
        # Load the JSON file
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Initialize an empty list to hold the processed data
        processed_data = []

        # Iterate through each location's data in the list
        for location_data in data:
            # Extract 'hourly' data for each location
            hourly_data = location_data.get('hourly', {})
            
            # Check if 'time', 'wind_speed_10m', and 'wind_speed_100m' are in the hourly data
            if 'time' in hourly_data and 'wind_speed_10m' in hourly_data and 'wind_speed_100m' in hourly_data:
                # Create a DataFrame for this location's hourly data
                location_df = pd.DataFrame({
                    'time': hourly_data['time'],
                    'wind_speed_10m': hourly_data['wind_speed_10m'],
                    'wind_speed_100m': hourly_data['wind_speed_100m']
                })
                
                # Add metadata columns (latitude and longitude) from the location data
                location_df['latitude'] = location_data['latitude']
                location_df['longitude'] = location_data['longitude']
                
                # Append the location's data to the processed_data list
                processed_data.append(location_df)

        # Concatenate all individual DataFrames from each location
        df = pd.concat(processed_data, ignore_index=True)

        # Add transformation: pivot the data to reshape it
        df['location'] = df['latitude'].astype(str) + "_" + df['longitude'].astype(str)  # Unique location identifier
        pivoted_df = df.pivot(index='time', columns='location', values=['wind_speed_10m', 'wind_speed_100m'])
        pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]  # Flatten multi-level columns
        pivoted_df.index.name = "time"  # Optional: explicitly name the index
        pivoted_df.index = pd.to_datetime(pivoted_df.index, errors='coerce')
        # Rename the specific columns
        pivoted_df = pivoted_df.rename(columns={
            "wind_speed_10m_52.40773_12.523191": "wind_speed_10m_mean",
            "wind_speed_100m_52.40773_12.523191": "wind_speed_100m_mean"
        })

        print(f"Data loaded and transformed successfully. Shape of DataFrame: {pivoted_df.shape}")
        return pivoted_df

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding the JSON file: {file_path}")
    except KeyError:
        print("The expected data structure is not found in the JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")

#-------------------------------------------------------------------------------------------------------------------------------------------

    # 2.1. SCADA Data

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


# 2.2 Turbine events

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
        



# 2.3 merged file of 1. and 2. to map events to turbine

def load_turbine_data_dynamic(year: int):
    turbine_data = load_turbine_electricity_data_dynamic(year)
    turbine_events = load_turbine_events_dynamic(year)
    merged_data = turbine_data.merge(turbine_events, how='left', left_index=True, right_index=True)
    merged_data['Status'] = merged_data.Status.fillna("Running").astype("category")
    
    return merged_data


# 2.4 3-months weather forecasts (d1 = day ahead forecasts, d0 = latest update forecasts) --> idea: analyse this dataset and add noise to measured data at the turbine 

def load_3_months_0day_1day_forecasts_zenodo(file_path="../data/weather_forecast/raw/previous_model_runs_forecasts_latitude_55.93_longitude_-2.3000002_three_months.json"):
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            # Open the JSON file and load the data
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract the hourly time and the wind speed values from the data
            minutely_15_data = data["minutely_15"]
            time_series = minutely_15_data["time"]
            
            # Now, let's extract the wind speed values at different heights (e.g., wind_speed_10m, wind_speed_80m, etc.)
            wind_speeds = {key: minutely_15_data.get(key, []) for key in minutely_15_data if key.startswith('wind_speed')}
            
            # Create a DataFrame from the wind speed data and the time
            df = pd.DataFrame(wind_speeds)
            df["time"] = time_series  # Add the time column
            
            # Ensure the time column is in datetime format
            df["time"] = pd.to_datetime(df["time"])
            
            # Set time as the index (optional)
            df.set_index("time", inplace=True)
            
            print(f"Data loaded successfully. Shape of the dataframe: {df.shape}")
            return df
        else:
            print(f"File not found at {file_path}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None




# old

""" def load_weather_forecast(data_path: str = "../data"):

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
        return None """

""" def load_actual_historical_weather_zenodo(file_path="../data/weather_forecast/raw/actual_historical_weather_3_months_zenodo_latitude_55.92267_longitude_-2.3926392_three_months.json"):
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            # Open the JSON file and load the data
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract the hourly time and the wind speed values from the data
            hourly_data = data["hourly"]
            time_series = hourly_data["time"]
            
            # Now, let's extract the wind speed values at different heights (e.g., wind_speed_10m, wind_speed_80m, etc.)
            wind_speeds = {key: hourly_data.get(key, []) for key in hourly_data if key.startswith('wind_speed')}
            
            # Create a DataFrame from the wind speed data and the time
            df = pd.DataFrame(wind_speeds)
            df["time"] = time_series  # Add the time column
            
            # Ensure the time column is in datetime format
            df["time"] = pd.to_datetime(df["time"])
            
            # Set time as the index (optional)
            df.set_index("time", inplace=True)
            
            print(f"Data loaded successfully. Shape of the dataframe: {df.shape}")
            return df
        else:
            print(f"File not found at {file_path}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None """


""" def load_merra(data_path: str = "../data", longitude: float = 13.125, lattitude: float = 53.0, 
               start_date: str = None, end_date: str = None):
    # Define path
    p = Path(f"{data_path}/merra2")
    
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


def load_merra2(data_path: str = "../data", longitude: float = -2.0, lattitude: float = 56, 
               start_date: str = None, end_date: str = None):
    # Define path
    p = Path(f"{data_path}/merra2_pennmanshiel/merra2")
    
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

    return df """


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



""" def load_turbine_events_raw(year: int, data_path: str = "../data"):
        Loads turbine events data for the specified year.
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
"""