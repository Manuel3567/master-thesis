import os
import xarray as xr
import requests
from datetime import datetime, timedelta
import warnings
import zipfile
from io import BytesIO
warnings.filterwarnings("ignore")
import json










# Function to download MERRA data for a range of dates
def download_merra(start_date, end_date, token=None, longitude=13.125, lattitude=53.0, output_dir="../data/merra2/"):


    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if token is None:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ["MERRA_TOKEN"]
    
    get_longitude = lambda x: int((x+180)*8/5) # conversion for gesdisc API
    get_lattitude = lambda y: int(y*2 + 180) # conversion for gesdisc API
    longitude_coord = longitude
    lattitude_coord = lattitude

    longitude = get_longitude(longitude)
    lattitude = get_lattitude(lattitude)

    # Create a session and set up authentication
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {token}'})
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize current_date as start_date
    current_date = start_date

    # Loop through each date
    while current_date < end_date:
        # Format the year, month, and day
        year = current_date.year
        month = f"{current_date.month:02d}"
        day = f"{current_date.day:02d}"

        # Generate the URL based on the template

        def generate_url(suffix = "400"):

            url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I1NXLFO.5.12.4/{year}/{month}/MERRA2_{suffix}.inst1_2d_lfo_Nx.{year}{month}{day}.nc4?HLML[0:1:23][{lattitude}][{longitude}],QLML[0:1:23][{lattitude}][{longitude}],SPEEDLML[0:1:23][{lattitude}][{longitude}],PS[0:1:23][{lattitude}][{longitude}],TLML[0:1:23][{lattitude}][{longitude}],lat[{lattitude}],lon[{longitude}],time[0:1:23]"
            #url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I1NXLFO.5.12.4/{year}/{month}/MERRA2_{suffix}.inst1_2d_lfo_Nx.{year}{month}{day}.nc4?HLML[0:1:23][{lattitude}][{longitude}],QLML[0:1:23][{lattitude}][{longitude}],SPEEDLML[0:1:23][{lattitude}][{longitude}],PS[0:1:23][{lattitude}][{longitude}],TLML[0:1:23][{lattitude}][{longitude}],lat[{lattitude}],lon[{longitude}],time[0:1:23]"

            return url
        
        # Define the output file path
        output_file = f"{output_dir}/merra2_{longitude_coord}_{lattitude_coord}_{year}_{month}_{day}.nc"
        
        # Check if the file already exists
        if os.path.exists(output_file):
            print(f"File already exists, skipping download: {output_file}")
        else:
            try:
                
                try:
                    url = generate_url()
                    print(f"Downloading: {url} to {output_file}")
                    # Use PydapDataStore to open the dataset with authentication
                    store = xr.backends.PydapDataStore.open(url, session=session)
                
                except Exception as e:
                    url = generate_url(suffix="401")
                    print(f"Use 401 suffix instead, downloading: {url} to {output_file}")
                    store = xr.backends.PydapDataStore.open(url, session=session)


                # Create an xarray dataset from the opened store
                ds = xr.open_dataset(store)
                
                # Save the dataset to a NetCDF file
                ds.to_netcdf(output_file)
                
                print(f"Saved dataset for {current_date} to {output_file}")
            
            except Exception as e:
                print(f"Failed to process {url}: {e}")
        
        # Increment the current_date by 1 day
        current_date += timedelta(days=1)


# Function to download MERRA data for a range of dates
def download_merra2(start_date, end_date, token=None, longitude=-2.0, lattitude=56, output_dir="../data/merra2_pennmanshiel/merra2/"):

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if token is None:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ["MERRA_TOKEN"]
    
    get_longitude = lambda x: int((x+180)*8/5) # conversion for gesdisc API
    get_lattitude = lambda y: int(y*2 + 180) # conversion for gesdisc API
    longitude_coord = longitude
    lattitude_coord = lattitude

    longitude = get_longitude(longitude)
    lattitude = get_lattitude(lattitude)

    # Create a session and set up authentication
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {token}'})
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize current_date as start_date
    current_date = start_date

    # Loop through each date
    while current_date < end_date:
        # Format the year, month, and day
        year = current_date.year
        month = f"{current_date.month:02d}"
        day = f"{current_date.day:02d}"

        # Generate the URL based on the template

        def generate_url(suffix = "400"):

            url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I1NXLFO.5.12.4/{year}/{month}/MERRA2_{suffix}.inst1_2d_lfo_Nx.{year}{month}{day}.nc4?HLML[0:1:23][{lattitude}][{longitude}],QLML[0:1:23][{lattitude}][{longitude}],SPEEDLML[0:1:23][{lattitude}][{longitude}],PS[0:1:23][{lattitude}][{longitude}],TLML[0:1:23][{lattitude}][{longitude}],lat[{lattitude}],lon[{longitude}],time[0:1:23]"
            #url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2I1NXLFO.5.12.4/{year}/{month}/MERRA2_{suffix}.inst1_2d_lfo_Nx.{year}{month}{day}.nc4?HLML[0:1:23][{lattitude}][{longitude}],QLML[0:1:23][{lattitude}][{longitude}],SPEEDLML[0:1:23][{lattitude}][{longitude}],PS[0:1:23][{lattitude}][{longitude}],TLML[0:1:23][{lattitude}][{longitude}],lat[{lattitude}],lon[{longitude}],time[0:1:23]"

            return url
        
        # Define the output file path
        output_file = f"{output_dir}/merra2_{longitude_coord}_{lattitude_coord}_{year}_{month}_{day}.nc"
        
        # Check if the file already exists
        if os.path.exists(output_file):
            print(f"File already exists, skipping download: {output_file}")
        else:
            try:
                
                try:
                    url = generate_url()
                    print(f"Downloading: {url} to {output_file}")
                    # Use PydapDataStore to open the dataset with authentication
                    store = xr.backends.PydapDataStore.open(url, session=session)
                
                except Exception as e:
                    url = generate_url(suffix="401")
                    print(f"Use 401 suffix instead, downloading: {url} to {output_file}")
                    store = xr.backends.PydapDataStore.open(url, session=session)


                # Create an xarray dataset from the opened store
                ds = xr.open_dataset(store)
                
                # Save the dataset to a NetCDF file
                ds.to_netcdf(output_file)
                
                print(f"Saved dataset for {current_date} to {output_file}")
            
            except Exception as e:
                print(f"Failed to process {url}: {e}")
        
        # Increment the current_date by 1 day
        current_date += timedelta(days=1)
    

def download_zenodo(start_year: int, end_year: int, output_dir="../data/zenodo_turbine_data/raw/"):
    # Time period: 2016-06-24 11:40 - 2021-06-30 23:50
    
    # Loop through the range of years specified
    for year in range(start_year, end_year + 1):
        print(f"\nStarting download for the year {year}...")
        
        # Directory setup for the specific year
        output_dir_year = os.path.join(output_dir, str(year))
        os.makedirs(output_dir_year, exist_ok=True)

        # Conditionally set turbine range and turbine codes
        if year == 2016:
            turbine_combinations = [("01-10", 3107), ("11-15", 3107)]
        elif year == 2017:
            turbine_combinations = [("01-10", 3114), ("11-15", 3115)]
        elif year == 2018:
            turbine_combinations = [("01-10", 3113), ("11-15", 3116)]
        elif year == 2019:
            turbine_combinations = [("01-10", 3112), ("11-15", 3117)]
        elif year == 2020:
            turbine_combinations = [("01-10", 3109), ("11-15", 3118)]
        elif year == 2021:
            turbine_combinations = [("01-10", 3108), ("11-15", 3108)]
        else:
            print(f"No data for the year {year}. Skipping...")
            continue  # Skip years not in the list

        # Loop through the turbine combinations for this year
        for turbine_range, code in turbine_combinations:
            # Construct the URL and expected file name
            url = f"https://zenodo.org/records/5946808/files/Penmanshiel_SCADA_{year}_WT{turbine_range}_{code}.zip?download=1"
            filename = f"Penmanshiel_SCADA_{year}_WT{turbine_range}_{code}.zip"
            file_path = os.path.join(output_dir_year, filename)

            # Check if the file already exists
            if os.path.exists(file_path):
                print(f"File already exists: {filename}. Skipping download.")
                continue  # Skip to the next file if it exists

            print(f"Attempting to download {filename}... : {url}")

            # Download the file and check for successful response
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    # Extract the zip file content
                    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                        zip_file.extractall(output_dir_year)
                    print(f"Downloaded and extracted: {filename} to {output_dir_year}")
                else:
                    print(f"Failed to download {filename}: HTTP {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {filename}: {e}")
            except zipfile.BadZipFile:
                print(f"Downloaded file is not a valid ZIP: {filename}")




def download_3_months_0day_1day_forecasts_zenodo(output_dir="../data/weather_forecast/raw"):
    """
    Downloads a 3-month 0-day and 1-day wind speed forecast for a location as close as possible to the Pennmanshiel Turbine 09 site (actual location: latitude: 55.904990, longitude: -2.291806) 
    with data interpolated at a 15-minute resolution
    
    This function retrieves historical wind speed forecasts for multiple heights (10m, 80m, 120m, and 180m) 
    using the Previous Runs API. Forecasts include same-day (0-day) and previous-day (1-day) wind speed predictions. 
    The data is stored in a JSON file in the specified output directory.

    Parameters:
        output_dir (str): The directory where the JSON data file will be saved. Defaults to 
                          "../data/weather_forecast/raw".

    Functionality:
        - Sends a GET request to the Previous Runs API for wind speed data.
        - Checks if the request is successful (status code 200).
        - Saves the fetched data in JSON format in the specified directory.

    API Details:
        - Latitude: 55.90499
        - Longitude: -2.291806
        - Forecast Heights: 10m, 80m, 120m, 180m
        - Forecast Types: Same-day (0-day) and previous-day (1-day) forecasts
        - Wind Speed Unit: m/s
        - Data Span: Past 92 days
        - Models: Best match
        - API costs for hourly data: 5.7, 15_minutely: ?

    Output:
        - A JSON file named "weather_forecasts_latitude_55.93_longitude_-2.3000002_three_months.json" containing 
          the downloaded data.
        - location returned: latitude:55.9,"longitude:-2.3000002

    Exceptions:
        - Prints an error message if a network issue or any other `RequestException` occurs.
        - Prints a status code error message if the API request fails.
    """
        
    url = "https://previous-runs-api.open-meteo.com/v1/forecast?latitude=55.90499&longitude=-2.291806&minutely_15=wind_speed_10m,wind_speed_10m_previous_day1,wind_speed_80m,wind_speed_80m_previous_day1,wind_speed_120m,wind_speed_120m_previous_day1,wind_speed_180m,wind_speed_180m_previous_day1&wind_speed_unit=ms&past_days=92&models=best_match"

    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON data
            data = response.json()
            os.makedirs(output_dir, exist_ok=True)

            file_path = os.path.join(output_dir, "previous_model_runs_forecasts_latitude_55.93_longitude_-2.3000002_three_months.json")

            # Save the data to the file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)  # Save with pretty indentation

            print(f"Data saved successfully to {file_path}")
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    

def download_actual_historical_weather_3_months_zenodo(output_dir="../data/weather_forecast/raw"):
    """
    Downloads hourly (15 minute not even available as interpolated data) historical weather data for a location near the Pennmanshiel Turbine 09 site 
    (latitude: 55.904990, longitude: -2.291806) with data spanning three months so that the time period is the same as 
    this used by the function: "download_3_months_0day_1day_forecasts_zenodo"

    This function retrieves archived hourly wind speed data from the Open Meteo Archive API for two heights 
    (10m and 100m) and saves the results in JSON format. The target location is approximated to latitude: 55.9 
    and longitude: -2.3000002, ensuring the closest possible match to the Pennmanshiel Turbine 09 site.

    Parameters:
        output_dir (str): The directory where the JSON data file will be saved. Defaults to 
                          "../data/weather_forecast/raw".

    Functionality:
        - Sends a GET request to the Open Meteo Archive API for wind speed data.
        - Checks for successful responses (HTTP status code 200).
        - Saves the retrieved weather data in a JSON file with formatted indentation in the specified directory.

    API Details:
        - **Latitude:** 55.9 (approximation)
        - **Longitude:** -2.3000002 (approximation)
        - **Forecast Heights:** 10m, 100m
        - **Wind Speed Unit:** m/s
        - **Time Range:** 2024-09-02 to 2024-12-02
        - **Data Resolution:** Hourly
        - **API Costs:** 1.3 calls 

    Output:
        - Saves a JSON file named "weather_forecasts_latitude_55.93_longitude_-2.3000002_three_months.json" 
          containing the retrieved wind speed data.
        - Location returned: latitude: 55.9, longitude: -2.3000002.

    Exceptions:
        - Prints an error message if a network-related error (`RequestException`) occurs.
        - Logs the HTTP status code if the API request is unsuccessful.

    Example:
        >>> download_actual_historical_weather_zenodo()
        Data saved successfully to ../data/weather_forecast/raw/weather_forecasts_latitude_55.93_longitude_-2.3000002_three_months.json
    """
        
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=55.9&longitude=-2.3000002&start_date=2024-09-02&end_date=2024-12-02&hourly=wind_speed_10m,wind_speed_100m&wind_speed_unit=ms"

    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON data
            data = response.json()
            os.makedirs(output_dir, exist_ok=True)

            file_path = os.path.join(output_dir, "actual_historical_weather_3_months_zenodo_latitude_55.92267_longitude_-2.3926392_three_months.json")
            
            # Save the data to the file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)  # Save with pretty indentation

            print(f"Data saved successfully to {file_path}")
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    

import requests
import pandas as pd
from time import sleep


def find_minimum_start_date(lat_start, lat_end, lon_start, lon_end, lat_steps, lon_steps):
    """
    Finds the earliest available historical data date for combinations of latitude and longitude.

    Parameters:
        lat_start (float): Starting latitude value.
        lat_end (float): Ending latitude value.
        lon_start (float): Starting longitude value.
        lon_end (float): Ending longitude value.
        lat_steps (float): Number of steps to divide the latitude range.
        lon_steps (float): Number of steps to divide the longitude range.

    Returns:
        dict: A dictionary containing the latitude, longitude, and the earliest date found.
    """
    latitudes = [lat_start + i for i in range(lat_steps)]
    longitudes = [lon_start + i for i in range(lon_steps)]
    
    earliest_date = None
    best_location = None
    
    for lat in latitudes:
        for lon in longitudes:
            try:
                sleep(20)
                url = (
                    f"https://api.open-meteo.com/v1/forecast?"
                    f"latitude={lat}&longitude={lon}&hourly=wind_speed_10m&wind_speed_unit=ms"
                    f"&start_date=2021-03-23&end_date=2021-03-24"
                )
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    print(data)
                    # Extract the minimum date if available
                    date = data.get("start_date")  # Adjust the key based on actual API response
                    if date:
                        if earliest_date is None or date < earliest_date:
                            earliest_date = date
                            best_location = {"latitude": lat, "longitude": lon, "earliest_date": date}
                            print(best_location)
            except requests.exceptions.RequestException as e:
                print(f"Error for latitude={lat}, longitude={lon}: {e}")

    return best_location


'''
def download_forecast_data(output_dir="../data/weather_forecast/raw"):

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=55.82&longitude=-2.34&start_date=2021-03-23&end_date=2021-12-31&hourly=wind_speed_10m,wind_speed_80m,wind_speed_120m&wind_speed_unit=ms&models=gfs_seamless"

    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON data
            data = response.json()
            os.makedirs(output_dir, exist_ok=True)

            file_path = os.path.join(output_dir, "weather_forecast_data.json")

            # Save the data to the file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)  # Save with pretty indentation

            print(f"Data saved successfully to {file_path}")
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
'''

from analysis.datasets import get_coordinates_of_geographic_mean_and_10_biggest_wind_turbines_by_installed_capacity
import os
import requests
import json


def download_open_meteo_wind_speeds_of_10_biggest_wind_park_locations_and_at_geographic_mean(start_date, end_date, output_dir):

    latitudes, longitudes = get_coordinates_of_geographic_mean_and_10_biggest_wind_turbines_by_installed_capacity()
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitudes}&longitude={longitudes}&start_date={start_date}&end_date={end_date}&hourly=wind_speed_10m,wind_speed_100m&wind_speed_unit=ms"
    print(url)
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        print(f"Status Code: {response.status_code}\n")  # Check the status code
        print(f"Response Content: {response.text}")  # Check the actual response
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
                # Parse the JSON data
                data = response.json()
                os.makedirs(output_dir, exist_ok=True)

                file_path = os.path.join(output_dir, "top_10_biggest_wind_parks_50hertz.json")

                # Save the data to the file
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=4)  # Save with pretty indentation

                print(f"Data saved successfully to {file_path}")
        else:
                print(f"Failed to retrieve data. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")