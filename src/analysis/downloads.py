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
def download_merra(start_date, end_date, token=None, longitude=13.125, lattitude=53.0, output_dir="../data/raw/merra2/"):


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
    