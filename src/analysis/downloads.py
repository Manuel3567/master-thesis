import os
import xarray as xr
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

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
        url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I1NXLFO.5.12.4/{year}/{month}/MERRA2_400.inst1_2d_lfo_Nx.{year}{month}{day}.nc4?HLML[0:1:23][{lattitude}][{longitude}],QLML[0:1:23][{lattitude}][{longitude}],SPEEDLML[0:1:23][{lattitude}][{longitude}],PS[0:1:23][{lattitude}][{longitude}],TLML[0:1:23][{lattitude}][{longitude}],lat[{lattitude}],lon[{longitude}],time[0:1:23]"

        # Define the output file path
        output_file = f"{output_dir}/merra2_{longitude_coord}_{lattitude_coord}_{year}_{month}_{day}.nc"
        
        # Check if the file already exists
        if os.path.exists(output_file):
            print(f"File already exists, skipping download: {output_file}")
        else:
            print(f"Downloading: {url} to {output_file}")
            try:
                # Use PydapDataStore to open the dataset with authentication
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