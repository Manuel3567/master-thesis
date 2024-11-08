.PHONY: all clean electricity merra data

# Makefile to download and unzip datasets

# Variables
DATA_DIR = data

# Default target
all: data
data: merra electricity

# Target to download the electricity data
electricity:
	@echo "Downloading electricityloaddiagrams20112014.zip..."
	@curl -o $(DATA_DIR)/raw/electricityloaddiagrams20112014.zip https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
	@echo "Unzipping $(DATA_DIR)/raw/electricityloaddiagrams20112014.zip into $(DATA_DIR)..."
	@mkdir -p $(DATA_DIR)
	@unzip -q $(DATA_DIR)/raw/electricityloaddiagrams20112014.zip -d $(DATA_DIR)
	@echo "Unzipped successfully."


merra:
	@echo "Downloading Merra2 data from NASA. Make sure to have a .env file with a valid MERRA_TOKEN set."
	@python -c "import analysis.downloads; analysis.downloads.download_merra('2017-01-01', '2024-09-01', output_dir='./data/raw/merra2')"
# Clean up the zip file and data directory
clean:
	@echo "Cleaning up..."
	@rm -rf $(DATA_DIR)
	@echo "Cleaned."