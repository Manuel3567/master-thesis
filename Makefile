.PHONY: all clean electricity data

# Makefile to download and unzip datasets

# Variables
DATA_DIR = data

# Default target
all: data
data: electricity

# Target to download the electricity data
electricity:
	@echo "Downloading electricityloaddiagrams20112014.zip..."
	@curl -o $(DATA_DIR)/raw/electricityloaddiagrams20112014.zip https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
	@echo "Unzipping $(DATA_DIR)/raw/electricityloaddiagrams20112014.zip into $(DATA_DIR)..."
	@mkdir -p $(DATA_DIR)
	@unzip -q $(DATA_DIR)/raw/electricityloaddiagrams20112014.zip -d $(DATA_DIR)
	@echo "Unzipped successfully."

# Clean up the zip file and data directory
clean:
	@echo "Cleaning up..."
	@rm -rf $(DATA_DIR)
	@echo "Cleaned."