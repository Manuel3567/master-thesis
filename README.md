# Setup
## 1. Clone the Repository
```
git clone https://github.com/Manuel3567/master-thesis.git
cd master-thesis
```
## 2. Set up a python virtual environment
```
python -m venv .venv
.venv\Scripts\activate
```
## 3. Install dependencies
if you are using a CUDA-compatible GPU
```
pip install -r requirements_gpu.txt
```
otherwise
```
pip install -r requirements.txt
```
## 4. Run
```
pip install -e .
```
# Prepare the data structure
4 data sources are used:
Sources 1. to 3. are downloaded manually by clicking on links that are provided below. The files then have to be saved in a specific directory format (details see below).
Source 4. is downloaded by running the code below in a jupyter notebook.

### 1. `entsoe`: `2016-2024`
### 2. `netztransparenz`: `EEG`, `legend`
### 3. `opendatasoft`: `PLZ`
### 4. `open_meteo`
The data sources need to be downloaded in a data folder in the following order:
1. entsoe
2. netztransparenz
3. opendatasoft
4. open meteo
```
project_root/
├── data/
│   ├── entsoe/               # Raw aggregated wind power data
│   ├── netztransparenz/      # wind power data of wind park data 
│   └── opendatasoft/         # PLZ list of Germany
|   └── open_meteo/           # Wind speed data
```
### 1. `entsoe`: `2016-2024` (LOGIN REQUIRED)
- login > Export Data > "Actual Generation per Production Type (year, CSV)"
    - downloads: [2016](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2016+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2016+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC), [2017](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2017+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2017+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC), [2018](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2018+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2018+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC), [2019](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2019+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2019+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC), [2020](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2020+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2020+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC), [2021](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2021+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2021+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC), [2022](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2022+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2022+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC), [2023](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2023+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2023+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC), [2024](https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=01.01.2024+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=01.01.2024+00:00|UTC|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B18&productionType.values=B19&dateTime.timezone=UTC&dateTime.timezone_input=UTC)

### 2. `Netztransparenz`:
- download [EEG (!make sure that underscores are used as separators instead of spaces in the file name)](https://www.netztransparenz.de/xspproxy/api/staticfiles/ntp-relaunch/dokumente/erneuerbare%20energien%20und%20umlagen/eeg/eeg-abrechnungen/eeg-jahresabrechnungen/eeg-anlagenstammdaten/stammdaten_2023/50hertz_transmission_gmbh_eeg-zahlungen_anlagenstammdaten_2023.zip)
- download [legend](https://www.netztransparenz.de/xspproxy/api/staticfiles/ntp-relaunch/dokumente/erneuerbare%20energien%20und%20umlagen/eeg/eeg-abrechnungen/eeg-jahresabrechnungen/eeg-anlagenstammdaten/anlagenstammdaten_legende.xlsx)

### 3. `opendatasoft`:
- Flat file formats > CSV
    - download [PLZ](https://public.opendatasoft.com/explore/dataset/georef-germany-postleitzahl/export/)

### 4. `open_meteo`
- (start, end dates have to be set):
- run this code in a new jupyter notebook
    ```
    from analysis.downloads import download_open_meteo_wind_speeds_of_10_biggest_wind_park_locations_and_at_geographic_mean
    download_open_meteo_wind_speeds_of_10_biggest_wind_park_locations_and_at_geographic_mean("2016-01-01", "2024-12-31")
    ```
### The final data structure should look like this:
```
project_root/
├── data/
│   ├── entsoe/               
│   │   ├── Actual Generation per Production Type_201601010000-201701010000.csv
│   │   ├── Actual Generation per Production Type_201701010000-201801010000.csv
│   │   ├── ...
│   │   └── Actual Generation per Production Type_202401010000-202501010000.csv
│   ├── netztransparenz/      
│   │   ├── 50Hertz_Transmission_GmbH_EEG-Zahlungen_Stammdaten_2023.csv
│   │   └── anlagenstammdaten_legende.xslx
│   ├── open_meteo/          
│   │   └── historical/
|   |       └── top_10_biggest_wind_parks_50hertz.json
│   └── opendatasoft/         
│       └── georef-germany-postleitzahl.csv
```
# Train models
Specify (create) a root directory: replace `output_dir` with your own directory. For each of the three models, a new folder with the name of the model will be automatically created if it does not yet exist.

### 5.1 Baseline
Run in a Jupyter notebook
```
output_dir = "C:\Users\Manuel\Documents\results"
id=1
from analysis.baseline_model import *
run_baseline_model(id, output_dir)
```
The output of this is in "C:\Users\Manuel\Documents\results\baseline_model"
### 5.2 NGBoost
Run in a Jupyter notebook
```
from analysis.ngboost import *
from analysis.datasets import load_entsoe
output_dir = "C:\Users\Manuel\Documents\results"
entsoe = load_entsoe()

evaluate_ngboost_model(
    entsoe, 
    target_column='power', 
    dist=Normal, 
    case=1, 
    n_estimators=100, 
    learning_rate=0.03, 
    random_state=42, 
    output_file=output_dir,
    train_start = "2016-01-01",
    train_end = "2022-12-31",
    validation_start = "2023-01-01",
    validation_end = "2023-12-31"
)
```
### 5.3 TabPFN
```
id=1
output_dir = "C:\Users\Manuel\Documents\results"
from analysis.TabPFN import *
run_tabpfn(id, output_dir)
```
# Evaluate models
### 6.1 Baseline
```
id=1
output_dir = "C:\Users\Manuel\Documents\results"
from analysis.baseline_model import *
calculate_scores_baseline(id, output_dir)
```
### 6.2 NGBoost
```
evaluation is included in evaluate_ngboost_model() 
```
### 6.3 TabPFN
```
id=1
output_dir = "C:\Users\Manuel\Documents\results"
from analysis.tabpfn import *
calculate_scores_tabpfn(id, output_dir)
```
### Output
```
├── output_dir/
│   ├── baseline/
│   │   ├── experiment_1.pkl
│   │   ├── ...
│   │   ├── experiment_2.pkl
│   │   └── quantiles/
│   │       ├── experiment_results_1.pkl
│   │       └── experiment_results_2.pkl
│   │       └── ...
│   ├── ngboost/
│   │   └── full_year/
│   │       ├── case1.xlsx
│   │       ├── case2.xlsx
│   │       ├── ...
│   │       ├── Merged_sheet.xlsx
│   │   └── q4_train/
│   │       ├── case1.xlsx
│   │       ├── case2.xlsx
│   │       ├── ...
│   │       ├── Merged_sheet.xlsx
│   └── tabpfn/
│       ├── experiment_1.pkl
│       ├── experiment_2.pkl
│       ├── ...
│       └── quantiles/
│           ├── experiment_results_1.pkl
│           └── experiment_results_2.pkl
│           └── ...
```
# Plots
## Geographic location of 50Hertz wind parks
```
from analysis.datasets import get_coordinates_of_grid_operators
from analysis.plots import plot_installed_capacity_scatter
aggregated_df = get_coordinates_of_grid_operators()
plot_installed_capacity_scatter(aggregated_df)
```
## Marginal distribution of power
```
from analysis.preprocessor import *
import matplotlib.pyplot as plt
import seaborn as sns

preprocessor = DataPreprocessor()
preprocessor.load_data()
entsoe = preprocessor.df

plt.figure(figsize=(6, 4))
# Plot histogram and KDE
sns.histplot(entsoe['power'], bins=150, color="lightblue")
plt.title("Marginal Distribution of Power")
plt.xlabel("Power [MW]")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```
## Marginal distribution of Log power
```
from analysis.preprocessor import *
import matplotlib.pyplot as plt
import seaborn as sns

preprocessor = DataPreprocessor()
preprocessor.load_data()
preprocessor = preprocessor.transform_power()
entsoe = preprocessor.df

plt.figure(figsize=(6, 4))
sns.histplot(entsoe['power'], bins=150, color="lightblue")
plt.title("Marginal Distribution of Log Power")
plt.xlabel("ln(Power/Power_max + eps)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

### Map of Experiment ID to training details
## ID Map
| Method   | ID   | Features                                        | Split               |
|:---------|:-----|:------------------------------------------------|:--------------------|
| Baseline | 1    | power, mean ws                                  | Q1 2022 / Q1 2023   |
| Baseline | 23   | p_t-96                                          | Q1 2022 / Q1 2023   |
| Baseline | 24   | p_t-96                                          | Q4 2022 / FY 2023   |
| Baseline | 25   | 2 mean, power_t-96                              | Q4 2022 / FY 2023   |
| Baseline | 26   | ws 10 loc 10m + 100, P_t-96                     | Q4 2022 / FY 2023   |
| Baseline | 27   | 2 mean, ws 10 loc 10m + 100, P_t-96             | Q4 2022 / FY 2023   |
| Baseline | 28   | 2 mean, ws 10 loc 10m + 100, P_t-96, time index | Q4 2022 / FY 2023   |
| Baseline | 29   | 2 mean, ws 10 loc 10m + 100, P_t-96, time index | 2016-2022 / FY 2023 |
| Baseline | 30   | 2 mean, ws 10 loc 10m + 100, P_t-96             | 2016-2022 / FY 2023 |
| Baseline | 31   | ws 10 loc 10m + 100, P_t-96                     | 2016-2022 / FY 2023 |
| Baseline | 32   | 2 mean, power_t-96                              | 2016-2022 / FY 2023 |
| Baseline | 33   | p_t-96                                          | 2016-2022 / FY 2023 |


Same ID but for 2016-2022 split
| Method   | ID  | Features                                                                                    | Split               |
|:---------|:----|:--------------------------------------------------------------------------------------------|:--------------------|
| NGBoost  | 1   | p_t-96, Loss function = CRPScore                                                           | Q4 2022 / FY 2023   |
| NGBoost  | 2   | p_t-96, Loss function = LogScore                                                           | Q4 2022 / FY 2023   |
| NGBoost  | 3   | 2 mean ws, Loss function = CRPScore                                                        | Q4 2022 / FY 2023   |
| NGBoost  | 4   | 2 mean ws, Loss function = LogScore                                                        | Q4 2022 / FY 2023   |
| NGBoost  | 5   | 2 mean, power_t-96, Loss function = CRPScore                                               | Q4 2022 / FY 2023   |
| NGBoost  | 6   | 2 mean, power_t-96, Loss function = LogScore                                               | Q4 2022 / FY 2023   |
| NGBoost  | 7   | ws 10 loc 10m + 100, P_t-96, Loss function = CRPScore                                      | Q4 2022 / FY 2023   |
| NGBoost  | 8   | ws 10 loc 10m + 100, P_t-96, Loss function = LogScore                                      | Q4 2022 / FY 2023   |
| NGBoost  | 9   | 2 mean, ws 10 loc 10m + 100, P_t-96, Loss function = CRPScore                              | Q4 2022 / FY 2023   |
| NGBoost  | 10  | 2 mean, ws 10 loc 10m + 100, P_t-96, Loss function = LogScore                              | Q4 2022 / FY 2023   |
| NGBoost  | 11  | 2 mean, ws 10 loc 10m + 100, P_t-96, time index, Loss function = CRPScore                  | Q4 2022 / FY 2023   |
| NGBoost  | 12  | 2 mean, ws 10 loc 10m + 100, P_t-96, time index, Loss function = LogScore                  | Q4 2022 / FY 2023   |
| NGBoost  | 13  | P_t-96, time index, Loss function = CRPScore                                               | Q4 2022 / FY 2023   |
| NGBoost  | 14  | P_t-96, time index, Loss function = LogScore                                               | Q4 2022 / FY 2023   |
| NGBoost  | 15  | 2 mean, time index, Loss function = CRPScore                                               | Q4 2022 / FY 2023   |
| NGBoost  | 16  | 2 mean, time index, Loss function = LogScore                                               | Q4 2022 / FY 2023   |

TabPFN
| ID  | Features                   | Split                                  |
|-----|----------------------------|----------------------------------------|
| 1   | P(t-96), 2 mean ws         | Q1 2022 / Q1 2023                       |
| 2   | P(t-96), 2 mean ws         | Q2 2022 / Q2 2023                       |
| 3   | P(t-96), 2 mean ws         | Q3 2022 / Q3 2023                       |
| 4   | P(t-96), 2 mean ws         | Q4 2022 / Q4 2023                       |
| 5   | P(t-96), 2 mean ws         | Q4 2022 / Q1 2023                       |
| 6   | P(t-96), 2 mean ws         | Q4 2022 / Q2 2023                       |
| 7   | P(t-96), 2 mean ws         | Q4 2022 / Q3 2023                       |
| 8   | P(t-96), 10 ws             | Q4 2022 / H1 2023                       |
| 9   | P(t-96), 10 ws             | Q4 2022 / H2 2023                       |
| 10  | P(t-96), 2 mean+10 ws      | Q4 2022 / H1 2023                       |
| 11  | P(t-96), 2 mean+10 ws      | Q4 2022 / H2 2023                       |
| 12  | (all)                      | Q4 2022 / H1 2023                       |
| 13  | (all)                      | Q4 2022 / H2 2023                       |
| 14  | P(t-96), 2 mean ws         | Q1 2022 / Q2 2023                       |
| 15  | P(t-96), 2 mean ws         | Q3 2022 / Q2 2023                       |
| 16  | P(t-96), 2 mean ws         | Q1 2022 / Q4 2023                       |
| 17  | P(t-96), 2 mean ws         | Q1 2022 / Q3 2023                       |
| 18  | P(t-96), 2 mean ws         | Q2 2022 / Q1 2023                       |
| 19  | P(t-96), 2 mean ws         | 2022.08.01 – 2022.12.31 / FY 2023       |
| 20  | P(t-96), 2 mean ws         | Q2 2022 / Q4 2023                       |
| 21  | P(t-96), 2 mean ws         | Q3 2022 / Q1 2023                       |
| 22  | P(t-96), 2 mean ws         | Q3 2022 / Q4 2023                       |
| 34  | P(t-96), 2 mean ws         | 2022-09-01 - 2022-12-31 / Q1 2023       |
| 35  | P(t-96), 2 mean ws         | 2022-08-01 - 2022-12-31 / Q1 2023       |
| 36  | P(t-96), 2 mean ws         | H2 2022 / Q1 2023                       |
| 37  | P(t-96), 2 mean ws         | FY 2022 / Q1 2023                       |
| 38  | power, all ws, time bin    | FY 2022 / Q1 2023                       |
| 39  | power, all ws, time bin    | FY 2022 / Q2 2023                       |
| 40  | power, all ws, time bin    | FY 2022 / Q3 2023                       |
| 41  | power, all ws, time bin    | FY 2022 / Q4 2023                       |
| 42  | power, mean ws             | FY 2022 / Q1 2023                       |
| 43  | power, mean ws             | FY 2022 / Q2 2023                       |
| 44  | power, mean ws             | FY 2022 / Q3 2023                       |
| 45  | power, mean ws             | FY 2022 / Q4 2023                       |
| 46  | power, ws at 10 loc        | FY 2022 / Q1 2023                       |
| 47  | power, ws at 10 loc        | FY 2022 / Q2 2023                       |
| 48  | power, ws at 10 loc        | FY 2022 / Q3 2023                       |
| 49  | power, ws at 10 loc        | FY 2022 / Q4 2023                       |
| 50  | power, all ws              | FY 2022 / Q1 2023                       |
| 51  | power, all ws              | FY 2022 / Q2 2023                       |
| 52  | power, all ws              | FY 2022 / Q3 2023                       |
| 53  | power, all ws              | FY 2022 / Q4 2023                       |
| 54  | power                      | FY 2022 / Q1 2023                       |
| 55  | power                      | FY 2022 / Q2 2023                       |
| 56  | power                      | FY 2022 / Q3 2023                       |
| 57  | power                      | FY 2022 / Q4 2023                       |
| 58  | power                      | Q4 2022 / H1 2023                       |
| 59  | power                      | Q4 2022 / H2 2023                       |


# Reproducability

## Baseline

The results of the baseline model can be found in the jupyter notebook "notebooks/022_models.ipynb". 
The values in the table indicate the Experiment ID with which the baseline model has to be run to reproduce table 5.1, 5.2

| Feature                | 2016–2022 Training, mean and quantiles | Q4 2022 Training, Mean
|------------------------|----------------|--------------|
| Power                  | 33             | 24           |
| Power, mean ws         | 32             | 25           |
| Power, ws at 10 loc    | 31             | 26           |
| Power, all ws          | 30             | 27           |
| Power, all ws, t-bin   | 29             | 28           |

