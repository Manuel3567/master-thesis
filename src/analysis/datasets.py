import pandas as pd

def load_electricity(data_path: str = "../data", kind: str = "pandas"):
    """
    source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
    """
    df = pd.read_csv(f"{data_path}/LD2011_2014.txt", sep=";", decimal=",", index_col=0)

    df.index = pd.to_datetime(df.index)

    return df