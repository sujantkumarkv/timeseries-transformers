import json
from typing import List
import pandas as pd
from pathlib import Path
import numpy as np


def add_lag_features(
    dataframe: pd.DataFrame,
    col_names: List,
    contexts: List,
    fill_na=True,
):
    """
    Computes simple lag features
    :param dataframe:
    :param col_names:
    :param contexts:
    :param fill_na:
    :return:
    """

    new_cols = []

    for context in contexts:
        for col_name in col_names:
            lag_col_name = f"{col_name}_lag_{context}"
            dataframe[lag_col_name] = dataframe[col_name].shift(periods=context)
            new_cols.append(lag_col_name)

    if fill_na:
        dataframe[new_cols] = dataframe[new_cols].fillna(0)

    return dataframe, new_cols


def process_df(temperature_df: pd.DataFrame, streamflow_df: pd.DataFrame, target_col: str = "streamflow"):
    """
    :param temperature_df:
    :param streamflow_df:
    :param target_col:
    :return:
    """
    # Calculate the average temperature
    temperature_df['temp'] = (temperature_df['maxtemp'] + temperature_df['mintemp']) / 2

    streamflow_df, lag_cols = add_lag_features(
        streamflow_df, col_names=[target_col], contexts=[1] #1 day of previous streamflow context
    )

    # return dataframe, new_cols
    temperature_df.drop(columns=['maxtemp', 'mintemp'], inplace=True)
    return temperature_df, streamflow_df


if __name__ == "__main__":

    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--csv_path")
    # parser.add_argument("--out_path")
    # parser.add_argument("--config_path")
    # args = parser.parse_args()

    # Get the absolute path to the directory where the script is located
    script_dir = Path(__file__).parent.absolute()

    # Construct the absolute path to the data directory
    data_dir = script_dir.parent.parent / 'data'

    # Load the data from each CSV file
    temperature_df = pd.read_csv(data_dir / 'csv/temperature.csv')
    waterlevel_df = pd.read_csv(data_dir / 'csv/waterlevel.csv')
    streamflow_df = pd.read_csv(data_dir / 'csv/streamflow.csv')
    # process maxtemp/mintemp cols
    temperature_df, streamflow_df = process_df(temperature_df, streamflow_df)
    # Merge the dataframes on the 'timestamp' column
    merged_df = temperature_df.merge(waterlevel_df, on='timestamp').merge( 
        streamflow_df, on='timestamp')

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv('./data/data.csv', index=False)

    # data.to_csv(args.out_path, index=False)

    config = {
        "features": ["temp", "waterlevel"],
        "target": "streamflow",
        "lag_features": ["streamflow_lag_1"]
    }

    # with open(args.config_path, "w") as f:
    #     json.dump(config, f, indent=4)
