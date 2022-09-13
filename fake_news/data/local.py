import pandas as pd
import os


def get_data()-> pd.DataFrame :
    """
    Get the raw data from the local desk
    """
    path = os.path.join(
    os.environ.get("LOCAL_DATA_PATH"),
    "data.csv")

    df = pd.read_csv(path)

    return df
