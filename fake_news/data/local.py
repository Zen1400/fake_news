import pandas as pd
import os


def get_data()-> pd.DataFrame :
    """
    Get the raw data(has to have text column and
    true column as target whose values are 1 or 2) from the local desk
    """
    path = os.path.join(
    os.environ.get("LOCAL_DATA_PATH"),
    "final_data.csv")

    df = pd.read_csv(path)

    return df
