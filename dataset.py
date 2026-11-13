import pandas as pd


def read_csv(path, *, options):
    return pd.read_csv(path, **options)
