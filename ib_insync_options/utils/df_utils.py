import pandas as pd


def df_get_max_rows_for(
    df: pd.DataFrame, groupby_col: str = "source_id", max_col: str = "edt"
):
    max_keys = df.reset_index().groupby(groupby_col, as_index=False).max()[max_col]
    return df.loc[max_keys]
