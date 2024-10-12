import pandas as pd

from ib_insync_options.utils.df_utils import df_get_max_rows_for


def test_df_get_max_rows_for():
    df = pd.DataFrame({"source_id": [1, 1, 1], "edt": [10, 12, 14]})
    df.set_index("edt", inplace=True)
    actual_df = df_get_max_rows_for(df)

    expected_df = pd.DataFrame({"source_id": [1], "edt": [14]})
    expected_df.set_index("edt", inplace=True)

    pd.testing.assert_frame_equal(actual_df, expected_df)
