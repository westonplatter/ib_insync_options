# import re

# import pandas as pd


# def snake_case_columns(df: pd.DataFrame):
#     df.columns = [x.strip().lower().replace(" ", "_") for x in df.columns]
#     return df


# def lower_camel_case_to_snake_case(s):
#     return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


# def lower_camel_case_to_snake_case_columns(df):
#     df.columns = [lower_camel_case_to_snake_case(x) for x in df.columns]
#     return df


# def convert_to_snake_case(s):
#     s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
#     s = s.replace(" ", "")
#     return s


# def convert_snake_case_to_capitalized(s):
#     return s.replace("_", " ").title().replace(" ", "")


from ib_insync_options.utils.formatting_utils import (
    convert_snake_case_to_capitalized,
    convert_to_snake_case,
    lower_camel_case_to_snake_case,
    lower_camel_case_to_snake_case_columns,
    snake_case_columns,
)

def test_convert_to_snake_case():
    assert convert_to_snake_case("HelloWorld") == "hello_world"
