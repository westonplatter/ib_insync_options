from enum import Enum
from typing import Any

import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import Series


class FacetCoreDfColumns(str, Enum):
    PRICE_BID = "bid"
    PRICE_ASK = "ask"
    BID = "bid"
    ASK = "ask"

    OPTION_TYPE = "option_type"
    CALL_PUT = "call_put"
    EFFECTIVE_DATETIME = "edt"
    EXPIRY = "expiry"
    DTE = "dte"
    IV = "iv"  # vendor supplied implied volatility
    IV_ASK = "iv_ask"
    IV_BID = "iv_bid"
    IV_FINX = "finx_iv"  # finx implied volatility
    IV_BDAY_FINX = "finx_bday_iv"  # finx implied volatility using business days and 252 days per year
    # PRICE_MODEL = "price_model"
    PRICE_MODEL = "opt_price"  # TODO(next) check rest of codebase for "opt_price"
    STRIKE = "strike"
    VOLUME = "volume"
    UNDERLYING_PRICE = "underlying_price"
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"

    MULTIPLIER = "multiplier"
    RIGHT = "right"

    # source/vendor column identifiers
    SOURCE_COLUMN = "source_column"
    SOURCE_IBKR_CONID = "source_ibkr_conid"
    SOURCE_ID = "source_id"


class CoreFacetSchema(pa.DataFrameModel):
    edt: Series[pd.Timestamp] = pa.Field(coerce=True, nullable=False)
    symbol: Series[str] = pa.Field()
    last: Series[float] = pa.Field(nullable=True)
    bid: Series[float] = pa.Field(nullable=True)
    ask: Series[float] = pa.Field(nullable=True)
    strike: Series[float] = pa.Field(nullable=False)
    expiry: Series[pd.Timestamp] = pa.Field(coerce=True, nullable=False)
    dte: Series[int] = pa.Field(nullable=False)
    call_put: Series[str] = pa.Field(nullable=False)
    multiplier: Series[Any] = pa.Field(nullable=False)
    volume: Series[Any] = pa.Field(nullable=True)
    source_id: Series[Any] = pa.Field(nullable=False)
    delta: Series[float] = pa.Field(nullable=True)
    vega: Series[float] = pa.Field(nullable=True)
    gamma: Series[float] = pa.Field(nullable=True)
    theta: Series[float] = pa.Field(nullable=True)
    opt_price: Series[float] = pa.Field(nullable=True)
    iv: Series[float] = pa.Field(nullable=True)
    iv_ask: Series[float] = pa.Field(nullable=True)
    iv_bid: Series[float] = pa.Field(nullable=True)
    underlying_price: Series[float] = pa.Field(nullable=True)


# class CoreFacet:
#     @classmethod
#     def ensure_columns(cls, df):
#         # logger.debug(f"CoreFacet: checking columns in {list(df.columns)}")
#         # logger.debug(f"class = {cls.__name__}")
#         # logger.debug(f"expected = {cls._expected_columns()}")

#         expected_cols = set(cls._expected_columns())
#         actual_cols = set(df.columns)

#         missing_cols = list(expected_cols - actual_cols)
#         extra_cols = list(actual_cols - expected_cols)

#         if len(missing_cols) > 0:
#             logger.warning(f"CoreFacet: missing column: {missing_cols}")

#     @classmethod
#     def _expected_columns(cls):
#         cols = """
#             edt
#             symbol
#             last
#             bid
#             ask
#             strike
#             expiry
#             dte
#             call_put
#             multiplier
#             volume
#             delta
#             vega
#             gamma
#             theta
#             iv
#             underlying_price
#             source_id
#         """.strip().split(
#             "\n"
#         )
#         return [c.strip() for c in cols]

#     @classmethod
#     def set_index(cls, df):
#         """
#         Set index columns (for db insertion)

#         Args:
#             df (pd.DataFrame): df with `_expected_columns`

#         Returns:
#             pd.DataFrame: new df with index set
#         """
#         df.set_index(
#             [
#                 "edt",
#                 "source_id",
#             ],
#             drop=True,
#         )
