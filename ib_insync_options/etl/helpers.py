from datetime import date

import pandas as pd
from loguru import logger


def filter_for_near_dated_within_bps(
    df: pd.DataFrame, last_price: float, max_dte: int = 30, price_diff_bps: float = 500
):
    """Filter for near-dated option rows. Written for IBKR options chain df.

    df is expected to have columns,
    - [ ] expiry
    - [ ] strike

    Args:
        df (pd.DataFrame): df
        last_price (float): instrument last price
        max_dte (int, optional): max Days to Expiration, dte. Defaults to 30.
        price_diff_bps (float, optional): price diff in basis points (bps). Defaults to 500.

    Returns:
        pd.DataFrame: subset of df that meets filter criteria

    """
    assert "expiry" in df.columns
    assert "strike" in df.columns

    # filter out strikes that are too far away from last price
    last_price_diff = round(last_price * price_diff_bps / 10_000, 2)

    logger.debug(
        f"last_price={last_price}, percent_diff={price_diff_bps/100}. Results in +/- {last_price_diff}"
    )

    min_strike, max_strike = [
        last_price - last_price_diff,
        last_price + last_price_diff,
    ]
    min_strike = round(min_strike, 2)
    max_strike = round(max_strike, 2)

    logger.debug(f"min_strike={min_strike}")
    logger.debug(f"max_strike={max_strike}")

    logger.debug(
        f"last_price={last_price}, percent_diff_bps={price_diff_bps}. Results in +/- {last_price_diff} ({min_strike}, {max_strike})"
    )

    df = df.copy()

    df = df[(df.strike > min_strike) & (df.strike < max_strike)]

    # filter out dates past a certain date
    max_expiry = df.expiry.min() + pd.Timedelta(days=max_dte)
    df = df[df.expiry < max_expiry]

    df = df[df["expiry"] > pd.to_datetime(date.today() + pd.Timedelta(days=1))]

    return df


def filter_for_monthlies_within_bps(
    df: pd.DataFrame, last_price: float, price_diff_bps: float = 500, max_dte: int = 90
):
    """Filter for long-dated, monthly option rows. Written for IBKR options chain df.

    df is expected to have columns,
    - [ ] expiry
    - [ ] strike
    - [ ] trading_class

    Args:
        df (pd.DataFrame): df
        last_price (float): instrument last price
        max_dte (int, optional): max Days to Expiration, dte. Defaults to 90.
        price_diff_bps (float, optional): price diff in basis points (bps). Defaults to 500.

    Returns:
        pd.DataFrame: subset of df that meets filter criteria

    """
    df = df.copy()

    # logger.debug(f"last_price = {last_price}")
    # filter out strikes that are too far away from last price
    last_price_diff = round(last_price * price_diff_bps / 10_000, 2)
    min_strike = round(last_price - last_price_diff, 2)
    # logger.debug(f"min_strike={min_strike}")
    max_strike = round(last_price + last_price_diff, 2)
    # logger.debug(f"max_strike={max_strike}")

    logger.debug(
        f"last_price={last_price}, percent_diff_bps={price_diff_bps}. Results in +/- {last_price_diff} ({min_strike}, {max_strike})"
    )

    df = df[(df.strike > min_strike) & (df.strike < max_strike)]
    # logger.debug(f"df.shape={df.shape}")

    # exclude all rows that end with a digit in the trading class
    all_rows_without_digits = ~df.trading_class.str.contains(r"\d{1}$")
    df = df[all_rows_without_digits]
    # logger.debug(f"df.shape={df.shape}")

    # filter out dates past a certain date
    max_expiry = df.expiry.min() + pd.Timedelta(days=max_dte)
    df = df[df.expiry < max_expiry]
    df = df[df["expiry"] > pd.to_datetime(date.today() + pd.Timedelta(days=1))]
    # logger.debug(f"df.shape={df.shape}")

    return df
