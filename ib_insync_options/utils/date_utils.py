from datetime import date, datetime

import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay


def calc_effective_date_of_dt(
    dt: pd.Timestamp = None, market: str = None, instrument: str = None
):
    """Calculate the data effective date for a given pd.Timestamp.

    Args:
        dt (pd.Timestamp, optional): [description]. Defaults to None.
        market (str, optional): [description]. Defaults to None.
        instrument (str, optional): [description]. Defaults to None.

    Returns:
        pd.Timestamp: data effective date

    """
    if dt is None:
        dt = pd.Timestamp.today(tz="US/Mountain")

    if dt.weekday() == 5:
        dt = dt - pd.Timedelta(days=1)

    last_market_date = dt.date()

    return last_market_date


def gen_now_dt(time_zone_str: str = "US/Mountain") -> datetime:
    """
    Generate dt for now in specified timezone.

    Returns:
        datetime.datetime: dt

    """
    utc_now = datetime.now(pytz.utc)
    mt_now = utc_now.astimezone(pytz.timezone(time_zone_str))
    return mt_now


def gen_now_date_str(time_zone_str: str = "US/Mountain") -> str:
    """
    Generate today's date string.

    Returns:
        str: today's date string

    """
    utc_now = datetime.now(pytz.utc)
    mt_now = utc_now.astimezone(pytz.timezone(time_zone_str))
    return mt_now.strftime("%Y-%m-%d")


def map_month_number_to_futures_month_code(dt: pd.Timestamp) -> str:
    codes = {
        "01": "F",
        "02": "G",
        "03": "H",
        "04": "J",
        "05": "K",
        "06": "M",
        "07": "N",
        "08": "Q",
        "09": "U",
        "10": "V",
        "11": "X",
        "12": "Z",
    }
    month_code = codes.get(dt.strftime("%m"), None)
    year_str = dt.strftime("%Y")[2:]

    return f"{month_code}{year_str}"


def gen_cme_crude_contract(ed: pd.Timestamp, number: int = 1) -> date:
    """
    Gen termination date for crude based on effective date.

    Args:
        ed (pd.Timestamp): effective date
        number (int, optional): month number. 1 = front month. Defaults to 1.

    Returns:
        datetime.date: termination date

    """
    # https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html
    # Trading terminates 3 business day before the 25th calendar day of the month
    # prior to the contract month. If the 25th calendar day is not a business day,
    # trading terminates 4 business days before the 25th calendar day of the month
    # prior to the contract month.
    # https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.calendar.html

    # Parse the input date and find the first day of the contract month
    effective_date = pd.to_datetime(ed)

    def _calc_termination_date(ed: pd.Timestamp) -> pd.Timestamp:
        # Calculate the 25th day of the next month
        ed_month_first_day = pd.to_datetime(f"{ed.year}-{ed.month}-01")
        next_month_25th_day = ed_month_first_day + pd.DateOffset(
            days=25 - 1, months=number - 1
        )

        # Calc offset
        date_25th = pd.date_range(start=next_month_25th_day, periods=1, freq="B").date[
            0
        ]
        is_business_day = date_25th == next_month_25th_day.date()
        if is_business_day:
            termination_date = next_month_25th_day - BDay(3)
        else:
            termination_date = next_month_25th_day - BDay(4)
        return termination_date

    termination_date = _calc_termination_date(effective_date)

    if effective_date >= termination_date:
        updated_ts = effective_date + relativedelta(months=1)
        updated_ed = pd.Timestamp(f"{updated_ts.year}-{updated_ts.month}-01")
        termination_date = _calc_termination_date(updated_ed)

    return termination_date


def gen_cme_crude_product_code(ed, number: int) -> str:
    termination_date = gen_cme_crude_contract(ed, number=number)
    contract_month = termination_date + relativedelta(months=1)
    return map_month_number_to_futures_month_code(contract_month)
