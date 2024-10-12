from datetime import datetime

import pandas as pd
import pytz


def gen_et_dt_close(dt: datetime) -> datetime:
    """
    gen et dt close

    Args:
        dt (datetime): _description_

    Returns:
        datetime: _description_
    """
    ts = pd.Timestamp(dt).tz_localize("US/Eastern")
    ts = ts.replace(hour=16)

    # dt = datetime(
    #     year=dt.year,
    #     month=dt.month,
    #     day=dt.day,
    #     hour=16,
    #     minute=0,
    #     second=0,
    #     microsecond=0,
    #     tzinfo=pytz.timezone('US/Eastern'))

    # return dt

    return ts
