# from datetime import datetime, timedelta
import datetime

import pandera as pa
import pytest
import pytz
from ib_insync import FuturesOption, OptionComputation, Ticker

from ib_insync_options.etl.facets import CoreFacetSchema
from ib_insync_options.etl.ibkr_etl import (  # IbkrDownloaderCrude,; IbkrDownloaderES,; IbkrDownloaderGold,; IbkrDownloaderNQ,; _helper_gen_dt,
    IbkrDownloader,
    map_month_code_to_number,
)
from ib_insync_options.utils.date_utils import calc_effective_date_of_dt


def gen_test_tickers():
    return [
        Ticker(
            contract=FuturesOption(
                conId=724326219,
                symbol="GC",
                lastTradeDateOrContractMonth="20240923",
                strike=2475.0,
                right="C",
                multiplier="100",
                exchange="COMEX",
                currency="USD",
                localSymbol="G4MU4 C2475",
                tradingClass="G4M",
            ),
            time=datetime.datetime(
                2024, 9, 22, 4, 32, 9, 779361, tzinfo=datetime.timezone.utc
            ),
            marketDataType=2,
            minTick=0.1,
            volume=0.0,
            close=116.3,
            halted=0.0,
            modelGreeks=OptionComputation(
                tickAttrib=0,
                impliedVol=0.22200803203321115,
                delta=0.999999999999999,
                optPrice=148.46913403414945,
                pvDividend=0.0,
                gamma=2.2491757106051506e-06,
                vega=0.00023428074982234648,
                theta=0.019109066483936742,
                undPrice=2623.5,
            ),
            bboExchange="110004",
        ),
        Ticker(
            contract=FuturesOption(
                conId=724325790,
                symbol="GC",
                lastTradeDateOrContractMonth="20240923",
                strike=2475.0,
                right="P",
                multiplier="100",
                exchange="COMEX",
                currency="USD",
                localSymbol="G4MU4 P2475",
                tradingClass="G4M",
            ),
            time=datetime.datetime(
                2024, 9, 22, 4, 32, 9, 780156, tzinfo=datetime.timezone.utc
            ),
            marketDataType=2,
            minTick=0.1,
            ask=0.2,
            askSize=77.0,
            last=0.1,
            lastSize=10.0,
            volume=54.0,
            high=0.2,
            low=0.1,
            close=0.4,
            halted=-1.0,
            modelGreeks=OptionComputation(
                tickAttrib=0,
                impliedVol=0.22200803203321115,
                delta=-1.9725361436775944e-05,
                optPrice=0.00016706841295041486,
                pvDividend=0.0,
                gamma=2.2491869360879966e-06,
                vega=0.00023428197597472264,
                theta=-0.00016706841295041486,
                undPrice=2623.5,
            ),
            bboExchange="110004",
            snapshotPermissions=3,
        ),
        Ticker(
            contract=FuturesOption(
                conId=724326178,
                symbol="GC",
                lastTradeDateOrContractMonth="20240923",
                strike=2500.0,
                right="C",
                multiplier="100",
                exchange="COMEX",
                currency="USD",
                localSymbol="G4MU4 C2500",
                tradingClass="G4M",
            ),
            time=datetime.datetime(
                2024, 9, 22, 4, 32, 9, 780478, tzinfo=datetime.timezone.utc
            ),
            marketDataType=2,
            minTick=0.1,
            volume=0.0,
            close=91.5,
            halted=-1.0,
            modelGreeks=OptionComputation(
                tickAttrib=0,
                impliedVol=0.19220245004595202,
                delta=0.999999999999999,
                optPrice=123.47453089110115,
                pvDividend=0.0,
                gamma=5.61791518357996e-06,
                vega=0.00049725133111167,
                theta=0.01481560774335187,
                undPrice=2623.5,
            ),
            bboExchange="110004",
            snapshotPermissions=3,
        ),
    ]


def test_map_month_code_to_number():
    code = "F"
    assert map_month_code_to_number(code) == "01"


# def test_ibkr_etl_downloader_init():
#     downloader = IbkrDownloader(ib="test_ib")
#     assert downloader.months_by_number == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#     ed = _helper_gen_dt(calc_effective_date_of_dt())
#     assert downloader.effective_date == ed


# def test_ibkr_etl_downloader_all_inits():
#     dt = calc_effective_date_of_dt()
#     ed = datetime.datetime(
#         dt.year, dt.month, dt.day, tzinfo=pytz.timezone("US/Mountain")
#     )

#     downloader_classes = [
#         IbkrDownloader,
#         IbkrDownloaderGold,
#         IbkrDownloaderCrude,
#         IbkrDownloaderES,
#         IbkrDownloaderNQ,
#     ]

#     for downloader_class in downloader_classes:
#         downloader = downloader_class(ib="test_ib")
#         assert downloader.effective_date == ed


def test_gen_option_tickers_df():
    mock_ib = "test_ib"
    downloader = IbkrDownloader(ib=mock_ib)
    tickers = gen_test_tickers()
    df = downloader.gen_option_tickers_df(tickers)

    try:
        CoreFacetSchema.validate(df)
    except pa.errors.SchemaError:
        pytest.fail("CoreFacetSchema validation raised SchemaError unexpectedly!")

    df_without_symbol = df.drop(columns=["symbol"])

    with pytest.raises(pa.errors.SchemaError):
        CoreFacetSchema.validate(df_without_symbol)
