from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandera as pa
import pytz
from ib_insync import (
    IB,
    Contract,
    Future,
    FuturesOption,
    Index,
    Option,
    Stock,
    Ticker,
    util,
)
from loguru import logger
from loman import ComputationFactory, calc_node, input_node
from tqdm import tqdm

from ib_insync_options.etl.facets import CoreFacetSchema, FacetCoreDfColumns
from ib_insync_options.utils.date_utils import calc_effective_date_of_dt
from ib_insync_options.utils.dict_utils import gen_json
from ib_insync_options.utils.formatting_utils import lower_camel_case_to_snake_case
from ib_insync_options.utils.list_utils import chunks

FUTURE_DAYS_PER_YEAR_US = 365
MARKET_DAYS_PER_YEAR_US_STOCK = 252


def _helper_gen_dt(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day, tzinfo=pytz.timezone("US/Mountain"))


def map_month_code_to_number(code: str) -> str:
    """
    Converts a futures month code to its corresponding month integer.
    """
    codes = {
        "F": "01",  # January
        "G": "02",  # February
        "H": "03",  # March
        "J": "04",  # April
        "K": "05",  # May
        "M": "06",  # June
        "N": "07",  # July
        "Q": "08",  # August
        "U": "09",  # September
        "V": "10",  # October
        "X": "11",  # November
        "Z": "12",  # December
    }
    return codes.get(code.upper(), None)


def map_ticker_to_exchange(product: str) -> str:
    """Map future product ticker to its corresponding (CME) exchange.

    Args:
        product (str): product ticker

    Raises:
        ValueError: unknown product

    Returns:
        str: exchange

    """
    # equities
    if product in [
        "ES",
        "MES",
        "NQ",
        "MNQ",
        "RTY",
    ]:
        return "CME"

    # energy
    if product in ["CL", "MCL"]:
        return "NYMEX"

    # metals
    if product in ["GC", "MGC"]:
        return "COMEX"

    # bonds
    if product in [
        "ZT",
        "ZF",
        "ZN",
        "ZB",
        "UB",
    ]:
        return "CBOT"

    # grains
    if product in ["ZC", "ZS", "ZW", "ZM", "ZO"]:
        return "CBOT"

    # currencies
    if product in [
        "M6E",
        "EUR",
        "M6B",
        "GBP",
        "M6A",
        "AUD",
        "M6C",
        "CAD",
        "M6J",
        "JPY",
    ]:
        return "CME"

    raise ValueError(f"Unknown product: {product}")


def gen_contract_future(parts: List[str], currency: str = "USD") -> Contract:
    """
    Example: parts= ["future", "CLK21"].
    """

    offset = 0

    if "eur" in parts[1].lower():
        offset = 1

    ticker = parts[1][0 : 2 + offset]
    month_code = parts[1][2 + offset : 3 + offset]
    year = "20" + parts[1][3 + offset : 5 + offset]
    exp_date = f"{year}{map_month_code_to_number(month_code)}"
    exchange = map_ticker_to_exchange(ticker)
    contract = Future(ticker, exp_date, exchange, currency=currency)
    return contract


class MarketDataType(Enum):
    # https://interactivebrokers.github.io/tws-api/market_data_type.html
    LIVE = 1
    FROZEN = 2
    DELAYED = 3
    DELAYED_FROZEN = 4


class IbkrEtl:
    @classmethod
    def lookup_contract_type(cls, asset_class: str) -> str:
        asset_class = asset_class.lower()
        if asset_class in ["futures", "future", "fut"]:
            return "future"
        if asset_class in ["futures_option", "futures_option", "futopt"]:
            return "futures_option"
        if asset_class in ["option", "opt"]:
            return "option"
        if asset_class in ["stock", "stk"]:
            return "stock"
        raise ValueError(f"Unknown asset class: {asset_class}")


# @dataclass
# class IbkrDownloaderGold(IbkrDownloader):
#     underlying_symbol: str = "GC"
#     pre_fetch_option_filters: List[dict] = field(
#         init=False,
#         default_factory=lambda: [
#             {"moneyness_gte": 95.0},
#             {"moneyness_lte": 120.0},
#             {"modulus_eq": 25.0},
#         ],
#     )
#     dte_max: int = 400


# @dataclass
# class IbkrDownloaderCrude(IbkrDownloader):
#     underlying_symbol: str = "CL"
#     pre_fetch_option_filters: List[dict] = field(
#         init=False,
#         default_factory=lambda: [
#             {"moneyness_gte": 90.0},
#             {"moneyness_lte": 110.0},
#             {"modulus_eq": 0.5},
#         ],
#     )
#     dte_max: int = 90


# @dataclass
# class IbkrDownloaderES(IbkrDownloader):
#     underlying_symbol: str = "ES"
#     pre_fetch_option_filters: List[dict] = field(
#         init=False,
#         default_factory=lambda: [
#             {"moneyness_gte": 90.0},
#             {"moneyness_lte": 110.0},
#             {"modulus_eq": 10.0},
#         ],
#     )
#     dte_max: int = 35


# class IbkrDownloaderNQ(IbkrDownloader):
#     underlying_symbol: str = "NQ"
#     pre_fetch_option_filters: List[dict] = field(
#         init=False,
#         default_factory=lambda: [
#             {"moneyness_gte": 90.0},
#             {"moneyness_lte": 110.0},
#             {"modulus_eq": 25.0},
#         ],
#     )
#     dte_max: int = 35


def default_effective_date():
    utc_now = datetime.now(pytz.utc)
    mt_now = utc_now.astimezone(pytz.timezone("US/Mountain"))
    dt = calc_effective_date_of_dt(mt_now)
    return datetime(dt.year, dt.month, dt.day, tzinfo=pytz.timezone("US/Mountain"))


@ComputationFactory
class IbkrFutureOptionData:
    # nodes with default values
    asset_category = input_node(value="future")
    dte_max = input_node(value=30)
    effective_date = input_node(lambda: default_effective_date())
    # market_data_type = input_node(value=MarketDataType.LIVE)

    # input nodes without default values
    raw_option_chain = input_node()
    pre_fetch_option_filters = input_node()
    qualified_contracts = input_node()
    underlying_df = input_node()
    underlying_symbol = input_node()
    enriched_option_chain_df = input_node()

    @calc_node
    def expiry_date_max(effective_date, dte_max):
        dt = effective_date + timedelta(days=dte_max)
        return datetime(dt.year, dt.month, dt.day, tzinfo=pytz.timezone("US/Mountain"))

    @calc_node
    def option_chain_df(raw_option_chain, underlying_symbol):
        df = raw_option_chain.copy()

        df = df.explode("expirations")
        df = df.explode("strikes")
        df = df.rename(columns={"expirations": "expiry", "strikes": "strike"})
        df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")

        df = df.rename(columns=lower_camel_case_to_snake_case)

        df.strike = df.strike.astype("float64")

        df = df.sort_values(by=["expiry", "strike"])
        df = df.reset_index(drop=True)

        df.underlying_con_id = df.underlying_con_id.astype(int)

        # remove the EC{symbol} contracts
        df = df[~df.trading_class.str.startswith("EC")]

        # set underlying_symbol
        df["underlying_symbol"] = underlying_symbol

        return df

    @calc_node
    def filter_option_chain_by_max_expiry_date(option_chain_df, expiry_date_max):
        df = option_chain_df.copy()
        max_dt = pd.to_datetime(expiry_date_max)
        # only keep options expiring before the max date
        df = df[pd.to_datetime(df["expiry"]).dt.date <= max_dt.date()]
        return df

    @calc_node
    def underlying_df_calc_price(underlying_df):
        df = underlying_df.copy()

        # if bid AND ask are -1.0, use last
        mask_bid_ask_negative_one = (df["bid"] == -1.0) & (df["ask"] == -1.0)
        df.loc[mask_bid_ask_negative_one, "finx_price"] = df.loc[
            mask_bid_ask_negative_one, "last"
        ]

        # if bid OR ask are missing, use last
        mask_bid_ask_missing = df[["bid", "ask"]].isna().any(axis=1)
        df.loc[mask_bid_ask_missing, "finx_price"] = df.loc[
            mask_bid_ask_missing, "last"
        ]

        # if bid and ask have non-missing values, use the average
        mask_bid_ask_present = ~mask_bid_ask_missing & ~mask_bid_ask_negative_one
        df.loc[mask_bid_ask_present, "finx_price"] = (
            df.loc[mask_bid_ask_present, "bid"] + df.loc[mask_bid_ask_present, "ask"]
        ) / 2

        # if finx_price is nan, use close
        mask_finx_price_nan = df["finx_price"].isna()
        df.loc[mask_finx_price_nan, "finx_price"] = df.loc[mask_finx_price_nan, "close"]

        return df

    @calc_node
    def enrich_option_chain_df_with_underlying_tickers(
        filter_option_chain_by_max_expiry_date, underlying_df_calc_price
    ):
        _option_chain_df = filter_option_chain_by_max_expiry_date.copy()
        _underlying_df = underlying_df_calc_price.copy()
        mdf = pd.merge(
            left=_option_chain_df,
            right=_underlying_df[["con_id", "finx_price"]],
            left_on="underlying_con_id",
            right_on="con_id",
            how="outer",
        )

        # calc moneyness and round to 2 decimal places
        mdf["finx_moneyness"] = round((mdf.strike / mdf.finx_price) * 100.0, 2)

        return mdf

    @calc_node
    def apply_fetch_filters(
        enrich_option_chain_df_with_underlying_tickers, pre_fetch_option_filters
    ):
        df = enrich_option_chain_df_with_underlying_tickers.copy()
        _original_len = len(enrich_option_chain_df_with_underlying_tickers)

        print("")  # get logger.debug to print
        logger.debug(f"Applying pre fetch option filters: {pre_fetch_option_filters}")
        logger.debug(f"Pre Filters,  len => {len(df)}")

        for _filter in pre_fetch_option_filters:
            for key, value in _filter.items():
                if key == "moneyness_gte":
                    df = df[df["finx_moneyness"] >= value]
                elif key == "moneyness_lte":
                    df = df[df["finx_moneyness"] <= value]
                elif key == "strike_modulus_eq":
                    df = df[df["strike"] % value == 0]

        _post_filter_len = len(df)
        _percent_reduced = (_original_len - _post_filter_len) / _original_len * 100
        logger.debug(
            f"Post Filters, len => {_post_filter_len} ({_percent_reduced:.0f}% reduction)"
        )

        return df

    @calc_node
    # apply fetch filters as an arg so we know the order of operations
    def gen_qualified_fops_df(
        qualified_futures_options, apply_fetch_filters, option_chain_df
    ):
        qualified_futures_options_json = gen_json(qualified_futures_options)

        df = pd.DataFrame(qualified_futures_options_json)
        df = df.rename(columns=lower_camel_case_to_snake_case)
        df = df.rename(columns={"last_trade_date_or_contract_month": "expiry"})
        df["expiry"] = pd.to_datetime(df["expiry"])

        df = df.drop(
            columns=[
                "sec_id_type",
                "sec_id",
                "description",
                "issuer_id",
                "combo_legs_descrip",
                "combo_legs",
                "delta_neutral_contract",
                "include_expired",
                "primary_exchange",
                "sec_type",
            ]
        )

        mdf = pd.merge(
            left=option_chain_df.set_index(["expiry", "trading_class", "strike"]),
            right=df.set_index(["expiry", "trading_class", "strike"])[["con_id"]],
            left_index=True,
            right_index=True,
            how="inner",
        )
        mdf = mdf.reset_index()
        return mdf

    @calc_node
    def enrich_option_tickers_df_with_price(
        option_tickers_df, gen_qualified_fops_df, underlying_df_calc_price
    ):
        # left outer join the underlying_con_id via con_id
        mdf = pd.merge(
            left=option_tickers_df,
            right=gen_qualified_fops_df[["con_id", "underlying_con_id"]],
            left_on="source_id",
            right_on="con_id",
            how="outer",
        )
        mdf = mdf.rename(columns={"underlying_con_id": "source_underlying_id"})
        mdf = mdf.drop(columns=["con_id"])

        # left outer join the underlying price (via finx_price) via con_id
        mmdf = pd.merge(
            left=mdf,
            right=underlying_df_calc_price[["con_id", "finx_price"]],
            left_on="source_underlying_id",
            right_on="con_id",
            how="outer",
        )

        # set underlying_price to finx_price where underlying_price is None
        mmdf["underlying_price"] = mmdf["underlying_price"].fillna(mmdf["finx_price"])

        # drop intermediate and unused columns
        mmdf = mmdf.drop(columns=["finx_price", "con_id"])

        return mmdf


class IbkrDownloader:
    """
    Service Client for interacting with IBKR's TWS API.
    Specifically, downloading future option data.
    """

    def __init__(self, ib, market_data_type=MarketDataType.LIVE):
        self.ib = ib
        self.market_data_type = market_data_type

    def qualify_contracts(self, contracts: List[Contract]) -> List[Contract]:
        request_chunk_size = 500
        contract_chunks = [x for x in chunks(contracts, request_chunk_size)]

        tqdm_desc = f"ibkr=qualify-contracts-{len(contracts)}"

        qualified_contracts = []
        for chunk in tqdm(contract_chunks, desc=tqdm_desc):
            self.ib.qualifyContracts(*chunk)
            qualified_contracts.extend(chunk)

        return qualified_contracts

    def fetch_option_chain_for_underlying(
        self, underlying_symbol, sec_type: str = "FUT"
    ):
        # initialize the Index for the underlying symbol. For example, ES => ES Index
        exchange = map_ticker_to_exchange(underlying_symbol)
        index = Index(underlying_symbol, exchange, "USD")
        self.ib.qualifyContracts(index)

        # fetch the option chain
        raw_option_chain = self.ib.reqSecDefOptParams(
            underlying_symbol, "SMART", "IND", index.conId
        )
        df = util.df(raw_option_chain)
        return df

    def fetch_tickers_for_contracts(
        self, contracts: List[Contract], request_chunk_size: int = 500
    ) -> List[Any]:
        """
        Fetches tickers for a list of contracts.

        Args:
            ib (IB): ib_insync IB object
            contracts (List[Contract]): list of IB contract objects
            market_data_type (int, optional): market data type. Defaults to 1.
                Details here https://interactivebrokers.github.io/tws-api/market_data_type.html

        Returns:
            List[Any]: list of ib_insync Ticker objects

        """
        all_tickers = []
        self.ib.reqMarketDataType(self.market_data_type.value)

        contract_chunks = [x for x in chunks(contracts, request_chunk_size)]
        tqdm_desc = f"ibkr=fetch-tickers-{len(contracts)}"

        for contracts_chunk in tqdm(contract_chunks, desc=tqdm_desc):
            tickers = self.ib.reqTickers(*contracts_chunk)
            all_tickers.extend(tickers)

        return all_tickers

    def _transform_future_tickers_to_df(self, tickers):
        def transform_ticker(ticker):
            return {
                "con_id": ticker.contract.conId,
                "symbol": ticker.contract.symbol,
                "local_symbol": ticker.contract.localSymbol,
                "last_trade_date_or_contract_month": ticker.contract.lastTradeDateOrContractMonth,
                "multiplier": ticker.contract.multiplier,
                "exchange": ticker.contract.exchange,
                "currency": ticker.contract.currency,
                "last": ticker.last,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "high": ticker.high,
                "low": ticker.low,
                "close": ticker.close,
                "volume": ticker.volume,
                "open_interest": ticker.futuresOpenInterest,
            }

        ticker_dicts = [transform_ticker(ticker) for ticker in tickers]

        df = pd.DataFrame(ticker_dicts)
        df.con_id = df.con_id.astype(int)
        return df

    def gen_futures_tickers_df_for_option_chain(
        self, option_chain_df: pd.DataFrame
    ) -> pd.DataFrame:
        contracts = []

        unique_underlying_con_ids = list(option_chain_df["underlying_con_id"].unique())

        # create a contract for each underlying_con_id in the option_chain_df
        for con_id in unique_underlying_con_ids:
            contract = Contract(conId=con_id)
            contracts.append(contract)

        future_contracts = self.qualify_contracts(contracts)
        tickers = self.fetch_tickers_for_contracts(future_contracts)
        df = self._transform_future_tickers_to_df(tickers)

        return df

    def fetch_underlying_df(self, option_chain_df: pd.DataFrame) -> pd.DataFrame:
        underlying_df = self.gen_futures_tickers_df_for_option_chain(option_chain_df)
        return underlying_df

    def gen_qualified_futures_options(
        self, option_chain_df: pd.DataFrame, request_chunk_size: int = 1000
    ) -> List[FuturesOption]:
        """
        Generates a list of IB qualified FuturesOption objects from a DataFrame.

        Args:
            ib (IB): ib_insync IB object
            df (pd.DataFrame): df with columns: [exchange, underlying_con_id, trading_class, multiplier, expiry, strike]
            request_chunk_size (int, optional): number of contracts to qualify at a time. Defaults to 1200.

        Returns:
            List[FuturesOption]: list of FuturesOption objects

        """
        df = option_chain_df.copy()

        options = []
        for _, row in df.iterrows():
            trade_date = row["expiry"].strftime("%Y%m%d")
            symbol = row["underlying_symbol"]
            strike = float(row["strike"])
            exchange = row["exchange"]
            multiplier = row["multiplier"]

            call = FuturesOption(
                symbol=symbol,
                lastTradeDateOrContractMonth=trade_date,
                strike=strike,
                right="C",
                exchange=exchange,
                multiplier=multiplier,
                currency="USD",
                tradingClass=row["trading_class"],
            )
            options.append(call)

            put = FuturesOption(
                symbol=symbol,
                lastTradeDateOrContractMonth=trade_date,
                strike=strike,
                right="P",
                exchange=exchange,
                multiplier=multiplier,
                currency="USD",
                tradingClass=row["trading_class"],
            )
            options.append(put)

        if len(options) > 1501:
            logger.info(
                f"Qualifying a large number of options (number = {len(options)}). This may take a long time."
            )

        contract_chunks = [x for x in chunks(options, request_chunk_size)]
        tqdm_desc = f"ibkr=fetch-future-options-{len(options)}"
        qualified_options = []
        for chunk in tqdm(contract_chunks, desc=tqdm_desc):
            self.ib.qualifyContracts(*chunk)
            qualified_options.extend(chunk)

        return qualified_options

    def gen_option_tickers_df(
        self, tickers: List[Ticker]
    ) -> pa.typing.DataFrame[CoreFacetSchema]:
        """
        Generates a DataFrame from a list of Ticker objects.

        Args:
            tickers (List[Ticker]): ib_insync Ticker objects

        Returns:
            pd.DataFrame: df

        """
        # observation date for tickers if none is provided
        utc_now = datetime.now(pytz.utc)
        et_now = utc_now.astimezone(pytz.timezone("US/Eastern"))
        mt_now = utc_now.astimezone(pytz.timezone("US/Mountain"))
        effective_date = calc_effective_date_of_dt(et_now)

        rows = []
        for ticker in tickers:
            if ticker.contract is None:
                continue

            if ticker.contract.lastTradeDateOrContractMonth is None:
                continue

            row = {}
            row[FacetCoreDfColumns.EFFECTIVE_DATETIME.value] = mt_now
            row["symbol"] = ticker.contract.symbol
            row["last"] = ticker.marketPrice()
            row[FacetCoreDfColumns.BID.value] = ticker.bid
            row[FacetCoreDfColumns.ASK.value] = ticker.ask
            row[FacetCoreDfColumns.STRIKE.value] = ticker.contract.strike

            # NOTE the date varies between YYYYMMDD or YYYY-MM-DD
            strptime_format = (
                "%Y-%m-%d"
                if "-" in ticker.contract.lastTradeDateOrContractMonth
                else "%Y%m%d"
            )
            expiry_dt = datetime.strptime(
                ticker.contract.lastTradeDateOrContractMonth, strptime_format
            )
            row[FacetCoreDfColumns.EXPIRY.value] = expiry_dt.date()

            # NOTE use now if ticker.time is None
            observed_dt = effective_date if ticker.time is None else ticker.time

            # NOTE some rows have no expiry date due IBKR not having contract definitions for them
            # we skip these DTE calculations and later drop the rows
            try:
                row[FacetCoreDfColumns.DTE.value] = (
                    expiry_dt.date() - observed_dt.date()
                ).days
            except Exception:
                row[FacetCoreDfColumns.DTE.value] = None

            # row[FacetCoreDfColumns.RIGHT.value] = ticker.contract.right
            row[FacetCoreDfColumns.CALL_PUT.value] = ticker.contract.right.lower()[0]
            row[FacetCoreDfColumns.MULTIPLIER.value] = ticker.contract.multiplier
            row[FacetCoreDfColumns.VOLUME.value] = ticker.volume
            row[FacetCoreDfColumns.SOURCE_ID.value] = ticker.contract.conId

            mg = ticker.modelGreeks
            bg = ticker.bidGreeks
            ag = ticker.askGreeks
            row[FacetCoreDfColumns.DELTA.value] = None if mg is None else mg.delta
            row[FacetCoreDfColumns.VEGA.value] = None if mg is None else mg.vega
            row[FacetCoreDfColumns.GAMMA.value] = None if mg is None else mg.gamma
            row[FacetCoreDfColumns.THETA.value] = None if mg is None else mg.theta
            row[FacetCoreDfColumns.PRICE_MODEL.value] = (
                None if mg is None else mg.optPrice
            )

            row[FacetCoreDfColumns.IV.value] = None if mg is None else mg.impliedVol
            row[FacetCoreDfColumns.IV_ASK.value] = None if ag is None else ag.impliedVol
            row[FacetCoreDfColumns.IV_BID.value] = None if bg is None else bg.impliedVol
            row[FacetCoreDfColumns.UNDERLYING_PRICE.value] = (
                None if mg is None else mg.undPrice
            )

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.dropna(subset=[FacetCoreDfColumns.DTE.value])
        df.sort_values(
            by=[
                FacetCoreDfColumns.CALL_PUT.value,
                FacetCoreDfColumns.EXPIRY.value,
                FacetCoreDfColumns.STRIKE.value,
            ]
        )
        df.reset_index(drop=True)
        df.iv_ask = df.iv_ask.fillna(np.nan)
        df.iv_bid = df.iv_bid.fillna(np.nan)
        return df
