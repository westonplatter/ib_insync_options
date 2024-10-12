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
from tqdm import tqdm

from ib_insync_options.etl.facets import CoreFacetSchema, FacetCoreDfColumns

# from ib_insync_options.option import Option
from ib_insync_options.utils.date_utils import calc_effective_date_of_dt
from ib_insync_options.utils.formatting_utils import lower_camel_case_to_snake_case
from ib_insync_options.utils.list_utils import chunks
from ib_insync_options.utils.networking import get_ibkr_host_ip

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
    Example: parts= ["future", "CLK21"]
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

    @classmethod
    def gen_contract(cls, ib: IB, full_ticker: str, currency: str = "USD") -> Contract:
        """
        Helper method for generating IB contract objects from a human readable string.

        Args:
            ib (IB): ib_insync IB object
            full_ticker (str): human readable ticker string. Example: future:CLM21

        Returns:
            Contract: IB contract object
        """
        parts = full_ticker.split(":")
        contract_type = cls.lookup_contract_type(parts[0].lower())

        asset_class_to_gen_function = {
            "future": gen_contract_future,
        }

        if contract_type in asset_class_to_gen_function:
            contract = asset_class_to_gen_function[contract_type](
                parts, currency=currency
            )
        else:
            raise NotImplementedError(f"{contract_type} not implemented yet")

        ib.qualifyContracts(contract)
        return contract

    @classmethod
    def fetch_last(
        cls,
        ib: IB,
        contract: Contract,
        market_data_type: MarketDataType = MarketDataType.LIVE,
    ) -> Tuple[float, datetime]:
        """
        Fetches the last price and time (UTC) for a given contract.

        Args:
            ib (IB): ib_insync IB object
            contract (Contract): IB contract object

        Returns:
            tuple[float, datetime]: last price and datetime in UTC
        """
        ib.reqMarketDataType(market_data_type.value)
        [ticker] = ib.reqTickers(contract)
        return (ticker.last, ticker.time, ticker)

    @classmethod
    def get_option_chain_future(cls, ib: IB, product_symbol: str) -> pd.DataFrame:
        """
        Fetches the option chain for a given future, and formats it into a DataFrame.

        Args:
            ib (IB): ib_insync IB object
            product_symbol (str): futures ticker. For example, "ES" for E-mini S&P 500 futures.

        Returns:
            pd.DataFrame: df with columns: 	[exchange, underlyingConId, tradingClass, multiplier, expirations, strikes]
        """
        exchange = map_ticker_to_exchange(product_symbol)
        index = Index(product_symbol, exchange, "USD")
        ib.qualifyContracts(index)
        chains = ib.reqSecDefOptParams(product_symbol, "SMART", "IND", index.conId)
        df = util.df(chains)
        df["product_symbol"] = product_symbol
        # df = df.explode("expirations")
        # df = df.explode("strikes")
        # df = df.rename(columns={"expirations": "expiry", "strikes": "strike"})
        # df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
        # df = df.rename(columns=lower_camel_case_to_snake_case)
        # df = df.sort_values(by=["expiry", "strike"])
        # df = df.reset_index(drop=True)
        # return df
        return cls._post_process_option_chain(df)

    @classmethod
    def get_option_chain_currency_future(
        cls, ib: IB, contract_future: Future
    ) -> pd.DataFrame:
        """
        TODO(next)

        Args:
            ib (IB): ib_insync IB object
            contract_future (Future): qualified IB contract object

        Returns:
            pd.DataFrame: df with columns: 	[exchange, underlyingConId, tradingClass, multiplier, expirations, strikes]
        """
        # product_symbol = contract_future.symbol
        # exchange = map_ticker_to_exchange(product_symbol)
        # exchange = contract_future.exchange

        # index = Index(product_symbol, exchange, "USD")
        # ib.qualifyContracts(index)
        # chains = ib.reqSecDefOptParams(product_symbol, "SMART", "IND", index.conId)
        chains = ib.reqSecDefOptParams(
            underlyingSymbol=contract_future.symbol,
            futFopExchange=contract_future.exchange,
            underlyingSecType="FUT",
            underlyingConId=contract_future.conId,
        )

        df = util.df(chains)
        df["product_symbol"] = contract_future.symbol
        # df = df.explode("expirations")
        # df = df.explode("strikes")
        # df = df.rename(columns={"expirations": "expiry", "strikes": "strike"})
        # df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
        # df = df.rename(columns=lower_camel_case_to_snake_case)
        # df = df.sort_values(by=["expiry", "strike"])
        # df = df.reset_index(drop=True)
        # return df
        return cls._post_process_option_chain(df)

    @classmethod
    def _post_process_option_chain(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post processes an option chain DataFrame.

        Args:
            df (pd.DataFrame): df with columns: 	[exchange, underlyingConId, tradingClass, multiplier, expirations, strikes]

        Returns:
            pd.DataFrame: df with columns: 	[exchange, underlyingConId, tradingClass, multiplier, expirations, strikes]
        """
        df = df.explode("expirations")
        df = df.explode("strikes")
        df = df.rename(columns={"expirations": "expiry", "strikes": "strike"})
        df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
        df = df.rename(columns=lower_camel_case_to_snake_case)
        df = df.sort_values(by=["expiry", "strike"])
        df = df.reset_index(drop=True)
        return df

    @classmethod
    def get_option_chain_stock(cls, ib: IB, product_symbol: str) -> pd.DataFrame:
        stock = Stock(product_symbol, "SMART", "USD")
        ib.qualifyContracts(stock)
        chains = ib.reqSecDefOptParams(product_symbol, "", stock.secType, stock.conId)
        df = util.df(chains)
        df["product_symbol"] = product_symbol
        df = df.explode("expirations")
        df = df.explode("strikes")
        df = df.rename(columns={"expirations": "expiry", "strikes": "strike"})
        df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
        df = df.rename(columns=lower_camel_case_to_snake_case)
        df = df.sort_values(by=["expiry", "strike"])
        df = df.reset_index(drop=True)
        return df

    @classmethod
    def gen_future_options_from_df(
        cls, ib: IB, df: pd.DataFrame, request_chunk_size: int = 1200
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
        options = []
        for _, row in df.iterrows():
            trade_date = row["expiry"].strftime("%Y%m%d")
            symbol = row["product_symbol"]
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

        if len(options) > 500:
            logger.warning(
                f"Qualifying a large number of options (number = {len(options)}). This may take a long time."
            )

        contract_chunks = [x for x in chunks(options, request_chunk_size)]
        tqdm_desc = f"ibkr=gen-future-options-{len(options)}"
        qualified_options = []
        for chunk in tqdm(contract_chunks, desc=tqdm_desc):
            ib.qualifyContracts(*chunk)
            qualified_options.extend(chunk)
        return qualified_options

    @classmethod
    def gen_options_from_df(
        cls, ib: IB, df: pd.DataFrame, request_chunk_size: int = 1200
    ) -> List[Option]:
        """
        Generates a list of Option objects from a DataFrame.

        Args:
            ib (IB): ib_insync IB object
            df (pd.DataFrame): df with columns: [exchange, underlying_con_id, trading_class, multiplier, expiry, strike]
            request_chunk_size (int, optional): number of contracts to qualify at a time. Defaults to 1200.

        Returns:
            List[Option]: list of Option objects
        """
        options = []
        for _, row in df.iterrows():
            trade_date = row["expiry"].strftime("%Y%m%d")
            symbol = row["product_symbol"]
            strike = float(row["strike"])
            exchange = row["exchange"]
            multiplier = row["multiplier"]

            call = Option(
                symbol=symbol,
                lastTradeDateOrContractMonth=trade_date,
                strike=strike,
                right="C",
                currency="USD",
                tradingClass=row["trading_class"],
            )
            options.append(call)

            put = Option(
                symbol=symbol,
                lastTradeDateOrContractMonth=trade_date,
                strike=strike,
                right="P",
                currency="USD",
                tradingClass=row["trading_class"],
            )
            options.append(put)

        contract_chunks = [x for x in chunks(options, request_chunk_size)]
        tqdm_desc = f"ibkr=gen-options-{len(options)}"
        qualified_options = []
        for chunk in tqdm(contract_chunks, desc=tqdm_desc):
            ib.qualifyContracts(*chunk)
            qualified_options.extend(chunk)
        return qualified_options

    @classmethod
    def get_tickers_for_contracts(
        cls,
        ib: IB,
        contracts: List[Contract],
        market_data_type: MarketDataType = MarketDataType.LIVE,
        request_chunk_size: int = 50,
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
        ib.reqMarketDataType(market_data_type.value)

        contract_chunks = [x for x in chunks(contracts, request_chunk_size)]
        tqdm_desc = f"ibkr=fetching-tickers-{len(contracts)}"

        for contracts_chunk in tqdm(contract_chunks, desc=tqdm_desc):
            tickers = ib.reqTickers(*contracts_chunk)
            all_tickers.extend(tickers)

        return all_tickers

        # tickers = ib.reqTickers(*contracts)
        # return tickers

    @classmethod
    def gen_tickers_df(cls, tickers: List[Ticker]) -> pd.DataFrame:
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
            except Exception as e:
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

            # use IBKR model values for finx modeling
            row["finx_iv"] = row["iv"]
            row["finx_delta"] = row["delta"]
            row["finx_gamma"] = row["gamma"]
            row["finx_vega"] = row["vega"]
            row["finx_theta"] = row["theta"]

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
        return df

    # def enrich_tickers_df(df: pd.DataFrame, r=0.0) -> pd.DataFrame:
    #     df = df.copy()
    #     df["moneyness"] = df["strike"] / df[FacetCoreDfColumns.UNDERLYING_PRICE.value]

    #     # non bus days are counted as 0.5 days
    #     bus_days = 1 + np.busday_count(
    #         df.edt.values.astype("M8[D]"), df.expiry.values.astype("M8[D]")
    #     )
    #     df["dte_bday"] = bus_days + ((df["dte"] - bus_days) * 0.5)

    #     df["finx_close"] = df["last"]

    #     for i, row in df.iterrows():
    #         if np.isnan(row["finx_close"]) or row["finx_close"] == -1:
    #             continue

    #         T = row["dte"] / FUTURE_DAYS_PER_YEAR_US
    #         if T == 0.0:
    #             T = 0.5 / FUTURE_DAYS_PER_YEAR_US

    #         o = Option(
    #             S=row[FacetCoreDfColumns.UNDERLYING_PRICE.value],
    #             K=row["strike"],
    #             r=r,
    #             T=T,
    #             sigma=None,
    #             option_type=row["call_put"],
    #             days_per_year=FUTURE_DAYS_PER_YEAR_US,
    #         )
    #         # function calls will either be a numeric or a analytical function call
    #         iv = o.iv(opt_value=row["finx_close"])
    #         # df.loc[i, "finx_iv"] = iv
    #         df.loc[i, "iv"] = iv

    #         T_bdays = row["dte_bday"] / MARKET_DAYS_PER_YEAR_US_STOCK
    #         if T_bdays == 0.0:
    #             T_bdays = 0.5 / MARKET_DAYS_PER_YEAR_US_STOCK

    #         o = Option(
    #             S=row["underlying_price"],
    #             K=row["strike"],
    #             r=r,
    #             T=T_bdays,
    #             sigma=None,
    #             option_type=row["call_put"],
    #             days_per_year=MARKET_DAYS_PER_YEAR_US_STOCK,
    #         )
    #         # function calls will either be a numeric or a analytical function call
    #         iv = o.iv(opt_value=row["finx_close"])

    #         df.loc[i, FacetCoreDfColumns.IV_BDAY_FINX.value] = iv
    #         o.iv = iv

    #     return df


@dataclass
class IbkrDownloader:
    _initialized: bool = field(init=False, default=False)

    ib: Optional[IB] = field(init=True, default=None)
    market_data_type: MarketDataType = field(init=True, default=MarketDataType.LIVE)
    months_by_number: List[int] = field(
        init=True, default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    effective_date: datetime = field(init=True, default=None)
    underlying_symbol: str = field(init=True, default=None)
    dte_max: int = field(init=False, default=30)
    expiry_date_max: datetime = field(init=True, default=None)
    contract_months_to_fetch: List[str] = field(init=False, default_factory=lambda: [])
    qualified_contracts: List[Contract] = field(init=False, default_factory=lambda: [])
    option_chain_df: pd.DataFrame = field(
        init=False, default_factory=lambda: pd.DataFrame()
    )
    enriched_option_chain_df: pd.DataFrame = field(
        init=False, default_factory=lambda: pd.DataFrame()
    )
    underlying_df: pd.DataFrame = field(
        init=False, default_factory=lambda: pd.DataFrame()
    )
    pre_fetch_option_filters: List[dict] = field(init=False, default_factory=lambda: [])
    asset_category: str = field(init=False, default="future")

    def __post_init__(self):
        if self.ib is None:
            self.ib = self._create_ib_connection()

        if self.effective_date is None:
            self.effective_date = self._create_effective_date()

        if self.expiry_date_max is None:
            self.expiry_date_max = self._create_expiry_date_max()

        self._initialized = True

    def _create_ib_connection(self) -> IB:
        ib = IB()
        host_ip = get_ibkr_host_ip()
        ib.connect(host_ip, 7497, clientId=1)
        return ib

    def _create_effective_date(self) -> datetime:
        utc_now = datetime.now(pytz.utc)
        mt_now = utc_now.astimezone(pytz.timezone("US/Mountain"))
        dt = calc_effective_date_of_dt(mt_now)
        return datetime(dt.year, dt.month, dt.day, tzinfo=pytz.timezone("US/Mountain"))

    def _create_expiry_date_max(self) -> datetime:
        dt = self.effective_date + timedelta(days=self.dte_max)
        return datetime(dt.year, dt.month, dt.day, tzinfo=pytz.timezone("US/Mountain"))

    def _calc_contract_months_to_fetch(self) -> List[str]:
        # I used this for fetching EUR future contracts back in 2023. This might be useful for other EUR futures.
        result = []

        sd = self.effective_date
        ed = self.expiry_date_max

        counter = 0

        while sd < ed:
            year = sd.year
            month = sd.month

            if month in self.months_by_number:
                result.append(sd.strftime("%Y%m"))

                # get index of the current month in the months_by_number list
                index = self.months_by_number.index(month)
                # if the index is the last month in the list, set the next month to the first month in the list
                if index == len(self.months_by_number) - 1:
                    next_month = self.months_by_number[0]
                    next_year = year + 1
                else:
                    next_month = self.months_by_number[index + 1]
                    next_year = year
                next_month_str = f"{next_year}{next_month:02d}"
                result.append(next_month_str)

            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

            sd = datetime(year, month, 1, tzinfo=pytz.timezone("US/Mountain"))
            counter += 1

        unique_contract_months = list(set(result))
        unique_contract_months.sort()
        return unique_contract_months

    def _gen_ibkr_contracts_to_fetch(self) -> List[Contract]:
        # I used this for fetching EUR future contracts back in 2023. This might be useful for other EUR futures.
        result = []
        for year_month in self.contract_months_to_fetch:
            contract = Contract()
            contract.symbol = self.underlying_symbol
            contract.lastTradeDateOrContractMonth = year_month
            contract.secType = "FUT"
            contract.exchange = map_ticker_to_exchange(self.underlying_symbol)
            result.append(contract)
        return result

    def qualify_contracts(self, contracts: List[Contract]) -> List[Contract]:
        request_chunk_size = 500
        contract_chunks = [x for x in chunks(contracts, request_chunk_size)]

        tqdm_desc = f"ibkr=qualify-contracts-{len(contracts)}"

        qualified_contracts = []
        for chunk in tqdm(contract_chunks, desc=tqdm_desc):
            self.ib.qualifyContracts(*chunk)
            qualified_contracts.extend(chunk)

        return qualified_contracts

    def fetch_option_chain(self) -> pd.DataFrame:
        # initialize the Index for the underlying symbol. For example, ES => ES Index
        exchange = map_ticker_to_exchange(self.underlying_symbol)
        index = Index(self.underlying_symbol, exchange, "USD")
        self.ib.qualifyContracts(index)

        # fetch the option chain
        chains = self.ib.reqSecDefOptParams(
            self.underlying_symbol, "SMART", "IND", index.conId
        )
        df = util.df(chains)

        # transform the returned data into a df
        df = df.explode("expirations")
        df = df.explode("strikes")
        df = df.rename(columns={"expirations": "expiry", "strikes": "strike"})
        df["expiry"] = pd.to_datetime(df["expiry"], format="%Y%m%d", errors="coerce")
        df = df.rename(columns=lower_camel_case_to_snake_case)
        df = df.sort_values(by=["expiry", "strike"])
        df = df.reset_index(drop=True)

        df.underlying_con_id = df.underlying_con_id.astype(int)

        # set underlying_symbol
        df["underlying_symbol"] = self.underlying_symbol

        self.option_chain_df = df
        return df

    def gen_futures_tickers_df_for_option_chain(self) -> pd.DataFrame:
        contracts = []

        unique_underlying_con_ids = list(
            self.option_chain_df["underlying_con_id"].unique()
        )

        # create a contract for each underlying_con_id in the option_chain_df
        for con_id in unique_underlying_con_ids:
            contract = Contract(conId=con_id)
            contracts.append(contract)

        future_contracts = self.qualify_contracts(contracts)
        tickers = self.fetch_tickers_for_contracts(future_contracts)
        df = self.transform_future_tickers_to_df(tickers)

        return df

    def transform_future_tickers_to_df(self, tickers):
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

    def fetch_underlying_df(self) -> pd.DataFrame:
        underlying_df = self.gen_futures_tickers_df_for_option_chain()
        self.underlying_df = underlying_df
        return underlying_df

    def enrich_option_chain_df_with_underlying_tickers(self) -> pd.DataFrame:
        mdf = pd.merge(
            left=self.option_chain_df,
            right=self.underlying_df[["con_id", "close", "bid", "ask"]],
            left_on="underlying_con_id",
            right_on="con_id",
            how="outer",
        )

        # calc the close as the mid given we're more likely to have the bid and ask vs a close
        # mdf.close = mdf.close.astype("float64")
        mdf["close"] = (mdf["bid"] + mdf["ask"]) / 2

        mdf.strike = mdf.strike.astype("float64")
        mdf.bid = mdf.bid.astype("float64")
        mdf.ask = mdf.ask.astype("float64")

        mdf["moneyness"] = round((mdf.strike / mdf.close) * 100.0, 2)

        self.enriched_option_chain_df = mdf
        return mdf

    def filter_option_chain(self, override_filters: List[dict] = None) -> pd.DataFrame:
        df = self.enriched_option_chain_df.copy()

        filters = self.pre_fetch_option_filters

        if override_filters is not None:
            logger.debug(f"Using override filters: {override_filters}")
            filters = override_filters

        for _filter in filters:
            for key, value in _filter.items():
                if key == "moneyness_gte":
                    df = df[df["moneyness"] >= value]
                elif key == "moneyness_lte":
                    df = df[df["moneyness"] <= value]
                elif key == "modulus_eq":
                    df = df[df["strike"] % value == 0]

        # filter the df to only include contracts that expire before the expiry_date_max
        df = df[df.expiry <= self.expiry_date_max.replace(tzinfo=None)]

        # remove the EC{symbol} contracts
        df = df[~df.trading_class.str.startswith("EC")]

        return df.copy()

    def fetch_future_options_from_df(
        self, option_chain_df: pd.DataFrame = None, request_chunk_size: int = 1000
    ) -> List[FuturesOption]:
        """
        Generates a list of IB qualified FuturesOption objects from a DataFrame.

        Args:
            ib (IB): ib_insync IB object
            option_chain_df (pd.DataFrame): df with columns: [exchange, underlying_con_id, trading_class, multiplier, expiry, strike]
            request_chunk_size (int, optional): number of contracts to qualify at a time. Defaults to 1200.

        Returns:
            List[FuturesOption]: list of FuturesOption objects

        """
        if option_chain_df is None:
            df = self.filter_option_chain()
        else:
            df = option_chain_df

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
            logger.warning(
                f"Qualifying a large number of options (number = {len(options)}). This may take a long time."
            )

        contract_chunks = [x for x in chunks(options, request_chunk_size)]
        tqdm_desc = f"ibkr=fetch-future-options-{len(options)}"
        qualified_options = []
        for chunk in tqdm(contract_chunks, desc=tqdm_desc):
            self.ib.qualifyContracts(*chunk)
            qualified_options.extend(chunk)

        return qualified_options

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

    # @pa.check_types
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


@dataclass
class IbkrDownloaderGold(IbkrDownloader):
    underlying_symbol: str = "GC"
    pre_fetch_option_filters: List[dict] = field(
        init=False,
        default_factory=lambda: [
            {"moneyness_gte": 95.0},
            {"moneyness_lte": 120.0},
            {"modulus_eq": 25.0},
        ],
    )
    dte_max: int = 400


@dataclass
class IbkrDownloaderCrude(IbkrDownloader):
    underlying_symbol: str = "CL"
    pre_fetch_option_filters: List[dict] = field(
        init=False,
        default_factory=lambda: [
            {"moneyness_gte": 90.0},
            {"moneyness_lte": 110.0},
            {"modulus_eq": 0.5},
        ],
    )
    dte_max: int = 90


@dataclass
class IbkrDownloaderES(IbkrDownloader):
    underlying_symbol: str = "ES"
    pre_fetch_option_filters: List[dict] = field(
        init=False,
        default_factory=lambda: [
            {"moneyness_gte": 90.0},
            {"moneyness_lte": 110.0},
            {"modulus_eq": 10.0},
        ],
    )
    dte_max: int = 35


class IbkrDownloaderNQ(IbkrDownloader):
    underlying_symbol: str = "NQ"
    pre_fetch_option_filters: List[dict] = field(
        init=False,
        default_factory=lambda: [
            {"moneyness_gte": 90.0},
            {"moneyness_lte": 110.0},
            {"modulus_eq": 25.0},
        ],
    )
    dte_max: int = 35
