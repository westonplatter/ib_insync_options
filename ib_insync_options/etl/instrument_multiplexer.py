import os
from enum import Enum
from typing import List

import pandas as pd
from loguru import logger
from sqlalchemy import text

from ib_insync_options.etl.db_ops import gen_engine, upsert_df
from ib_insync_options.etl.ibkr_etl import IbkrEtl
from ib_insync_options.utils.date_utils import gen_now_dt


class ColumnNames(Enum):
    UPDATED_AT = "updated_at"
    ASSET_CLASS = "asset_class"
    EFFECTIVE_DATETIME = "edt"


def if_log_sql(sql, method_name=None):
    value = os.environ.get("FINX_LOG_SQL", "0").lower()
    do_we_log_sql = value in ["true", "t", "1", "yes"]

    if do_we_log_sql:
        logger.debug(f"method_name: {method_name}. SQL => \n{sql}")


class InstrumentMultiplexer:
    """
    This class provides **class methods** for routing requests the appropriate etl adapters
    caching the instrument data.

    Honestly, it's the glue code for various services connectors - so expect hacks.
    """

    @classmethod
    def get_instrument(
        cls, engine, instrument_input: str, effective_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        """
        Main method for InstrumentMultiplexer.

        Args:
            engine (sqlalchemy.engine.Engine): db engine
            instrument_input (str): examples: SPY, y:SPY, yahoo:SPY, ibkr:fut:CL
            effective_date (pd.Timestamp): effective date

        Returns:
            pd.DataFrame: finx FacetCoreDf df

        """
        parts = instrument_input.split(":")

        if len(parts) == 1:
            adapter = "yahoo"
            ticker_parts = [parts[0]]
        else:
            adapter = parts[0]
            ticker_parts = parts[1:]
            # ticker parts [asset_class, ticker]

        if adapter == "polygon" or adapter == "p":
            return cls.polygon_get_option_pricer_df(
                engine, ticker_parts, effective_date
            )

        if adapter == "yahoo" or adapter == "y":
            # TODO: implement yahoo_get_option_pricer_df with engine
            return cls.yahoo_get_option_pricer_df(ticker_parts, effective_date)

        if adapter == "ibkr" or adapter == "ib":
            return cls.ibkr_get_option_pricer_df(engine, ticker_parts, effective_date)

    @classmethod
    def get_instrument_files(cls, engine, instrument_input: str) -> dict:
        """
        Returns a dict of cache files for a given instrument.

        Args:
            engine (sqlalchemy.engine.Engine): db engine
            instrument_input (str): examples: SPY, y:SPY, yahoo:SPY, ibkr:fut:CL

        Returns:
            dict: key/value dict of cache files. Key is date_str, value is full path.

        """
        logger.debug(f"get_instrument_files: {instrument_input}")

        parts = instrument_input.split(":")

        if len(parts) == 1:
            adapter = "yahoo"
            ticker_parts = [parts[0]]
        else:
            adapter = parts[0]
            ticker_parts = parts[1:]
            # ticker parts [asset_class, ticker]

        # if adapter == "yahoo" or adapter == "y":
        #     # ticker = ticker_parts[0].upper()
        #     # return cls._yahoo_cache_list(ticker)
        #     raise NotImplementedError("Yahoo cache list not implemented")

        # if adapter == "polygon" or adapter == "p":
        #     ticker = ticker_parts[0].upper()
        #     return cls._polygon_cache_list(engine, ticker)

        if adapter == "ibkr" or adapter == "ib":
            asset_class = IbkrEtl.lookup_contract_type(ticker_parts[0])
            ticker = ticker_parts[1].upper()
            return cls._ibkr_cache_list(engine, asset_class, ticker)

    @classmethod
    def _ibkr_cache_exists(
        cls, engine, asset_class: str, ticker: str, effective_date: pd.Timestamp = None
    ):
        """
        Returns True if cache exists for the given asset_class, ticker, and effective_date.

        Args:
            engine (sqlalchemy.engine.Engine): db engine
            asset_class (str): asset class
            ticker (str): ticker
            effective_date (pd.Timestamp): effective date

        Returns:
            bool: True if cache exists for the given asset_class, ticker, and effective_date

        """
        edt_str = effective_date.strftime("%Y-%m-%d")
        sql = """
            SELECT COUNT(*) AS count
            FROM ibkr_option_core_facet
            WHERE 
                asset_class = :asset_class
                AND symbol = :symbol
                AND DATE(edt) = :edt_str
        """
        with engine.connect() as conn:
            df = pd.read_sql(
                sql=text(sql),
                con=conn,
                params={
                    "asset_class": asset_class,
                    "symbol": ticker,
                    "edt_str": edt_str,
                },
            )
        return df["count"].values[0] > 0

    @classmethod
    def _ibkr_cache_read(
        cls, engine, asset_class: str, ticker: str, effective_date: pd.Timestamp = None
    ):
        """
        Fetch a Finx FacetCoreDf df with the latest data for the given asset_class, ticker, and effective_date.

        Args:
            engine (sqlalchemy.engine.Engine): db engine
            asset_class (str): asset class
            ticker (str): ticker
            effective_date (pd.Timestamp): effective date

        Returns:
            pd.DataFrame: Finx FacetCoreDf df

        """
        edt_str = effective_date.strftime("%Y-%m-%d")

        sql = """
            WITH ranked_ibkr_option_core_facet AS (
                SELECT 
                    ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY edt DESC) AS sql_row_num, 
                    *
                FROM ibkr_option_core_facet
                WHERE
                    asset_class = :asset_class
                    AND symbol = :symbol 
                    AND DATE(edt) = :edt_str
            )

            SELECT *
            FROM ranked_ibkr_option_core_facet
            WHERE sql_row_num = 1;
        """
        if_log_sql(sql, method_name="_ibkr_cache_read")

        with engine.connect() as conn:
            df = pd.read_sql(
                sql=text(sql),
                con=conn,
                params={
                    "asset_class": asset_class,
                    "symbol": ticker,
                    "edt_str": edt_str,
                },
            )
            df.drop(columns=["sql_row_num"], inplace=True)
        return df

    @classmethod
    def _ibkr_cache_write(
        cls,
        asset_class: str,
        ticker: str,
        effective_date: pd.Timestamp,
        df: pd.DataFrame,
    ):
        df[ColumnNames.UPDATED_AT.value] = gen_now_dt(time_zone_str="US/Mountain")
        df[ColumnNames.ASSET_CLASS.value] = asset_class

        engine = gen_engine()
        df = df.set_index(
            [
                "edt",
                "source_id",
            ],
            drop=True,
        )
        table_name = "ibkr_option_core_facet"
        upsert_df(engine, df, table_name)

        # hack - there's a better way to do this
        engine.pool.dispose()

        return f"wrote {ticker} to ibkr_option_core_facet with {len(df)} rows"

    @classmethod
    def ibkr_get_option_pricer_df(
        cls, engine, ticker_parts: List[str], effective_date: pd.Timestamp = None
    ) -> pd.DataFrame:
        """
        Returns a Finx FacetCoreDf df.
        """
        logger.debug(f"IBKR ticker_parts: {ticker_parts}")
        asset_class = IbkrEtl.lookup_contract_type(ticker_parts[0])
        ticker = ticker_parts[1].upper()
        df = cls._ibkr_cache_read(engine, asset_class, ticker, effective_date)
        return df

    @classmethod
    def _ibkr_cache_list(cls, engine, asset_class: str, ticker: str):
        """
        Returns key/value dict of available data. Key is date_str, value is "edt" value.

        Args:
            engine (sqlalchemy.engine.Engine): db engine
            asset_class (str): asset class
            ticker (str): ticker

        Returns:
            dict: key/value dict of available data. Key is date_str, value is "edt" value

        """
        sql = """
            SELECT DISTINCT DATE(edt) AS edt 
            FROM ibkr_option_core_facet 
            WHERE
                asset_class = :asset_class
                AND symbol = :symbol
            ORDER BY edt
        """
        with engine.connect() as conn:
            df = pd.read_sql(
                sql=text(sql),
                con=conn,
                params={"asset_class": asset_class, "symbol": ticker},
            )
        return {str(v): v for v in df["edt"].tolist()}
