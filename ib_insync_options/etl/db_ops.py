import os
import uuid

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text


def gen_engine():
    """
    Generate a sqlalchemy engine for the ib_insync_options db.

    Returns:
        sqlalchemy.engine.Engine: engine
    """
    db_name = os.environ.get("IB_INSYNC_OPTION_DB_NAME", "ib_insync_options_dev")
    db_user = os.environ.get("IB_INSYNC_OPTION_DB_USER", "postgres")
    db_password = os.environ.get("IB_INSYNC_OPTION_DB_PASSWORD", None)
    db_host = os.environ.get("IB_INSYNC_OPTION_DB_HOST", "localhost")
    db_port = os.environ.get("IB_INSYNC_OPTION_DB_PORT", "5432")
    db_connection_string = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    return create_engine(db_connection_string)


def upsert_df(engine: sqlalchemy.engine.Engine, df: pd.DataFrame, table_name: str):
    """
    Credit goes @pedrovgp for https://stackoverflow.com/a/69617559/665578

    Implements the equivalent of pd.DataFrame.to_sql(..., if_exists='update')
    (which does not exist). Creates or updates the db records based on the
    dataframe records.
    Conflicts to determine update are based on the dataframes index.
    This will set unique keys constraint on the table equal to the index names
    1. Create a temp table from the dataframe
    2. Insert/update from temp table into table_name
    Returns: True if successful
    """
    with engine.connect() as conn:
        # If the table does not exist, we should just use to_sql to create it
        if not conn.execute(
            text(
                f"""SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE  table_schema = 'public'
                AND    table_name   = '{table_name}');
                """
            )
        ).first()[0]:
            df.to_sql(table_name, engine)
            return True

    # If it already exists...
    temp_table_name = f"temp_{uuid.uuid4().hex[:6]}"
    df.to_sql(temp_table_name, engine, index=True)

    index = list(df.index.names)
    index_sql_txt = ", ".join([f'"{i}"' for i in index])
    columns = list(df.columns)
    headers = index + columns
    headers_sql_txt = ", ".join(
        [f'"{i}"' for i in headers]
    )  # index1, index2, ..., column 1, col2, ...

    # col1 = excluded.col1, col2=excluded.col2
    update_column_stmt = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in columns])

    constraint_name = f"{table_name}_unique_constraint_upsert"

    # For the ON CONFLICT clause, postgres requires that the columns have unique constraint
    query_pk = text(
        f"""
    ALTER TABLE "{table_name}" DROP CONSTRAINT IF EXISTS {constraint_name};
    ALTER TABLE "{table_name}" ADD CONSTRAINT {constraint_name} UNIQUE ({index_sql_txt});
    """
    )
    with engine.connect() as conn:
        conn.execute(query_pk)
        conn.commit()
        # logger.info(f"query_pk = {query_pk}")

        # Compose and execute upsert query
        query_upsert = text(
            f"""
            INSERT INTO "{table_name}" ({headers_sql_txt}) 
            SELECT {headers_sql_txt} FROM "{temp_table_name}"
            ON CONFLICT ({index_sql_txt}) DO UPDATE 
            SET {update_column_stmt};
        """
        )
        conn.execute(query_upsert)
        conn.commit()
        # logger.info(f"query_upsert = {query_upsert}")

        conn.execute(text(f"DROP TABLE {temp_table_name}"))
        conn.commit()
        # logger.debug(text(f"DROP TABLE {temp_table_name}"))

    return True
