import pandas as pd

from ib_insync_options.utils.date_utils import (
    gen_cme_crude_contract,
    gen_cme_crude_product_code,
)


def test_gen_cme_crude_contract():
    ed = pd.Timestamp("2024-05-16")
    number = 1
    result = gen_cme_crude_contract(ed, number)
    assert result == pd.Timestamp("2024-05-21")


def test_gen_cme_crude_contract_end_of_month():
    ed = pd.Timestamp("2024-07-27")
    number = 1
    result = gen_cme_crude_contract(ed, number)
    assert result == pd.Timestamp("2024-08-20")


def test_gen_cme_crude_product_code():
    ed = pd.Timestamp("2024-04-16")
    number = 1
    result = gen_cme_crude_product_code(ed, number)
    assert result == "K24"


def test_gen_cme_crude_product_code_month_end():
    ed = pd.Timestamp("2024-04-26")
    number = 1
    result = gen_cme_crude_product_code(ed, number)
    assert result == "K24"


def test_gen_cme_crude_product_code_month_end():
    ed = pd.Timestamp("2024-04-26")
    number = 1
    result = gen_cme_crude_product_code(ed, number)
    assert result == "M24"
