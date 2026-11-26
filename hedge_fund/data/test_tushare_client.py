"""Unit tests for TushareDataClient normalization, PIT filtering & reconciliation (mocked HTTP)."""

from hedge_fund.data.tushare_client import TushareDataClient, normalize_ashare_code


def test_normalize_ashare_code():
    assert normalize_ashare_code("600519") == "600519.SH"
    assert normalize_ashare_code("002450") == "002450.SZ"
    assert normalize_ashare_code("300750") == "300750.SZ"
    assert normalize_ashare_code("830799") == "830799.BJ"
    assert normalize_ashare_code("sz000858") == "000858.SZ"
    assert normalize_ashare_code("600519.SH") == "600519.SH"


def test_dedup_annual_enforces_point_in_time_and_update_flag():
    rows = [
        {"end_date": "20231231", "ann_date": "20240403", "update_flag": "0", "revenue": 100},
        {"end_date": "20231231", "ann_date": "20240403", "update_flag": "1", "revenue": 105},
        {"end_date": "20241231", "ann_date": "20250402", "update_flag": "1", "revenue": 200},  # future!
        {"end_date": "20240331", "ann_date": "20240425", "update_flag": "1", "revenue": 30},   # quarterly
    ]
    out = TushareDataClient._dedup_annual(rows, as_of_compact="20240501")
    assert list(out.keys()) == ["20231231"]
    assert out["20231231"]["revenue"] == 105


def test_tushare_enable_forensic_false_returns_none_dossier(tmp_path):
    client = TushareDataClient(token="dummy-token", cache_dir=tmp_path, enable_forensic=False)
    assert client.get_statement_dossier("600519.SH", "2024-05-01") is None
