"""Unit tests for deterministic ForensicAccountingGate and LocalFinancialStatementClient."""

from hedge_fund.forensic.gate import ForensicAccountingGate
from hedge_fund.forensic.models import (
    AuditAndGovernanceNotes,
    FinancialStatementDossier,
    FinancialStatementPeriod,
)


def _clean_period(year: str, **overrides) -> FinancialStatementPeriod:
    data = dict(
        report_period=f"{year}-12-31",
        filing_date=f"{int(year)+1}-04-15",
        market_cap=100000.0,
        revenue=20000.0,
        gross_profit=16000.0,
        operating_income=10000.0,
        net_income=7500.0,
        minority_interest_income=200.0,
        interest_income=450.0,
        interest_expense=10.0,
        operating_cash_flow=8200.0,
        capex=500.0,
        free_cash_flow=7700.0,
        cash_and_equivalents=18000.0,
        interest_bearing_debt=0.0,
        accounts_receivable=200.0,
        inventory=4000.0,
        goodwill=0.0,
        intangible_and_dev_assets=500.0,
        total_assets=30000.0,
        total_liabilities=6000.0,
        shareholders_equity=24000.0,
        current_assets=22200.0,
        current_liabilities=5000.0,
        shares_diluted=1000.0,
        gross_margin=0.80,
        net_margin=0.375,
        return_on_equity=0.31,
        debt_to_equity=0.0,
        current_ratio=4.44,
        price_to_earnings_ratio=15.0,
    )
    data.update(overrides)
    return FinancialStatementPeriod(**data)


def test_clean_white_horse_passes_with_zero_flags():
    dossier = FinancialStatementDossier(
        ticker="CLEAN",
        periods=[_clean_period(str(y)) for y in (2023, 2022, 2021, 2020)],
    )
    res = ForensicAccountingGate().evaluate(dossier)
    assert res.fail_closed is False
    assert res.risk_cap == "NONE"
    assert res.flag_codes == []


def test_unbalanced_balance_sheet_and_qualified_audit_fail_closed():
    broken = _clean_period("2023", total_assets=30500.0)  # 500M gap
    dossier = FinancialStatementDossier(
        ticker="BROKEN",
        periods=[broken, _clean_period("2022")],
        notes_and_audit=AuditAndGovernanceNotes(
            audit_opinion="Qualified Opinion",
            external_guarantee_to_equity=0.38,
        ),
    )
    res = ForensicAccountingGate().evaluate(dossier)
    assert res.fail_closed is True
    assert res.risk_cap == "FAIL_CLOSED"
    assert set(res.flag_codes) >= {
        "ACCOUNTING_IDENTITY_BROKEN",
        "NON_STANDARD_AUDIT_OPINION",
        "EXCESSIVE_EXTERNAL_GUARANTEE",
    }


def test_cash_debt_double_high_and_low_yield_triggers_bearish_cap():
    fraud = _clean_period(
        "2023",
        cash_and_equivalents=15000.0,  # 50% assets
        interest_bearing_debt=10000.0,  # 33% assets
        interest_income=110.0,  # 0.73% implied yield
        interest_expense=600.0,
        operating_cash_flow=3000.0,
        free_cash_flow=1000.0,
    )
    dossier = FinancialStatementDossier(
        ticker="DOUBLEHIGH",
        periods=[fraud, fraud],
    )
    res = ForensicAccountingGate().evaluate(dossier)
    assert res.fail_closed is False
    assert res.risk_cap == "BEARISH_CAP"
    assert "CASH_DEBT_DOUBLE_HIGH" in res.flag_codes
    assert "ABNORMAL_LOW_INTEREST_YIELD" in res.flag_codes
    assert "OCF_NET_INCOME_DIVERGENCE" in res.flag_codes


def test_goodwill_overhang_and_high_pledge_triggers_bearish_cap():
    gw = _clean_period("2023", goodwill=15000.0, shareholders_equity=24000.0)
    dossier = FinancialStatementDossier(
        ticker="GOODWILL",
        periods=[gw, gw],
        notes_and_audit=AuditAndGovernanceNotes(major_shareholder_pledge_ratio=0.87),
    )
    res = ForensicAccountingGate().evaluate(dossier)
    assert res.risk_cap == "BEARISH_CAP"
    assert "GOODWILL_OVERHANG" in res.flag_codes
    assert "PLEDGE_RATIO_CRITICAL" in res.flag_codes
