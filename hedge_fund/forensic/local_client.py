"""Open, zero-cost local JSON adapter implementing DataClient + forensic dossier lookup.

Breaks the hard dependency on paid US-only FINANCIAL_DATASETS_API_KEY so any
public A-share, HK, or US 3-statement JSON dataset can be evaluated offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hedge_fund.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
    PriceSnapshot,
)
from hedge_fund.forensic.models import (
    AuditAndGovernanceNotes,
    FinancialStatementDossier,
    FinancialStatementPeriod,
)


class LocalFinancialStatementClient:
    """Point-in-time DataClient backed by standard open 3-statement JSON files."""

    def __init__(self, cases_by_ticker: dict[str, dict[str, Any]], currency: str = "CNY") -> None:
        self._cases = cases_by_ticker
        self._currency = currency

    @classmethod
    def from_dataset_file(cls, path: str | Path) -> "LocalFinancialStatementClient":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = {c["ticker"]: c for c in raw["cases"]}
        currency = "CNY" if "CNY" in str(raw.get("currency_unit", "CNY")) else "USD"
        return cls(cases, currency=currency)

    def get_statement_dossier(self, ticker: str, end_date: str, limit: int = 10) -> FinancialStatementDossier | None:
        """Return point-in-time full 3-statement dossier filed on or before *end_date*."""
        case = self._cases.get(ticker)
        if not case:
            return None
        filed = [
            FinancialStatementPeriod(**p)
            for p in case.get("periods", [])
            if p.get("filing_date", "9999-12-31") <= end_date[:10]
        ][:limit]
        notes = AuditAndGovernanceNotes(**(case.get("notes_and_audit") or {}))
        return FinancialStatementDossier(
            ticker=ticker,
            sector=case.get("sector"),
            industry=case.get("industry"),
            currency=self._currency,
            periods=filed,
            notes_and_audit=notes,
        )

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        dossier = self.get_statement_dossier(ticker, end_date, limit=limit)
        if dossier is None:
            return []
        rows: list[FinancialMetrics] = []
        for p in dossier.periods:
            shares = p.shares_diluted or 1.0
            dso = round((p.accounts_receivable / p.revenue) * 365.0, 2) if p.revenue else None
            icov = (
                round(p.operating_income / p.interest_expense, 2)
                if (p.operating_income is not None and p.interest_expense > 0)
                else None
            )
            ocf_ratio = (
                round(p.operating_cash_flow / p.current_liabilities, 3)
                if (p.current_liabilities and p.current_liabilities > 0)
                else None
            )
            cash_ratio = (
                round(p.cash_and_equivalents / p.current_liabilities, 3)
                if (p.current_liabilities and p.current_liabilities > 0)
                else None
            )
            debt_to_assets = round(p.total_liabilities / p.total_assets, 3) if p.total_assets > 0 else None
            rows.append(
                FinancialMetrics(
                    ticker=ticker,
                    report_period=p.report_period,
                    period=period,
                    currency=self._currency,
                    filing_date=p.filing_date,
                    market_cap=(p.market_cap * 1e6) if p.market_cap is not None else None,
                    price_to_earnings_ratio=p.price_to_earnings_ratio,
                    return_on_equity=p.return_on_equity,
                    gross_margin=p.gross_margin,
                    operating_margin=p.operating_margin,
                    net_margin=p.net_margin,
                    debt_to_equity=p.debt_to_equity,
                    debt_to_assets=debt_to_assets,
                    current_ratio=p.current_ratio,
                    cash_ratio=cash_ratio,
                    operating_cash_flow_ratio=ocf_ratio,
                    interest_coverage=icov,
                    days_sales_outstanding=dso,
                    revenue_growth=p.revenue_growth,
                    earnings_per_share=round(p.net_income / shares, 4),
                    book_value_per_share=round(p.shareholders_equity / shares, 4),
                    free_cash_flow_per_share=round(p.free_cash_flow / shares, 4),
                )
            )
        return rows

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        case = self._cases.get(ticker)
        if not case:
            return None
        return CompanyFacts(
            ticker=ticker,
            name=f"Public Issuer {case.get('case_id', ticker)}",
            sector=case.get("sector"),
            industry=case.get("industry"),
        )

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs) -> list[Price]:
        return []

    def get_snapshot(self, ticker: str) -> PriceSnapshot | None:
        return None

    def get_news(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[CompanyNews]:
        return []

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[InsiderTrade]:
        return []

    def get_earnings(self, ticker: str) -> Earnings | None:
        return None

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        return []

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        rows = self.get_financial_metrics(ticker, end_date, limit=1)
        return rows[0].market_cap if rows else None

    def screen(self, as_of: str, filters: list, **kwargs) -> list:
        return []

    def screener_fields(self):
        return None
