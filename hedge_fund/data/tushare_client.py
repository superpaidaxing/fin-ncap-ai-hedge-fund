"""Tushare Pro A-share DataClient (zero extra dependencies, pure HTTP).

Implements `hedge_fund.data.protocol.DataClient` plus `get_statement_dossier()`
so the entire v2 engine (5 LLM investor personas, ForensicAccountingGate, and
backtesting) works natively on China A-share public filings.

Design guarantees:
- Strict Point-in-Time: financial statements are filtered by statutory
  announcement date (`ann_date <= end_date`), never future filings.
- Consolidated statements only (`report_type="1"`), deduplicated by latest
  `update_flag` per fiscal period.
- Unit normalization: Tushare statements (CNY yuan) -> CNY millions for
  forensic dossier; `daily_basic.total_mv` (10k CNY) -> CNY yuan / millions;
  `fina_indicator` percentages -> decimal ratios.
- Disk cache (`~/.cache/aihf_tushare`) avoids redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any

import requests

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

TUSHARE_URL = "https://api.tushare.pro"


def normalize_ashare_code(code: str) -> str:
    """Normalize A-share ticker like '600519' or 'sz002450' -> '600519.SH' / '002450.SZ'."""
    raw = code.strip().upper()
    if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", raw):
        return raw
    m = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", raw)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    if re.fullmatch(r"\d{6}", raw):
        if raw.startswith(("6", "9")):
            return f"{raw}.SH"
        if raw.startswith(("0", "2", "3")):
            return f"{raw}.SZ"
        if raw.startswith(("4", "8")):
            return f"{raw}.BJ"
    return raw


def _f(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _f_opt(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_iso_date(yyyymmdd: str | None) -> str | None:
    if not yyyymmdd or len(str(yyyymmdd)) < 8:
        return None
    s = str(yyyymmdd)[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _to_yyyymmdd(iso_or_compact: str) -> str:
    return iso_or_compact[:10].replace("-", "")


class TushareError(RuntimeError):
    """Raised when Tushare Pro API returns an error or token is missing."""


class TushareDataClient:
    """Point-in-time China A-share DataClient powered by Tushare Pro HTTP API."""

    def __init__(
        self,
        token: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        enable_forensic: bool = True,
        timeout: float = 20.0,
    ) -> None:
        self._token = (token or os.environ.get("TUSHARE_TOKEN") or "").strip()
        if not self._token:
            raise TushareError("Missing TUSHARE_TOKEN environment variable for A-share data.")
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "aihf_tushare"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._enable_forensic = enable_forensic
        self._timeout = timeout
        self._session = requests.Session()

    def _call(self, api_name: str, params: dict[str, Any], fields: str = "") -> list[dict[str, Any]]:
        key_payload = json.dumps({"api": api_name, "params": params, "fields": fields}, sort_keys=True)
        cache_key = hashlib.sha256(key_payload.encode()).hexdigest()[:24]
        cache_file = self._cache_dir / f"{api_name}_{cache_key}.json"
        if cache_file.is_file():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except OSError:
                pass

        resp = self._session.post(
            TUSHARE_URL,
            json={"api_name": api_name, "token": self._token, "params": params, "fields": fields},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise TushareError(f"Tushare {api_name} error ({body.get('code')}): {body.get('msg')}")
        data = body.get("data") or {}
        cols = data.get("fields") or []
        items = data.get("items") or []
        rows = [dict(zip(cols, r)) for r in items]
        try:
            cache_file.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return rows

    @staticmethod
    def _dedup_annual(rows: list[dict[str, Any]], as_of_compact: str) -> dict[str, dict[str, Any]]:
        """Keep point-in-time annual (1231) rows announced on or before as_of, latest update_flag wins."""
        by_end: dict[str, dict[str, Any]] = {}
        for r in rows:
            end_d = str(r.get("end_date") or "")
            ann_d = str(r.get("ann_date") or r.get("f_ann_date") or "")
            if not end_d.endswith("1231"):
                continue
            if ann_d and ann_d > as_of_compact:
                continue
            prev = by_end.get(end_d)
            if prev is None or str(r.get("update_flag") or "0") >= str(prev.get("update_flag") or "0"):
                by_end[end_d] = r
        return by_end

    def _latest_daily_basic(self, ts_code: str, as_of_compact: str) -> dict[str, Any] | None:
        end_dt = _date.fromisoformat(_fmt_iso_date(as_of_compact) or _date.today().isoformat())
        start_dt = end_dt - timedelta(days=30)
        rows = self._call(
            "daily_basic",
            {"ts_code": ts_code, "start_date": start_dt.strftime("%Y%m%d"), "end_date": as_of_compact},
            "ts_code,trade_date,close,pe_ttm,pb,total_mv,total_share",
        )
        if not rows:
            return None
        rows.sort(key=lambda x: str(x.get("trade_date") or ""), reverse=True)
        return rows[0]

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        ts_code = normalize_ashare_code(ticker)
        rows = self._call(
            "stock_basic",
            {"ts_code": ts_code},
            "ts_code,name,industry,market,area,list_date",
        )
        if not rows:
            return None
        r = rows[0]
        return CompanyFacts(
            ticker=ts_code,
            name=r.get("name"),
            sector=r.get("industry"),
            industry=r.get("industry"),
            exchange=r.get("market"),
            location=r.get("area"),
        )

    def get_market_snapshot_info(self, ticker: str, as_of: str) -> dict[str, Any]:
        """Helper for rich CLI header: name, industry, latest close, PE_TTM, PB, total_mv (CNY bn)."""
        ts_code = normalize_ashare_code(ticker)
        facts = self.get_company_facts(ts_code)
        db = self._latest_daily_basic(ts_code, _to_yyyymmdd(as_of))
        return {
            "ts_code": ts_code,
            "name": facts.name if facts else ts_code,
            "industry": facts.industry if facts else "未知行业",
            "trade_date": _fmt_iso_date(db.get("trade_date")) if db else as_of[:10],
            "close": _f_opt(db.get("close")) if db else None,
            "pe_ttm": _f_opt(db.get("pe_ttm")) if db else None,
            "pb": _f_opt(db.get("pb")) if db else None,
            "total_mv_billion": round(_f(db.get("total_mv")) / 1e4, 2) if db and db.get("total_mv") else None,  # 万元 -> 亿元
        }

    def _build_periods(self, ts_code: str, as_of_compact: str, limit: int = 5) -> tuple[list[FinancialStatementPeriod], AuditAndGovernanceNotes]:
        incomes = self._dedup_annual(
            self._call(
                "income",
                {"ts_code": ts_code, "report_type": "1"},
                "ts_code,ann_date,f_ann_date,end_date,update_flag,total_revenue,revenue,n_income,n_income_attr_p,minority_gain,int_exp,fin_exp,fin_exp_int_inc,fin_exp_int_exp,oper_cost,operate_profit",
            ),
            as_of_compact,
        )
        balances = self._dedup_annual(
            self._call(
                "balancesheet",
                {"ts_code": ts_code, "report_type": "1"},
                "ts_code,ann_date,f_ann_date,end_date,update_flag,money_cap,st_borr,lt_borr,bond_payable,non_cur_liab_due_1y,accounts_receiv,inventories,goodwill,intan_assets,r_and_d,total_assets,total_liab,total_hldr_eqy_exc_min_int,total_hldr_eqy_inc_min_int,total_cur_assets,total_cur_liab",
            ),
            as_of_compact,
        )
        cashflows = self._dedup_annual(
            self._call(
                "cashflow",
                {"ts_code": ts_code, "report_type": "1"},
                "ts_code,ann_date,f_ann_date,end_date,update_flag,n_cashflow_act,c_pay_acq_const_fiolta,free_cashflow",
            ),
            as_of_compact,
        )
        indicators = self._dedup_annual(
            self._call(
                "fina_indicator",
                {"ts_code": ts_code},
                "ts_code,ann_date,end_date,update_flag,roe,grossprofit_margin,netprofit_margin,current_ratio,debt_to_assets,eps,bps,tr_yoy,or_yoy",
            ),
            as_of_compact,
        )

        common_ends = sorted(set(incomes) & set(balances) & set(cashflows), reverse=True)[:limit]
        periods: list[FinancialStatementPeriod] = []

        for end_d in common_ends:
            inc = incomes[end_d]
            bs = balances[end_d]
            cf = cashflows[end_d]
            ind = indicators.get(end_d, {})

            ann_d = str(bs.get("ann_date") or inc.get("ann_date") or end_d)
            filing_iso = _fmt_iso_date(ann_d) or _fmt_iso_date(end_d) or end_d
            period_iso = _fmt_iso_date(end_d) or end_d

            # Fetch market cap / PE around filing_date (strictly <= as_of)
            val_anchor = min(ann_d, as_of_compact)
            db = self._latest_daily_basic(ts_code, val_anchor)

            # Convert CNY yuan -> CNY millions for clean readable statements
            M = 1e6
            revenue = (_f(inc.get("revenue")) or _f(inc.get("total_revenue"))) / M
            oper_cost = _f(inc.get("oper_cost")) / M
            gross_profit = revenue - oper_cost if revenue > 0 else None
            operating_income = _f(inc.get("operate_profit")) / M
            net_income = (_f(inc.get("n_income_attr_p")) or _f(inc.get("n_income"))) / M
            minority_gain = _f(inc.get("minority_gain")) / M

            int_inc = _f(inc.get("fin_exp_int_inc")) / M
            int_exp = (_f(inc.get("fin_exp_int_exp")) or _f(inc.get("int_exp"))) / M
            fin_exp = _f(inc.get("fin_exp")) / M
            if int_inc == 0.0 and fin_exp < 0:
                int_inc = abs(fin_exp)
            if int_exp == 0.0 and fin_exp > 0:
                int_exp = fin_exp

            ocf = _f(cf.get("n_cashflow_act")) / M
            capex = _f(cf.get("c_pay_acq_const_fiolta")) / M
            fcf_raw = _f_opt(cf.get("free_cashflow"))
            fcf = (fcf_raw / M) if fcf_raw is not None else (ocf - capex)

            cash = _f(bs.get("money_cap")) / M
            ib_debt = (
                _f(bs.get("st_borr"))
                + _f(bs.get("lt_borr"))
                + _f(bs.get("bond_payable"))
                + _f(bs.get("non_cur_liab_due_1y"))
            ) / M
            ar = _f(bs.get("accounts_receiv")) / M
            inv = _f(bs.get("inventories")) / M
            gw = _f(bs.get("goodwill")) / M
            intang = (_f(bs.get("intan_assets")) + _f(bs.get("r_and_d"))) / M
            total_assets = _f(bs.get("total_assets")) / M
            total_liab = _f(bs.get("total_liab")) / M
            # Use total equity INCLUDING minority interest so Assets == Liab + Equity reconciles cleanly
            equity_inc = _f(bs.get("total_hldr_eqy_inc_min_int")) / M
            equity_exc = _f(bs.get("total_hldr_eqy_exc_min_int")) / M
            equity = equity_inc if equity_inc > 0 else equity_exc
            cur_assets = _f(bs.get("total_cur_assets")) / M
            cur_liab = _f(bs.get("total_cur_liab")) / M

            # Valuation & per-share
            total_mv_m = (_f(db.get("total_mv")) / 100.0) if (db and db.get("total_mv")) else None  # 万元 -> 百万元
            pe_ttm = _f_opt(db.get("pe_ttm")) if db else None
            if pe_ttm is None and total_mv_m and net_income > 0:
                pe_ttm = round(total_mv_m / net_income, 2)
            shares_m = (_f(db.get("total_share")) / 100.0) if (db and db.get("total_share")) else None  # 万股 -> 百万股

            roe_pct = _f_opt(ind.get("roe"))
            gm_pct = _f_opt(ind.get("grossprofit_margin"))
            nm_pct = _f_opt(ind.get("netprofit_margin"))
            rev_yoy_pct = _f_opt(ind.get("or_yoy")) if ind.get("or_yoy") is not None else _f_opt(ind.get("tr_yoy"))
            curr_ratio = _f_opt(ind.get("current_ratio"))
            if curr_ratio is None and cur_liab > 0:
                curr_ratio = round(cur_assets / cur_liab, 2)

            periods.append(
                FinancialStatementPeriod(
                    report_period=period_iso,
                    filing_date=filing_iso,
                    market_cap=round(total_mv_m, 2) if total_mv_m is not None else None,
                    revenue=round(revenue, 2),
                    gross_profit=round(gross_profit, 2) if gross_profit is not None else None,
                    operating_income=round(operating_income, 2),
                    net_income=round(net_income, 2),
                    minority_interest_income=round(minority_gain, 2),
                    interest_income=round(int_inc, 2),
                    interest_expense=round(int_exp, 2),
                    operating_cash_flow=round(ocf, 2),
                    capex=round(capex, 2),
                    free_cash_flow=round(fcf, 2),
                    cash_and_equivalents=round(cash, 2),
                    interest_bearing_debt=round(ib_debt, 2),
                    accounts_receivable=round(ar, 2),
                    inventory=round(inv, 2),
                    goodwill=round(gw, 2),
                    intangible_and_dev_assets=round(intang, 2),
                    total_assets=round(total_assets, 2),
                    total_liabilities=round(total_liab, 2),
                    shareholders_equity=round(equity, 2),
                    current_assets=round(cur_assets, 2),
                    current_liabilities=round(cur_liab, 2),
                    shares_diluted=round(shares_m, 4) if shares_m else None,
                    revenue_growth=round(rev_yoy_pct / 100.0, 4) if rev_yoy_pct is not None else None,
                    price_to_earnings_ratio=round(pe_ttm, 2) if pe_ttm is not None else None,
                    return_on_equity=round(roe_pct / 100.0, 4) if roe_pct is not None else (round(net_income / equity, 4) if equity > 0 else None),
                    gross_margin=round(gm_pct / 100.0, 4) if gm_pct is not None else (round(gross_profit / revenue, 4) if (gross_profit is not None and revenue > 0) else None),
                    operating_margin=round(operating_income / revenue, 4) if revenue > 0 else None,
                    net_margin=round(nm_pct / 100.0, 4) if nm_pct is not None else (round(net_income / revenue, 4) if revenue > 0 else None),
                    debt_to_equity=round(ib_debt / equity, 3) if equity > 0 else None,
                    current_ratio=round(curr_ratio, 2) if curr_ratio is not None else None,
                )
            )

        # Fetch statutory audit opinion & pledge ratio strictly <= as_of
        audit_opinion = "Standard Unqualified"
        audit_agency = ""
        try:
            audits = self._call("fina_audit", {"ts_code": ts_code}, "ts_code,ann_date,end_date,audit_result,audit_agency")
            valid_audits = [a for a in audits if str(a.get("ann_date") or "") <= as_of_compact and str(a.get("end_date") or "").endswith("1231")]
            valid_audits.sort(key=lambda x: str(x.get("end_date") or ""), reverse=True)
            if valid_audits:
                audit_opinion = str(valid_audits[0].get("audit_result") or "Standard Unqualified")
                audit_agency = str(valid_audits[0].get("audit_agency") or "")
        except Exception:
            pass

        pledge_ratio = 0.0
        try:
            pledges = self._call("pledge_stat", {"ts_code": ts_code}, "ts_code,end_date,pledge_ratio")
            valid_pledges = [p for p in pledges if str(p.get("end_date") or "") <= as_of_compact and p.get("pledge_ratio") is not None]
            valid_pledges.sort(key=lambda x: str(x.get("end_date") or ""), reverse=True)
            if valid_pledges:
                # Tushare pledge_ratio is total-share pledge % (e.g. 45.2 means 45.2% of total shares)
                pledge_ratio = round(float(valid_pledges[0]["pledge_ratio"]) / 100.0, 4)
        except Exception:
            pass

        notes = AuditAndGovernanceNotes(
            audit_opinion=f"{audit_opinion} ({audit_agency})" if audit_agency else audit_opinion,
            restricted_cash_ratio=0.0,  # Tushare standard table does not expose footnote restricted-cash ratio
            major_shareholder_pledge_ratio=pledge_ratio,
            external_guarantee_to_equity=0.0,
            related_party_notes=f"Tushare Pro PIT filing <= {as_of_compact}; audit={audit_opinion}; total_share_pledge={pledge_ratio*100:.2f}%",
        )
        return periods, notes

    def get_statement_dossier(self, ticker: str, end_date: str, limit: int = 5) -> FinancialStatementDossier | None:
        if not self._enable_forensic:
            return None
        ts_code = normalize_ashare_code(ticker)
        facts = self.get_company_facts(ts_code)
        periods, notes = self._build_periods(ts_code, _to_yyyymmdd(end_date), limit=limit)
        if not periods:
            return None
        return FinancialStatementDossier(
            ticker=ts_code,
            sector=facts.sector if facts else None,
            industry=facts.industry if facts else None,
            currency="CNY",
            periods=periods,
            notes_and_audit=notes,
        )

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        ts_code = normalize_ashare_code(ticker)
        periods, _ = self._build_periods(ts_code, _to_yyyymmdd(end_date), limit=min(limit, 8))
        rows: list[FinancialMetrics] = []
        for p in periods:
            shares = p.shares_diluted or 1.0
            dso = round((p.accounts_receivable / p.revenue) * 365.0, 2) if p.revenue > 0 else None
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
            rows.append(
                FinancialMetrics(
                    ticker=ts_code,
                    report_period=p.report_period,
                    period="annual",
                    currency="CNY",
                    filing_date=p.filing_date,
                    market_cap=(p.market_cap * 1e6) if p.market_cap is not None else None,
                    price_to_earnings_ratio=p.price_to_earnings_ratio,
                    return_on_equity=p.return_on_equity,
                    gross_margin=p.gross_margin,
                    operating_margin=p.operating_margin,
                    net_margin=p.net_margin,
                    debt_to_equity=p.debt_to_equity,
                    debt_to_assets=round(p.total_liabilities / p.total_assets, 3) if p.total_assets > 0 else None,
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

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs) -> list[Price]:
        ts_code = normalize_ashare_code(ticker)
        rows = self._call(
            "daily",
            {"ts_code": ts_code, "start_date": _to_yyyymmdd(start_date), "end_date": _to_yyyymmdd(end_date)},
            "trade_date,open,high,low,close,vol",
        )
        rows.sort(key=lambda x: str(x.get("trade_date") or ""))
        return [
            Price(
                open=_f(r["open"]),
                high=_f(r["high"]),
                low=_f(r["low"]),
                close=_f(r["close"]),
                volume=int(_f(r["vol"]) * 100),  # 手 -> 股
                time=_fmt_iso_date(str(r["trade_date"])) or str(r["trade_date"]),
            )
            for r in rows
        ]

    def get_snapshot(self, ticker: str) -> PriceSnapshot | None:
        ts_code = normalize_ashare_code(ticker)
        db = self._latest_daily_basic(ts_code, _date.today().strftime("%Y%m%d"))
        if not db:
            return None
        return PriceSnapshot(
            ticker=ts_code,
            price=_f(db["close"]),
            time=_fmt_iso_date(str(db["trade_date"])) or str(db["trade_date"]),
        )

    def get_news(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[CompanyNews]:
        return []

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[InsiderTrade]:
        return []

    def get_earnings(self, ticker: str) -> Earnings | None:
        return None

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        return []

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        db = self._latest_daily_basic(normalize_ashare_code(ticker), _to_yyyymmdd(end_date))
        return (_f(db["total_mv"]) * 1e4) if (db and db.get("total_mv")) else None

    def screen(self, as_of: str, filters: list, **kwargs) -> list:
        return []

    def screener_fields(self):
        return None
