"""Deterministic forensic accounting gate (zero-hallucination rule engine).

All calculations are pure Python over point-in-time publicly filed numbers.
No LLM is involved in computing accounting identities, ratios, or fail-closed gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from hedge_fund.forensic.models import (
    FinancialStatementDossier,
    ForensicAuditResult,
    ForensicRedFlag,
)


@dataclass(frozen=True)
class ForensicThresholds:
    """Explicit, auditable financial risk thresholds."""

    bs_identity_tolerance: float = 0.001  # 0.1% of total assets
    cash_high_ratio: float = 0.25
    debt_high_ratio: float = 0.20
    low_cash_yield: float = 0.012  # 1.2% implied annual deposit yield
    interest_spread_multiplier: float = 2.0  # interest expense >= 2x interest income
    restricted_cash_critical: float = 0.30
    ocf_to_ni_low: float = 0.70
    dso_high_days: float = 100.0
    dso_jump_days: float = 25.0
    inventory_to_assets_high: float = 0.20
    inventory_days_high: float = 250.0
    goodwill_to_equity_high: float = 0.30
    intangible_to_assets_high: float = 0.25
    minority_loss_masking_ratio: float = 0.30
    pledge_ratio_critical: float = 0.70
    external_guarantee_fail_closed: float = 0.25
    # Risk cap trigger: >= 2 critical flags OR severity score >= 4 forces BEARISH_CAP
    bearish_cap_score: int = 4


class ForensicAccountingGate:
    """Evaluates a full FinancialStatementDossier deterministically."""

    def __init__(self, thresholds: ForensicThresholds | None = None) -> None:
        self.t = thresholds or ForensicThresholds()

    def evaluate(self, dossier: FinancialStatementDossier) -> ForensicAuditResult:
        if not dossier.periods:
            return ForensicAuditResult(
                ticker=dossier.ticker,
                latest_period="N/A",
                fail_closed=True,
                fail_closed_reasons=["No publicly filed financial statement periods provided."],
                risk_cap="FAIL_CLOSED",
                severity_score=10,
            )

        p0 = dossier.periods[0]
        notes = dossier.notes_and_audit
        flags: list[ForensicRedFlag] = []
        fail_closed_reasons: list[str] = []

        # ------------------------------------------------------------------
        # 1. Accounting identity check (Assets == Liabilities + Equity)
        # ------------------------------------------------------------------
        bs_gap = abs(p0.total_assets - (p0.total_liabilities + p0.shareholders_equity))
        bs_gap_ratio = bs_gap / p0.total_assets if p0.total_assets > 0 else 1.0
        if bs_gap_ratio > self.t.bs_identity_tolerance:
            msg = (
                f"Balance sheet does not balance: Assets ({p0.total_assets:.1f}) != "
                f"Liabilities ({p0.total_liabilities:.1f}) + Equity ({p0.shareholders_equity:.1f}), "
                f"gap={bs_gap:.1f} ({bs_gap_ratio*100:.2f}%)"
            )
            fail_closed_reasons.append(msg)
            flags.append(
                ForensicRedFlag(
                    code="ACCOUNTING_IDENTITY_BROKEN",
                    severity="FAIL_CLOSED",
                    title="资产负债表勾稽不平",
                    metric_value=f"gap={bs_gap:.1f} ({bs_gap_ratio*100:.2f}%)",
                    threshold=f"<= {self.t.bs_identity_tolerance*100:.2f}%",
                    evidence=msg,
                )
            )

        # ------------------------------------------------------------------
        # 2. Non-standard audit opinion (Fail-Closed hard gate)
        # ------------------------------------------------------------------
        opinion_lower = notes.audit_opinion.lower()
        is_clean_opinion = ("unqualified" in opinion_lower or "无保留" in notes.audit_opinion) and (
            "qualified" not in opinion_lower.replace("unqualified", "")
            and "保留" not in notes.audit_opinion.replace("无保留", "")
            and "adverse" not in opinion_lower
            and "disclaimer" not in opinion_lower
            and "否定" not in notes.audit_opinion
            and "无法表示" not in notes.audit_opinion
        )
        if not is_clean_opinion:
            msg = f"Non-standard audit opinion issued: {notes.audit_opinion}"
            fail_closed_reasons.append(msg)
            flags.append(
                ForensicRedFlag(
                    code="NON_STANDARD_AUDIT_OPINION",
                    severity="FAIL_CLOSED",
                    title="非标准审计意见",
                    metric_value=notes.audit_opinion[:60],
                    threshold="Standard Unqualified",
                    evidence=msg,
                )
            )

        # ------------------------------------------------------------------
        # 3. Excessive external guarantees (Fail-Closed hard gate)
        # ------------------------------------------------------------------
        if notes.external_guarantee_to_equity >= self.t.external_guarantee_fail_closed:
            msg = (
                f"External guarantee balance is {notes.external_guarantee_to_equity*100:.1f}% of equity "
                f"(exceeds {self.t.external_guarantee_fail_closed*100:.0f}% hard limit)"
            )
            fail_closed_reasons.append(msg)
            flags.append(
                ForensicRedFlag(
                    code="EXCESSIVE_EXTERNAL_GUARANTEE",
                    severity="FAIL_CLOSED",
                    title="超限对外担保风险",
                    metric_value=f"{notes.external_guarantee_to_equity*100:.1f}%",
                    threshold=f"< {self.t.external_guarantee_fail_closed*100:.0f}%",
                    evidence=msg,
                )
            )

        # ------------------------------------------------------------------
        # 4. Cash-Debt Double High & Implied Cash Yield Anomaly
        # ------------------------------------------------------------------
        cash_to_assets = p0.cash_and_equivalents / p0.total_assets if p0.total_assets > 0 else 0.0
        debt_to_assets = p0.interest_bearing_debt / p0.total_assets if p0.total_assets > 0 else 0.0
        avg_cash = (
            (p0.cash_and_equivalents + dossier.periods[1].cash_and_equivalents) / 2.0
            if len(dossier.periods) >= 2
            else p0.cash_and_equivalents
        )
        # Use latest cash for conservative implied yield (and also cross-check avg)
        implied_cash_yield = (
            p0.interest_income / p0.cash_and_equivalents if p0.cash_and_equivalents > 0 else None
        )

        if cash_to_assets >= self.t.cash_high_ratio and debt_to_assets >= self.t.debt_high_ratio:
            flags.append(
                ForensicRedFlag(
                    code="CASH_DEBT_DOUBLE_HIGH",
                    severity="CRITICAL",
                    title="存贷双高（巨额货币资金与高有息负债并存）",
                    metric_value=f"Cash/Assets={cash_to_assets*100:.1f}%, Debt/Assets={debt_to_assets*100:.1f}%",
                    threshold=f"Cash>={self.t.cash_high_ratio*100:.0f}% & Debt>={self.t.debt_high_ratio*100:.0f}%",
                    evidence=(
                        f"Cash {p0.cash_and_equivalents:.1f} ({cash_to_assets*100:.1f}% of assets) coexists with "
                        f"interest-bearing debt {p0.interest_bearing_debt:.1f} ({debt_to_assets*100:.1f}% of assets)"
                    ),
                )
            )

        if (
            implied_cash_yield is not None
            and cash_to_assets >= 0.20
            and implied_cash_yield < self.t.low_cash_yield
            and p0.interest_expense >= p0.interest_income * self.t.interest_spread_multiplier
        ):
            flags.append(
                ForensicRedFlag(
                    code="ABNORMAL_LOW_INTEREST_YIELD",
                    severity="CRITICAL",
                    title="货币资金隐含收益率异常偏低且息费倒挂",
                    metric_value=f"yield={implied_cash_yield*100:.2f}%, int_exp/int_inc={p0.interest_expense/max(p0.interest_income,1e-6):.1f}x",
                    threshold=f"yield >= {self.t.low_cash_yield*100:.2f}%",
                    evidence=(
                        f"Implied interest yield on cash is only {implied_cash_yield*100:.2f}% "
                        f"(interest income {p0.interest_income:.1f} vs cash {p0.cash_and_equivalents:.1f}, "
                        f"avg_cash={avg_cash:.1f}) while paying {p0.interest_expense:.1f} in interest expense"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # 5. Restricted Cash / Group Cash Pooling
        # ------------------------------------------------------------------
        if notes.restricted_cash_ratio >= self.t.restricted_cash_critical:
            flags.append(
                ForensicRedFlag(
                    code="RESTRICTED_CASH_OR_POOLING",
                    severity="CRITICAL",
                    title="巨额货币资金受限或被集团归集",
                    metric_value=f"{notes.restricted_cash_ratio*100:.1f}%",
                    threshold=f"< {self.t.restricted_cash_critical*100:.0f}%",
                    evidence=(
                        f"Restricted or pooled cash accounts for {notes.restricted_cash_ratio*100:.1f}% "
                        f"of reported cash balance"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # 6. Cash Flow Quality: OCF vs Net Income & Chronic Negative FCF
        # ------------------------------------------------------------------
        ocf_to_ni_latest = p0.operating_cash_flow / p0.net_income if p0.net_income > 0 else None
        cum_ocf = sum(p.operating_cash_flow for p in dossier.periods)
        cum_ni = sum(p.net_income for p in dossier.periods)
        ocf_to_ni_cum = cum_ocf / cum_ni if cum_ni > 0 else None

        if (
            ocf_to_ni_latest is not None
            and ocf_to_ni_cum is not None
            and ocf_to_ni_latest < self.t.ocf_to_ni_low
            and ocf_to_ni_cum < self.t.ocf_to_ni_low
        ):
            flags.append(
                ForensicRedFlag(
                    code="OCF_NET_INCOME_DIVERGENCE",
                    severity="CRITICAL",
                    title="经营性现金流长期显著低于净利润（净现比失真）",
                    metric_value=f"latest={ocf_to_ni_latest:.2f}, multi-yr={ocf_to_ni_cum:.2f}",
                    threshold=f">= {self.t.ocf_to_ni_low:.2f}",
                    evidence=(
                        f"Operating cash flow converts only {ocf_to_ni_latest*100:.1f}% of net income "
                        f"(multi-year cumulative conversion {ocf_to_ni_cum*100:.1f}%)"
                    ),
                )
            )

        neg_fcf_count = sum(1 for p in dossier.periods if p.free_cash_flow < 0)
        if neg_fcf_count >= max(3, len(dossier.periods) - 1):
            flags.append(
                ForensicRedFlag(
                    code="FCF_CHRONIC_NEGATIVE",
                    severity="CRITICAL",
                    title="自由现金流连续多年深度为负（持续失血）",
                    metric_value=f"{neg_fcf_count}/{len(dossier.periods)} periods negative (latest={p0.free_cash_flow:.1f})",
                    threshold="Positive FCF in majority of periods",
                    evidence=(
                        f"Free cash flow was negative in {neg_fcf_count} of {len(dossier.periods)} filed periods "
                        f"(latest FCF={p0.free_cash_flow:.1f})"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # 7. Receivables DSO Surge & Inventory Bloating
        # ------------------------------------------------------------------
        dso_series = [
            round((p.accounts_receivable / p.revenue) * 365.0, 1) if p.revenue > 0 else 0.0
            for p in dossier.periods
        ]
        dso_latest = dso_series[0]
        dso_oldest = dso_series[-1]
        if dso_latest >= self.t.dso_high_days or (dso_latest >= 90.0 and (dso_latest - dso_oldest) >= self.t.dso_jump_days):
            flags.append(
                ForensicRedFlag(
                    code="RECEIVABLES_DSO_SURGE",
                    severity="WARNING" if dso_latest < 130.0 else "CRITICAL",
                    title="应收账款周转天数（DSO）偏高或持续拉长",
                    metric_value=f"DSO={dso_latest:.1f} days (oldest={dso_oldest:.1f})",
                    threshold=f"< {self.t.dso_high_days:.0f} days",
                    evidence=(
                        f"Accounts receivable ({p0.accounts_receivable:.1f}) implies DSO of {dso_latest:.1f} days "
                        f"vs {dso_oldest:.1f} days in earliest period"
                    ),
                )
            )

        inv_to_assets = p0.inventory / p0.total_assets if p0.total_assets > 0 else 0.0
        cogs = (p0.revenue - p0.gross_profit) if p0.gross_profit is not None else (p0.revenue * 0.7)
        inv_days = round((p0.inventory / cogs) * 365.0, 1) if cogs > 0 else 0.0
        # Industry-aware guard: high gross margin (>=75%, e.g. aged spirits) with minimal
        # interest-bearing debt (<=5% of assets) and strong multi-year cash conversion (>=0.90)
        # naturally carries long inventory aging cycles without solvency stress.
        is_aged_inventory_profile = (
            (p0.gross_margin or 0.0) >= 0.75
            and debt_to_assets <= 0.05
            and (ocf_to_ni_cum or 0.0) >= 0.90
        )
        if inv_to_assets >= self.t.inventory_to_assets_high and inv_days >= self.t.inventory_days_high and not is_aged_inventory_profile:
            flags.append(
                ForensicRedFlag(
                    code="INVENTORY_BLOATING",
                    severity="CRITICAL",
                    title="存货高企且周转异常缓慢（潜在虚增/减值雷）",
                    metric_value=f"Inv/Assets={inv_to_assets*100:.1f}%, DIO={inv_days:.0f} days",
                    threshold=f"Inv/Assets < {self.t.inventory_to_assets_high*100:.0f}% or DIO < {self.t.inventory_days_high:.0f}d",
                    evidence=(
                        f"Inventory reached {p0.inventory:.1f} ({inv_to_assets*100:.1f}% of assets, "
                        f"DIO={inv_days:.0f} days) alongside weak cash conversion"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # 8. Goodwill Overhang & Tangible Book Value Erosion
        # ------------------------------------------------------------------
        goodwill_to_equity = p0.goodwill / p0.shareholders_equity if p0.shareholders_equity > 0 else 0.0
        tangible_equity = p0.shareholders_equity - p0.goodwill - p0.intangible_and_dev_assets
        tangible_equity_ratio = tangible_equity / p0.shareholders_equity if p0.shareholders_equity > 0 else 0.0
        if goodwill_to_equity >= self.t.goodwill_to_equity_high:
            flags.append(
                ForensicRedFlag(
                    code="GOODWILL_OVERHANG",
                    severity="CRITICAL",
                    title="巨额并购商誉悬顶（有形净资产被大幅掏空）",
                    metric_value=f"Goodwill/Equity={goodwill_to_equity*100:.1f}%, Tangible/Book={tangible_equity_ratio*100:.1f}%",
                    threshold=f"Goodwill/Equity < {self.t.goodwill_to_equity_high*100:.0f}%",
                    evidence=(
                        f"Goodwill ({p0.goodwill:.1f}) equals {goodwill_to_equity*100:.1f}% of book equity "
                        f"({p0.shareholders_equity:.1f}); tangible book equity is only {tangible_equity:.1f} "
                        f"({tangible_equity_ratio*100:.1f}% of reported book value)"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # 9. Aggressive Intangible & R&D Capitalization
        # ------------------------------------------------------------------
        intangible_to_assets = p0.intangible_and_dev_assets / p0.total_assets if p0.total_assets > 0 else 0.0
        if intangible_to_assets >= self.t.intangible_to_assets_high:
            flags.append(
                ForensicRedFlag(
                    code="INTANGIBLE_CAPITALIZATION_HIGH",
                    severity="CRITICAL",
                    title="无形资产与开发支出占比过高（激进资本化美化利润）",
                    metric_value=f"{intangible_to_assets*100:.1f}% of total assets",
                    threshold=f"< {self.t.intangible_to_assets_high*100:.0f}%",
                    evidence=(
                        f"Intangible and capitalized dev assets ({p0.intangible_and_dev_assets:.1f}) "
                        f"represent {intangible_to_assets*100:.1f}% of total assets"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # 10. Minority Interest Loss Masking & Major Shareholder Pledge
        # ------------------------------------------------------------------
        minority_to_ni = p0.minority_interest_income / p0.net_income if p0.net_income > 0 else 0.0
        if p0.net_income > 0 and p0.minority_interest_income < 0 and abs(minority_to_ni) >= self.t.minority_loss_masking_ratio:
            flags.append(
                ForensicRedFlag(
                    code="MINORITY_LOSS_MASKING",
                    severity="CRITICAL",
                    title="少数股东损益巨亏掩饰归母净利润",
                    metric_value=f"Minority={p0.minority_interest_income:.1f} vs Parent NI={p0.net_income:.1f}",
                    threshold="Minority loss < 30% of parent NI",
                    evidence=(
                        f"Parent net income is positive ({p0.net_income:.1f}) while minority interest "
                        f"absorbed a loss of {p0.minority_interest_income:.1f} ({minority_to_ni*100:.1f}% of parent NI)"
                    ),
                )
            )

        if notes.major_shareholder_pledge_ratio >= self.t.pledge_ratio_critical:
            flags.append(
                ForensicRedFlag(
                    code="PLEDGE_RATIO_CRITICAL",
                    severity="CRITICAL",
                    title="控股股东股权质押比例触及平仓高危区",
                    metric_value=f"{notes.major_shareholder_pledge_ratio*100:.1f}%",
                    threshold=f"< {self.t.pledge_ratio_critical*100:.0f}%",
                    evidence=f"Major shareholder pledged {notes.major_shareholder_pledge_ratio*100:.1f}% of holdings",
                )
            )

        # ------------------------------------------------------------------
        # Aggregate severity & risk cap decision
        # ------------------------------------------------------------------
        weight = {"FAIL_CLOSED": 5, "CRITICAL": 2, "WARNING": 1}
        severity_score = sum(weight[f.severity] for f in flags)
        fail_closed = len(fail_closed_reasons) > 0
        if fail_closed:
            risk_cap = "FAIL_CLOSED"
        elif severity_score >= self.t.bearish_cap_score:
            risk_cap = "BEARISH_CAP"
        else:
            risk_cap = "NONE"

        return ForensicAuditResult(
            ticker=dossier.ticker,
            latest_period=p0.report_period,
            fail_closed=fail_closed,
            fail_closed_reasons=fail_closed_reasons,
            risk_cap=risk_cap,
            severity_score=severity_score,
            red_flags=flags,
            key_metrics={
                "bs_identity_gap_ratio": f"{bs_gap_ratio*100:.2f}%",
                "cash_to_assets": f"{cash_to_assets*100:.1f}%",
                "debt_to_assets": f"{debt_to_assets*100:.1f}%",
                "implied_cash_yield": f"{implied_cash_yield*100:.2f}%" if implied_cash_yield is not None else None,
                "restricted_cash_ratio": f"{notes.restricted_cash_ratio*100:.1f}%",
                "ocf_to_ni_latest": round(ocf_to_ni_latest, 2) if ocf_to_ni_latest is not None else None,
                "ocf_to_ni_cumulative": round(ocf_to_ni_cum, 2) if ocf_to_ni_cum is not None else None,
                "dso_latest": dso_latest,
                "inventory_to_assets": f"{inv_to_assets*100:.1f}%",
                "goodwill_to_equity": f"{goodwill_to_equity*100:.1f}%",
                "tangible_equity_ratio": f"{tangible_equity_ratio*100:.1f}%",
                "intangible_to_assets": f"{intangible_to_assets*100:.1f}%",
                "minority_to_ni": f"{minority_to_ni*100:.1f}%",
                "major_shareholder_pledge_ratio": f"{notes.major_shareholder_pledge_ratio*100:.1f}%",
                "external_guarantee_to_equity": f"{notes.external_guarantee_to_equity*100:.1f}%",
                "negative_fcf_periods": f"{neg_fcf_count}/{len(dossier.periods)}",
                "audit_opinion": notes.audit_opinion[:40],
            },
        )
