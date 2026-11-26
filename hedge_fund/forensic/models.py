"""Typed schemas for full 3-statement financial dossiers and forensic audit outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FinancialStatementPeriod(BaseModel):
    """One publicly filed period containing income, balance-sheet and cash-flow lines."""

    report_period: str
    filing_date: str
    market_cap: float | None = None

    # Income statement
    revenue: float
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float
    minority_interest_income: float = 0.0
    interest_income: float = 0.0
    interest_expense: float = 0.0

    # Cash flow statement
    operating_cash_flow: float
    capex: float = 0.0
    free_cash_flow: float

    # Balance sheet
    cash_and_equivalents: float
    interest_bearing_debt: float
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    goodwill: float = 0.0
    intangible_and_dev_assets: float = 0.0
    total_assets: float
    total_liabilities: float
    shareholders_equity: float
    current_assets: float | None = None
    current_liabilities: float | None = None

    # Per-share / precomputed ratios (optional; derived automatically if absent)
    shares_diluted: float | None = None
    revenue_growth: float | None = None
    price_to_earnings_ratio: float | None = None
    return_on_equity: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None


class AuditAndGovernanceNotes(BaseModel):
    """Structured public footnote & governance disclosures."""

    audit_opinion: str = "Standard Unqualified"
    restricted_cash_ratio: float = 0.0
    major_shareholder_pledge_ratio: float = 0.0
    external_guarantee_to_equity: float = 0.0
    related_party_notes: str = ""


class FinancialStatementDossier(BaseModel):
    """Full point-in-time public financial dossier for a company (newest period first)."""

    ticker: str
    sector: str | None = None
    industry: str | None = None
    currency: str = "CNY"
    periods: list[FinancialStatementPeriod]
    notes_and_audit: AuditAndGovernanceNotes = Field(default_factory=AuditAndGovernanceNotes)


class ForensicRedFlag(BaseModel):
    """Single deterministic financial anomaly backed by formula and observed value."""

    code: str
    severity: Literal["FAIL_CLOSED", "CRITICAL", "WARNING"]
    title: str
    metric_value: str
    threshold: str
    evidence: str


class ForensicAuditResult(BaseModel):
    """Deterministic pre-LLM financial audit verdict and structured metrics."""

    ticker: str
    latest_period: str
    fail_closed: bool = False
    fail_closed_reasons: list[str] = Field(default_factory=list)
    risk_cap: Literal["FAIL_CLOSED", "BEARISH_CAP", "NONE"] = "NONE"
    severity_score: int = 0
    red_flags: list[ForensicRedFlag] = Field(default_factory=list)
    key_metrics: dict[str, float | str | None] = Field(default_factory=dict)

    @property
    def flag_codes(self) -> list[str]:
        return [f.code for f in self.red_flags]

    def render_prompt_block(self, blind: bool = False) -> str:
        """Compact deterministic audit summary injected into LLM prompts."""
        lines = [
            "=== DETERMINISTIC FORENSIC ACCOUNTING AUDIT (PYTHON-VERIFIED) ===",
            f"Gate Status: {self.risk_cap} | Fail-Closed: {self.fail_closed} | Severity Score: {self.severity_score}",
        ]
        # Only expose scale-free ratios in blind mode; include all ratios in live mode
        km = self.key_metrics
        ratio_items = [
            ("BS Identity Gap Ratio", km.get("bs_identity_gap_ratio")),
            ("Cash/Assets", km.get("cash_to_assets")),
            ("Interest-Bearing Debt/Assets", km.get("debt_to_assets")),
            ("Implied Cash Yield", km.get("implied_cash_yield")),
            ("Restricted Cash Ratio", km.get("restricted_cash_ratio")),
            ("OCF/NetIncome (Latest)", km.get("ocf_to_ni_latest")),
            ("OCF/NetIncome (Multi-Yr)", km.get("ocf_to_ni_cumulative")),
            ("DSO (Days)", km.get("dso_latest")),
            ("Inventory/Assets", km.get("inventory_to_assets")),
            ("Goodwill/Equity", km.get("goodwill_to_equity")),
            ("Tangible Equity/Book Equity", km.get("tangible_equity_ratio")),
            ("Intangibles+Dev/Assets", km.get("intangible_to_assets")),
            ("Minority Loss vs NetIncome", km.get("minority_to_ni")),
            ("Major Pledge Ratio", km.get("major_shareholder_pledge_ratio")),
            ("External Guarantee/Equity", km.get("external_guarantee_to_equity")),
            ("Negative FCF Periods", km.get("negative_fcf_periods")),
            ("Audit Opinion", km.get("audit_opinion")),
        ]
        summary_pairs = [f"{k}={v}" for k, v in ratio_items if v is not None]
        lines.append("Verified Ratios: " + " | ".join(summary_pairs))

        if not self.red_flags:
            lines.append("Detected Red Flags: NONE (All deterministic accounting & governance checks passed).")
        else:
            lines.append(f"Detected Red Flags ({len(self.red_flags)}):")
            for rf in self.red_flags:
                evidence = rf.evidence if not blind else f"{rf.title} (value={rf.metric_value}, threshold={rf.threshold})"
                lines.append(f"  - [{rf.severity}] {rf.code}: {evidence}")
        lines.append("Hard Audit Rule: You MUST address every detected red flag above in your reasoning.")
        return "\n".join(lines)
