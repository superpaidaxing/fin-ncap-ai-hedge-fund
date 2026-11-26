"""Deterministic forensic accounting gate & open financial statement adapter.

Implements the financial risk principle "A Prompt Is Not a Control":
accounting identity verification, audit-opinion fail-closed checks, and
balance-sheet / cash-flow red-flag probes run deterministically in Python
before any LLM investor persona renders an opinion.
"""

from hedge_fund.forensic.gate import ForensicAccountingGate, ForensicThresholds
from hedge_fund.forensic.local_client import LocalFinancialStatementClient
from hedge_fund.forensic.models import (
    AuditAndGovernanceNotes,
    FinancialStatementDossier,
    FinancialStatementPeriod,
    ForensicAuditResult,
    ForensicRedFlag,
)

__all__ = [
    "AuditAndGovernanceNotes",
    "FinancialStatementDossier",
    "FinancialStatementPeriod",
    "ForensicAccountingGate",
    "ForensicAuditResult",
    "ForensicRedFlag",
    "ForensicThresholds",
    "LocalFinancialStatementClient",
]
