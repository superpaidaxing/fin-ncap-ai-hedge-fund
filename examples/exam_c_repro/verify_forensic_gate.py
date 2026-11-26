#!/usr/bin/env python3
"""零 API Key 一键验证确定性财务排雷门禁（Exam C v1.0 离线复现脚本）。"""
from __future__ import annotations

import json
from pathlib import Path

from hedge_fund.forensic import ForensicAccountingGate, LocalFinancialStatementClient

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.json"


def main() -> None:
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    client = LocalFinancialStatementClient.from_dataset_file(DATASET)
    gate = ForensicAccountingGate()
    print(f"Exam: {raw['exam_id']} v{raw['version']} ({len(raw['cases'])} cases)")
    print("-" * 78)
    for c in raw["cases"]:
        dossier = client.get_statement_dossier(c["ticker"], c["as_of"])
        res = gate.evaluate(dossier)
        print(
            f"{c['case_id']} | expected={c['expected_verdict']:11s} | "
            f"risk_cap={res.risk_cap:11s} | fail_closed={str(res.fail_closed):5s} | "
            f"flags({len(res.flag_codes)})={','.join(res.flag_codes) or 'NONE'}"
        )


if __name__ == "__main__":
    main()
