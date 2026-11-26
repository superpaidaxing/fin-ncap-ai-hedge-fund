#!/usr/bin/env python3
"""考卷 C（exam_c_fin_diligence_v1）确定性自动评分引擎。

评分不依赖任何 LLM 裁判，100% 确定性计算 4 大硬指标：
1. 排雷拦截准确率 (risk_interception_rate, 权重 40%):
   对 5 家真实雷区（CASE_01~CASE_05），模型综合裁决必须为 bearish / fail_closed（绝不能 bullish 放行）。
2. 白马零误杀率 (white_horse_specificity, 权重 15%):
   对阴性对照组（CASE_06），不能误判为财务造假雷或 fail_closed（应 bullish 或 neutral）。
3. 核心财务红旗召回率 (red_flag_recall, 权重 30%):
   对照每家 Ground Truth 必检出红旗清单（共 18 个具体财务排雷标签），检查输出的结构化 flags 或推理文本是否真正命中核心病灶。
4. 勾稽异常 Fail-Closed 硬拦截率 (accounting_fail_closed, 权重 15%):
   对 CASE_05（资产负债表存在 2.3 亿元勾稽不平 + 保留审计意见），系统是否触发确定性硬拒绝/Fail-Closed，而非照单全收输出常规多空研报。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
DATASET_PATH = BENCH_DIR / "dataset.json"

# 允许通过结构化 flag ID 精确命中，或在原版无结构化输出时通过中英文关键词严格核查推理文本是否真正指出了该病灶
RED_FLAG_KEYWORDS: dict[str, list[str]] = {
    "CASH_DEBT_DOUBLE_HIGH": ["存贷双高", "cash and debt", "high cash", "high debt", "monetary funds", "货币资金.*负债", "simultaneous.*cash.*debt"],
    "ABNORMAL_LOW_INTEREST_YIELD": ["利息收入", "收益率", "interest income", "interest yield", "deposit rate", "存款利率", "0.79%", "0.86%"],
    "INVENTORY_BLOATING": ["存货", "inventory", "inventories"],
    "OCF_NET_INCOME_DIVERGENCE": ["净现比", "经营.*现金流", "operating cash flow", "cash flow.*net income", "earnings quality", "cash conversion"],
    "RESTRICTED_CASH_OR_POOLING": ["受限", "归集", "restricted cash", "cash pooling", "pledged deposit", "集团资金"],
    "RECEIVABLES_DSO_SURGE": ["应收", "receivable", "dso", "days sales outstanding"],
    "FCF_CHRONIC_NEGATIVE": ["自由现金流.*负", "negative.*free cash flow", "negative fcf", "fcf.*negative", "burning cash", "现金流.*负"],
    "INTANGIBLE_CAPITALIZATION_HIGH": ["无形资产", "研发资本化", "intangible", "capitaliz"],
    "MINORITY_LOSS_MASKING": ["少数股东", "minority interest", "non-controlling"],
    "GOODWILL_OVERHANG": ["商誉", "goodwill"],
    "PLEDGE_RATIO_CRITICAL": ["质押", "pledge"],
    "ACCOUNTING_IDENTITY_BROKEN": ["勾稽", "不平", "balance sheet.*not balance", "accounting identity", "assets.*liabilities.*equity", "2.3"],
    "NON_STANDARD_AUDIT_OPINION": ["保留意见", "非标", "qualified opinion", "audit opinion", "auditor"],
    "EXCESSIVE_EXTERNAL_GUARANTEE": ["对外担保", "违规担保", "external guarantee", "guarantee"],
}


def dataset_sha256() -> str:
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()


def _detect_flags_in_record(rec: dict[str, Any], must_flags: list[str]) -> tuple[list[str], list[str]]:
    explicit = {str(x).upper() for x in (rec.get("detected_red_flags") or [])}
    text_blob = " ".join(
        [
            str(rec.get("consensus_reasoning", "")),
            *[str(a.get("reasoning", "")) for a in (rec.get("agent_results") or [])],
            *[str(x) for x in (rec.get("gate_reasons") or [])],
        ]
    ).lower()

    hit: list[str] = []
    missed: list[str] = []
    for flag in must_flags:
        if flag in explicit:
            hit.append(flag)
            continue
        pats = RED_FLAG_KEYWORDS.get(flag, [])
        if any(re.search(p, text_blob, re.I) for p in pats):
            hit.append(flag)
        else:
            missed.append(flag)
    return hit, missed


def score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    ds = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    by_id = {p["case_id"]: p for p in predictions}
    weights = ds["scoring_weights"]

    risk_total = 0
    risk_intercepted = 0
    bullish_on_fraud = 0
    white_total = 0
    white_passed = 0
    total_flags = 0
    hit_flags = 0
    fc_total = 0
    fc_passed = 0

    case_diffs: list[dict[str, Any]] = []

    for case in ds["cases"]:
        cid = case["case_id"]
        rec = by_id.get(cid, {})
        verdict = str(rec.get("consensus_signal", "missing")).lower()
        fail_closed = bool(rec.get("fail_closed_triggered", False))
        must_flags = case["must_detect_red_flags"]
        hit, missed = _detect_flags_in_record(rec, must_flags)

        total_flags += len(must_flags)
        hit_flags += len(hit)

        expected = case["expected_verdict"]
        verdict_ok = False
        if expected in ("MUST_REJECT", "FAIL_CLOSED"):
            risk_total += 1
            if fail_closed or verdict == "bearish":
                risk_intercepted += 1
                verdict_ok = True
            if verdict == "bullish" and not fail_closed:
                bullish_on_fraud += 1
        elif expected == "PASS_CLEAN":
            white_total += 1
            if (not fail_closed) and verdict in ("bullish", "neutral"):
                white_passed += 1
                verdict_ok = True

        if case.get("requires_fail_closed"):
            fc_total += 1
            if fail_closed:
                fc_passed += 1

        case_diffs.append(
            {
                "case_id": cid,
                "ticker": case["ticker"],
                "archetype": case["archetype"],
                "expected_verdict": expected,
                "actual_signal": verdict,
                "fail_closed_triggered": fail_closed,
                "verdict_pass": verdict_ok,
                "must_flags_count": len(must_flags),
                "hit_flags": hit,
                "missed_flags": missed,
                "agent_signals": {
                    a.get("agent", "?"): {
                        "signal": a.get("signal"),
                        "confidence": a.get("confidence"),
                        "abstained": a.get("abstained", False),
                    }
                    for a in (rec.get("agent_results") or [])
                },
            }
        )

    risk_rate = risk_intercepted / risk_total if risk_total else 0.0
    white_rate = white_passed / white_total if white_total else 1.0
    flag_recall = hit_flags / total_flags if total_flags else 0.0
    fc_rate = fc_passed / fc_total if fc_total else 0.0

    total_score = round(
        100.0
        * (
            weights["risk_interception_rate"] * risk_rate
            + weights["white_horse_specificity"] * white_rate
            + weights["red_flag_recall"] * flag_recall
            + weights["accounting_fail_closed"] * fc_rate
        ),
        2,
    )

    return {
        "exam_id": ds["exam_id"],
        "exam_version": ds["version"],
        "dataset_sha256": dataset_sha256(),
        "total_score": total_score,
        "metrics": {
            "risk_interception_rate": round(risk_rate, 4),
            "risk_intercepted": f"{risk_intercepted}/{risk_total}",
            "bullish_on_fraud_cases": bullish_on_fraud,
            "white_horse_specificity": round(white_rate, 4),
            "red_flag_recall": round(flag_recall, 4),
            "red_flags_hit": f"{hit_flags}/{total_flags}",
            "accounting_fail_closed_rate": round(fc_rate, 4),
            "fail_closed_hit": f"{fc_passed}/{fc_total}",
        },
        "case_diffs": case_diffs,
    }


if __name__ == "__main__":
    print(f"Exam C v1.0 dataset SHA256: {dataset_sha256()}")

