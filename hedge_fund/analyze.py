"""Single-stock A-share & multi-master forensic diligence workbench (`aihf analyze`).

Respects the original v2 architecture (`DataClient` -> `FundamentalsSnapshot` +
`ForensicAccountingGate` -> 5 `LLMAgent` investor personas) and presents a rich
Chinese/English financial terminal readout.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hedge_fund.data.tushare_client import TushareDataClient, normalize_ashare_code
from hedge_fund.llm import PromptCache, make_llm
from hedge_fund.signals.buffett import BuffettAgent
from hedge_fund.signals.druckenmiller import DruckenmillerAgent
from hedge_fund.signals.graham import GrahamAgent
from hedge_fund.signals.lynch import LynchAgent
from hedge_fund.signals.munger import MungerAgent

ALL_MASTER_SPECS = [
    ("buffett", "巴菲特", "护城河与长期复利", BuffettAgent),
    ("graham", "格雷厄姆", "安全边际与防御底线", GrahamAgent),
    ("munger", "查理·芒格", "逆向思维与排雷毒舌", MungerAgent),
    ("lynch", "彼得·林奇", "成长性价比 PEG", LynchAgent),
    ("druckenmiller", "德鲁肯米勒", "业绩拐点与非对称风控", DruckenmillerAgent),
]


class GeminiFreeRESTClient:
    """Zero-dependency REST client for Google Gemini Free Tier with key failover."""

    def __init__(self, model: str = "gemini-3-flash-preview") -> None:
        self.model = model
        keys = [
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GEMINI_API_KEY2"),
            os.environ.get("GOOGLE_API_KEY"),
        ]
        self._keys = [k.strip() for k in keys if k and k.strip()]
        if not self._keys:
            raise RuntimeError("Missing GEMINI_API_KEY / GOOGLE_API_KEY for Gemini REST client.")
        self._idx = 0
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        last_err = ""
        candidates = [self.model]
        if "gemini-3.1-flash-lite" not in candidates:
            candidates.append("gemini-3.1-flash-lite")
        for m in candidates:
            for attempt in range(len(self._keys) * 2):
                key = self._keys[(self._idx + attempt) % len(self._keys)]
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
                payload = {
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
                }
                try:
                    self.calls += 1
                    resp = requests.post(url, json=payload, timeout=45)
                    if resp.status_code == 200:
                        data = resp.json()
                        parts = data["candidates"][0]["content"]["parts"]
                        text = "\n".join(p.get("text", "") for p in parts if "text" in p).strip()
                        self.model = m
                        self._idx = (self._idx + 1) % len(self._keys)
                        return text
                    last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
                    if resp.status_code == 404:
                        break
                    time.sleep(2.0)
                except Exception as exc:
                    last_err = str(exc)
                    time.sleep(1.5)
        raise RuntimeError(f"Gemini REST failed: {last_err}")


def _resolve_llm(model: str | None):
    if model and not model.startswith("gemini"):
        return make_llm(model=model)
    if any(os.environ.get(k) for k in ("GEMINI_API_KEY", "GEMINI_API_KEY2", "GOOGLE_API_KEY")):
        return GeminiFreeRESTClient(model=model or "gemini-3-flash-preview")
    return make_llm(model=model)


def _sig_badge(signal: str, fail_closed: bool = False) -> Text:
    if fail_closed:
        return Text("FAIL-CLOSED 熔断拒收", style="bold white on red")
    s = signal.lower()
    if s == "bullish":
        return Text("BULLISH 看多", style="bold green")
    if s == "bearish":
        return Text("BEARISH 看空", style="bold red")
    return Text("NEUTRAL 中性", style="bold yellow")


def run_analyze(
    ticker: str,
    *,
    as_of: str | None = None,
    lang: str = "zh",
    agents_filter: str = "all",
    enable_forensic: bool = True,
    model: str | None = None,
    save_json: str | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    """Run live A-share PIT diligence with ForensicAccountingGate + 5 investor personas."""
    load_dotenv(override=False)
    console = console or Console()
    as_of_date = as_of or dt.date.today().isoformat()
    ts_code = normalize_ashare_code(ticker)

    client = TushareDataClient(enable_forensic=enable_forensic)
    llm = _resolve_llm(model)
    cache_dir = Path.home() / ".cache" / "aihf_analyze_prompts"
    cache = PromptCache(cache_dir)

    if agents_filter.lower() in ("all", "5", ""):
        selected = ALL_MASTER_SPECS
    else:
        wanted = {x.strip().lower() for x in agents_filter.split(",") if x.strip()}
        selected = [s for s in ALL_MASTER_SPECS if s[0] in wanted]
        if not selected:
            raise ValueError(f"Unknown agents filter: {agents_filter}")

    t0 = time.monotonic()
    with console.status(f"[cyan]正在通过 Tushare Pro 拉取 {ts_code} 截至 {as_of_date} 的法定三表、审计意见与质押数据…", spinner="dots"):
        mkt = client.get_market_snapshot_info(ts_code, as_of_date)
        # Always pull full dossier for table display; forensic gate only injected when enable_forensic=True
        display_client = client if enable_forensic else TushareDataClient(enable_forensic=True)
        dossier = display_client.get_statement_dossier(ts_code, as_of_date, limit=5)

    # 1. Header panel
    mode_tag = "[bold green]二开完整版（确定性排雷安检门 ON）[/]" if enable_forensic else "[bold red]原版对比模式（排雷门禁 OFF · 仅 12 列表面比率）[/]"
    mv_str = f"{mkt['total_mv_billion']:.1f} 亿元" if mkt["total_mv_billion"] is not None else "-"
    pe_str = f"{mkt['pe_ttm']:.2f}x" if mkt["pe_ttm"] is not None else "-"
    pb_str = f"{mkt['pb']:.2f}x" if mkt["pb"] is not None else "-"
    px_str = f"¥{mkt['close']:.2f}" if mkt["close"] is not None else "-"
    header = (
        f"[bold white]{mkt['name']}[/] [cyan]({ts_code})[/]  |  行业：[bold]{mkt['industry']}[/]  |  分析基准日：[bold]{as_of_date}[/]\n"
        f"收盘价({mkt['trade_date']})：[bold]{px_str}[/]  |  总市值：[bold]{mv_str}[/]  |  PE(TTM)：[bold]{pe_str}[/]  |  PB：[bold]{pb_str}[/]\n"
        f"运行模式：{mode_tag}  |  大模型：[dim]{llm.model}[/]"
    )
    console.print(Panel(header, title="Fin-NCAP × AI Hedge Fund · A 股深度尽调工作台", border_style="cyan", box=box.ROUNDED))

    # 2. 5-year 3-statement snapshot table
    if dossier and dossier.periods:
        fin_tbl = Table(title="近 5 年法定合并年报核心三表科目（单位：亿元）", box=box.SIMPLE_HEAVY, header_style="bold cyan")
        for col, just in [
            ("报告期", "left"), ("公告日", "left"), ("营收", "right"), ("归母净利", "right"),
            ("ROE", "right"), ("毛利率", "right"), ("货币资金", "right"), ("有息负债", "right"),
            ("存货", "right"), ("应收账款", "right"), ("商誉", "right"), ("经营现金流", "right"), ("自由现金流", "right"),
        ]:
            fin_tbl.add_column(col, justify=just)
        for p in dossier.periods:
            # dossier units are CNY millions -> / 100 = 亿元
            fin_tbl.add_row(
                p.report_period,
                p.filing_date,
                f"{p.revenue/100:.1f}",
                f"{p.net_income/100:.1f}",
                f"{(p.return_on_equity or 0)*100:.1f}%",
                f"{(p.gross_margin or 0)*100:.1f}%",
                f"{p.cash_and_equivalents/100:.1f}",
                f"{p.interest_bearing_debt/100:.1f}",
                f"{p.inventory/100:.1f}",
                f"{p.accounts_receivable/100:.1f}",
                f"{p.goodwill/100:.1f}",
                f"{p.operating_cash_flow/100:.1f}",
                f"{p.free_cash_flow/100:.1f}",
            )
        console.print(fin_tbl)

    # 3. Run 5 master agents through exact v2 LLMAgent pipeline
    first_agent = selected[0][3](llm=llm, cache=cache, blind=False, lang=lang)
    snap = first_agent.build_snapshot(ts_code, as_of_date, client)
    audit = snap.forensic_audit

    if enable_forensic and audit is not None:
        km = audit.key_metrics
        if audit.fail_closed:
            border, status_str = "bold red", "[bold white on red] FAIL-CLOSED 一票否决（0次LLM短路拒收） [/]"
        elif audit.risk_cap == "BEARISH_CAP":
            border, status_str = "red", f"[bold red] BEARISH_CAP 高危红灯（严禁越权喊多 · 风险分 {audit.severity_score}） [/]"
        elif audit.red_flags:
            border, status_str = "yellow", f"[bold yellow] WARNING 局部预警（风险分 {audit.severity_score}） [/]"
        else:
            border, status_str = "green", "[bold green] PASS 绿灯通过（全部确定性财务排雷核验正常） [/]"

        audit_lines = [
            f"安检门裁决：{status_str}",
            f"核心校验：三表勾稽差 {km.get('bs_identity_gap_ratio')} | 现金/资产 {km.get('cash_to_assets')} | 有息债/资产 {km.get('debt_to_assets')} | "
            f"存款隐含收益 {km.get('implied_cash_yield')} | 商誉/权益 {km.get('goodwill_to_equity')} | 累计净现比 {km.get('ocf_to_ni_cumulative')} | "
            f"DSO {km.get('dso_latest')}天 | 总股本质押 {km.get('major_shareholder_pledge_ratio')} | 审计意见：{km.get('audit_opinion')}",
        ]
        if audit.red_flags:
            audit_lines.append("")
            for rf in audit.red_flags:
                tag_style = "bold white on red" if rf.severity == "FAIL_CLOSED" else ("bold red" if rf.severity == "CRITICAL" else "bold yellow")
                audit_lines.append(f"  [{tag_style}] {rf.severity} [/] [bold]{rf.title}[/] ({rf.code}) -> {rf.metric_value}  [dim](阈值 {rf.threshold})[/]")
        console.print(Panel("\n".join(audit_lines), title="确定性财务排雷安检门（ForensicAccountingGate · 纯 Python 零幻觉）", border_style=border))
    else:
        console.print(Panel("[yellow]已关闭确定性排雷门禁（--no-forensic），5 位大师此刻仅能看到原版 12 列表面财务比率，用于现场对比原版盲区。[/]", border_style="yellow"))

    # 4. Invoke all selected masters
    agent_results: list[dict[str, Any]] = []
    for key, zh_name, style_desc, cls in selected:
        ag = cls(llm=llm, cache=cache, blind=False, lang=lang)
        with console.status(f"[cyan]正在请教 {zh_name}（{style_desc}）…", spinner="dots"):
            st = time.monotonic()
            sig = ag.predict(ts_code, as_of_date, client)
            elapsed = round(time.monotonic() - st, 2)
        md = sig.metadata
        rec = {
            "agent": key,
            "zh_name": zh_name,
            "style": style_desc,
            "signal": md.get("signal", "neutral"),
            "confidence": md.get("confidence", 0.0),
            "value": round(sig.value, 4),
            "abstained": bool(md.get("abstained", False)),
            "fail_closed": bool(md.get("fail_closed", False)),
            "risk_cap_overridden": bool(md.get("risk_cap_overridden", False)),
            "llm_raw_signal": md.get("llm_raw_signal"),
            "reasoning": sig.reasoning,
            "elapsed_sec": elapsed,
        }
        agent_results.append(rec)
        if not rec["fail_closed"] and not md.get("cached"):
            time.sleep(0.5)

    # Render 5-master table
    mtbl = Table(title=f"投资大师联合尽调建议（{len(agent_results)} 位全员上场）", box=box.ROUNDED, show_lines=True, header_style="bold magenta")
    mtbl.add_column("投资大师", width=16)
    mtbl.add_column("裁决信号", width=18, justify="center")
    mtbl.add_column("置信度", width=8, justify="right")
    mtbl.add_column("深度尽调理由（基于公告日可公开财务事实）")

    for r in agent_results:
        badge = _sig_badge(r["signal"], r["fail_closed"])
        if r["risk_cap_overridden"]:
            badge.append("\n[门禁硬封顶]", style="dim red")
        mtbl.add_row(
            f"[bold]{r['zh_name']}[/]\n[dim]{r['style']}[/]",
            badge,
            f"{r['confidence']:.0f}%",
            r["reasoning"],
        )
    console.print(mtbl)

    # Consensus calculation
    score_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    vals = [score_map.get(r["signal"], 0.0) * (r["confidence"] / 100.0) for r in agent_results if not r["abstained"]]
    avg_score = round(sum(vals) / len(vals), 4) if vals else 0.0
    fail_closed_any = any(r["fail_closed"] for r in agent_results)
    if fail_closed_any:
        consensus = "FAIL_CLOSED"
    elif avg_score >= 0.15:
        consensus = "BULLISH"
    elif avg_score <= -0.15:
        consensus = "BEARISH"
    else:
        consensus = "NEUTRAL"

    total_sec = round(time.monotonic() - t0, 2)
    cons_color = "red" if consensus in ("BEARISH", "FAIL_CLOSED") else ("green" if consensus == "BULLISH" else "yellow")
    console.print(
        Panel(
            f"综合共识结论：[bold {cons_color}]{consensus}[/]（加权得分 {avg_score:+.2f}）  |  "
            f"总耗时：{total_sec:.1f}s  |  LLM 调用：{getattr(llm, 'calls', 0)} 次\n"
            "[dim]免责声明：本工具仅基于公开历史财务数据做开源软件工程评测与财务排雷研究演示，不构成任何证券投资建议或收益承诺。[/]",
            border_style=cons_color,
        )
    )

    payload = {
        "ticker": ts_code,
        "market_info": mkt,
        "as_of": as_of_date,
        "enable_forensic": enable_forensic,
        "lang": lang,
        "model": llm.model,
        "consensus": consensus,
        "consensus_score": avg_score,
        "fail_closed": fail_closed_any,
        "forensic_audit": audit.model_dump() if audit is not None else None,
        "periods": [p.model_dump() for p in dossier.periods] if dossier else [],
        "agent_results": agent_results,
        "elapsed_total_sec": total_sec,
        "llm_calls": getattr(llm, "calls", 0),
    }
    if save_json:
        out_p = Path(save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[dim]已保存完整结构化尽调报告至：{out_p}[/]")
    return payload
