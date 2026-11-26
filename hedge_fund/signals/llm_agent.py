"""LLMAgent — base class for LLM investor agents (the second AlphaModel flavor).

An LLMAgent reasons over a point-in-time FundamentalsSnapshot in a persona's
voice and emits the same Signal every quant model does. The base class owns
all the machinery; a persona is just a name + a system prompt:

    class BuffettAgent(LLMAgent):
        @property
        def name(self) -> str:
            return "buffett"

        def get_system_prompt(self) -> str:
            return "You are Warren Buffett..."

Failure contract (locked decisions):
- Data-layer errors PROPAGATE (fail loud — a broken snapshot must never
  silently become a neutral view).
- LLM call/parse failures ABSTAIN: Signal(value=0.0, metadata.abstained=True).
- Every LLM decision persists its exact prompt + response (via PromptCache),
  and an unchanged snapshot never pays for a second LLM call.

At a bar (a minute fund's session) the prompt also carries the day so far
— the move since the previous close, results released, headlines — see
features/intraday.py. It is bucketed so quiet bars stay cache hits.
"""

from __future__ import annotations

import logging
import re

from hedge_fund.data.bars import is_intraday
from hedge_fund.data.protocol import DataClient
from hedge_fund.features.breakpoints import MEBreakpoints, shared_breakpoints
from hedge_fund.features.intraday import build_intraday_context
from hedge_fund.features.snapshot import FundamentalsSnapshot, InsufficientData, build_snapshot
from hedge_fund.llm import LLMCallError, LLMClient, PromptCache, extract_json, make_llm, prompt_key
from hedge_fund.models import Signal
from hedge_fund.signals.base import AlphaModel

logger = logging.getLogger(__name__)

# What the model must return; folded into Signal.value below.
_SIGNAL_TO_SIGN = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


class LLMAgent(AlphaModel):
    """Base for persona agents. Subclasses define `name` and `get_system_prompt`."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        cache: PromptCache | None = None,
        blind: bool = False,
        breakpoints: MEBreakpoints | None = None,
    ) -> None:
        self._llm = llm if llm is not None else make_llm()
        self._cache = cache if cache is not None else PromptCache()
        # Blind prompts withhold the ticker, industry, calendar dates and
        # dollar values so a backtest can't lean on what the LLM remembers
        # about the company (FundamentalsSnapshot.render). Size survives only
        # as a market-cap percentile from a breakpoints table, loaded once per
        # process on first use unless one is injected (tests). Backtests set
        # blind; live runs don't and never touch the breakpoints.
        self._blind = blind
        self._breakpoints = breakpoints

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        try:
            snapshot = self.build_snapshot(ticker, date, data_client)
        except InsufficientData as exc:
            return self._abstain(ticker, date, f"insufficient data: {exc}")
        # Any other data-layer exception (e.g. FDClientError) propagates.

        # Hard deterministic Fail-Closed gate ("A Prompt Is Not a Control"):
        # broken accounting identity, non-standard audit opinion, or excessive
        # external guarantees reject immediately without delegating to an LLM.
        audit = snapshot.forensic_audit
        if audit is not None and audit.fail_closed:
            reasons_txt = "; ".join(audit.fail_closed_reasons)
            return Signal(
                model_name=self.name,
                ticker=ticker,
                date=date,
                value=-1.0,
                reasoning=f"FAIL-CLOSED deterministic rejection: {reasons_txt}",
                metadata={
                    "signal": "bearish",
                    "confidence": 100.0,
                    "model": "deterministic_forensic_gate",
                    "snapshot_hash": snapshot.content_hash,
                    "cached": False,
                    "abstained": False,
                    "fail_closed": True,
                    "risk_cap": "FAIL_CLOSED",
                    "detected_red_flags": audit.flag_codes,
                    "gate_reasons": audit.fail_closed_reasons,
                },
            )

        system = self.get_system_prompt()
        user = self.build_user_prompt(snapshot)
        if is_intraday(date):
            # A bar, not a close: add the day so far. Bucketed, so a quiet
            # bar renders the same prompt as the last one and costs nothing.
            context = build_intraday_context(ticker, date, data_client)
            user = f"{user}\n\n{context.render(blind=self._blind)}"
        cache_key = getattr(self._llm, "cache_key", None)
        key = (cache_key(self.name, system, user) if callable(cache_key)
               else prompt_key(self.name, self._llm.model, system, user))

        cached = self._cache.get(key)
        if cached is not None and "parsed" in cached:
            return self._to_signal(ticker, date, cached["parsed"], key, snapshot, cached=True)

        record = {
            "agent": self.name,
            "model": self._llm.model,
            "ticker": ticker,
            "as_of": date,
            "snapshot_hash": snapshot.content_hash,
            "system": system,
            "user": user,
        }

        try:
            response = self._llm.complete(system, user)
        except Exception as exc:
            if isinstance(exc, LLMCallError) and exc.diagnostic_record is not None:
                self._cache.put(key, {**record, "call_error": str(exc),
                                      "diagnostics": exc.diagnostic_record})
                signal = self._abstain(ticker, date, f"LLM call failed: {exc}")
                signal.metadata["prompt_key"] = key
            else:
                signal = self._abstain(ticker, date, f"LLM call failed: {exc}")
            logger.warning("%s LLM call failed for %s@%s: %s", self.name, ticker, date, exc)
            return signal

        record["response"] = response
        try:
            parsed = self._parse(response)
        except Exception as exc:
            # Persist the raw response even when unparseable — the debug trail.
            self._cache.put(key, {**record, "parse_error": str(exc)})
            logger.warning("%s parse failed for %s@%s: %s", self.name, ticker, date, exc)
            return self._abstain(ticker, date, f"parse failed: {exc}")

        self._cache.put(key, {**record, "parsed": parsed})
        return self._to_signal(ticker, date, parsed, key, snapshot, cached=False)

    # ------------------------------------------------------------------
    # Subclass surface
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """The persona — every subclass must define its voice."""
        raise NotImplementedError(f"{type(self).__name__} must define get_system_prompt()")

    def build_snapshot(self, ticker: str, date: str, data_client: DataClient) -> FundamentalsSnapshot:
        """What this persona is allowed to know. Default: the shared
        point-in-time fundamentals snapshot — right for value/quality
        personas. Override for personas that reason over different data
        (macro, news); when a second snapshot TYPE exists, extract the
        implicit interface (ticker/as_of/content_hash/render) into a
        Protocol — not before."""
        breakpoints = None
        if self._blind:
            if self._breakpoints is None:
                self._breakpoints = shared_breakpoints()
            breakpoints = self._breakpoints
        return build_snapshot(ticker, date, data_client, breakpoints=breakpoints)

    def build_user_prompt(self, snapshot: FundamentalsSnapshot) -> str:
        """Default user prompt: the rendered snapshot. Override to enrich."""
        return snapshot.render(blind=self._blind)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, response: str) -> dict:
        """Extract + validate {signal, confidence, reasoning}.

        Resiliently recovers truncated JSON objects (e.g., cut off mid-reasoning
        by max_output_tokens) only when an explicit JSON object prefix with
        valid `signal` and `confidence` fields is present; pure prose still
        raises and abstains.
        """
        try:
            data = extract_json(response)
            truncated_recovered = False
        except Exception:
            data = self._recover_truncated_json(response)
            if data is None:
                raise
            truncated_recovered = True

        signal = str(data.get("signal", "")).lower()
        if signal not in _SIGNAL_TO_SIGN:
            raise ValueError(f"invalid signal {data.get('signal')!r}")
        confidence = float(data.get("confidence", 0))
        if not 0 <= confidence <= 100:
            raise ValueError(f"confidence out of range: {confidence}")
        parsed = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": str(data.get("reasoning", "")),
        }
        if truncated_recovered:
            parsed["truncated_recovered"] = True
        if "provider_metadata" in data:
            if not isinstance(data["provider_metadata"], dict):
                raise ValueError("provider_metadata must be an object")
            parsed["provider_metadata"] = data["provider_metadata"]
        return parsed

    @staticmethod
    def _recover_truncated_json(response: str) -> dict | None:
        """Recover signal/confidence/partial reasoning from a mid-stream truncated JSON object."""
        if "{" not in response or '"signal"' not in response or '"confidence"' not in response:
            return None
        m_sig = re.search(r'"signal"\s*:\s*"(bullish|bearish|neutral)"', response, re.I)
        m_conf = re.search(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)', response)
        if not m_sig or not m_conf:
            return None
        m_reason = re.search(r'"reasoning"\s*:\s*"([^"]*)', response)
        reasoning = (m_reason.group(1).strip() + " [truncated]") if m_reason else "[truncated response recovered]"
        return {
            "signal": m_sig.group(1).lower(),
            "confidence": float(m_conf.group(1)),
            "reasoning": reasoning,
        }

    def _to_signal(
        self,
        ticker: str,
        date: str,
        parsed: dict,
        key: str,
        snapshot: FundamentalsSnapshot,
        cached: bool,
    ) -> Signal:
        raw_signal = parsed["signal"]
        final_signal = raw_signal
        confidence = parsed["confidence"]
        reasoning = parsed["reasoning"]

        audit = snapshot.forensic_audit
        risk_cap_overridden = False
        detected_flags = audit.flag_codes if audit is not None else []
        risk_cap = audit.risk_cap if audit is not None else "NONE"

        if audit is not None and audit.risk_cap == "BEARISH_CAP" and final_signal != "bearish":
            risk_cap_overridden = True
            final_signal = "bearish"
            confidence = max(confidence, 85.0)
            reasoning = (
                f"[DETERMINISTIC RISK CAP OVERRIDE: LLM raw={raw_signal} blocked by "
                f"{len(detected_flags)} red flags ({', '.join(detected_flags)})] {reasoning}"
            )

        value = _SIGNAL_TO_SIGN[final_signal] * confidence / 100.0
        meta: dict = {
            "signal": final_signal,
            "confidence": confidence,
            "model": self._llm.model,
            "prompt_key": key,
            "snapshot_hash": snapshot.content_hash,
            "cached": cached,
            "abstained": False,
        }
        if audit is not None:
            meta["fail_closed"] = audit.fail_closed
            meta["risk_cap"] = risk_cap
            meta["detected_red_flags"] = detected_flags
            meta["llm_raw_signal"] = raw_signal
            meta["risk_cap_overridden"] = risk_cap_overridden
        if parsed.get("truncated_recovered"):
            meta["truncated_recovered"] = True
        if "provider_metadata" in parsed:
            meta["provider_metadata"] = parsed["provider_metadata"]

        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=reasoning,
            metadata=meta,
        )

    def _abstain(self, ticker: str, date: str, reason: str) -> Signal:
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=0.0,
            reasoning=f"abstained: {reason}",
            metadata={"abstained": True, "abstain_reason": reason, "cached": False},
        )
