#!/usr/bin/env python3
"""Deterministically score A-share short-term entry or position signals.

The script enforces data gates and hard vetoes before applying any score.
It reads one JSON object from a file path or stdin and writes JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ENTRY_WEIGHTS = {
    "market": 10,
    "theme": 15,
    "identity": 20,
    "expectation": 15,
    "structure": 20,
    "flow": 10,
    "risk_reward": 10,
}

DATA_GATES = {"stale_quote", "timestamp_missing", "insufficient_data"}
ENTRY_VETOES = {
    "non_core_deep_water",
    "theme_decline",
    "support_broken",
    "accelerating_outflow",
    "liquidity_trap",
    "single_indicator_only",
}

SELL_SEVERITY = {
    "low_open_after_long_upper": 2,
    "sharp_corner": 2,
    "price_volume_divergence": 1,
    "excessive_positive_deviation": 1,
    "rebound_fails_vwap": 2,
    "rebound_fails_zero": 2,
    "limit_up_break": 2,
    "auction_below_expectation": 2,
    "high_turnover_low_open": 3,
    "edge_selloff": 3,
    "support_break": 3,
}


def load_input(path: str | None) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Input must be one JSON object")
    return data


def require_score(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
        raise ValueError(f"scores.{name} must be an integer from 0 to 3")
    return value


def evaluate_entry(data: dict[str, Any]) -> dict[str, Any]:
    flags = set(data.get("flags", []))
    unknown_flags = flags - DATA_GATES - ENTRY_VETOES
    if unknown_flags:
        raise ValueError(f"Unknown flags: {sorted(unknown_flags)}")

    gates = sorted(flags & DATA_GATES)
    if gates:
        return {
            "mode": "entry",
            "decision": "等待数据",
            "score": None,
            "reasons": gates,
            "note": "数据门槛优先于评分",
        }

    vetoes = set(flags & ENTRY_VETOES)
    setup = data.get("setup")
    # A known structural veto does not require invented scores to become valid.
    if vetoes:
        return {
            "mode": "entry",
            "decision": "回避",
            "score": None,
            "reasons": sorted(vetoes),
            "note": "结构性否决不能被其他加分项抵消",
        }
    scores = data.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("scores must be an object")
    parsed = {name: require_score(name, scores.get(name)) for name in ENTRY_WEIGHTS}

    if setup == "deep_water" and parsed["identity"] < 3:
        vetoes.add("non_core_deep_water")
    if vetoes:
        return {
            "mode": "entry",
            "decision": "回避",
            "score": None,
            "reasons": sorted(vetoes),
            "note": "结构性否决不能被其他加分项抵消",
        }

    total = round(
        sum(parsed[name] / 3 * weight for name, weight in ENTRY_WEIGHTS.items()), 1
    )
    if total < 45:
        decision = "回避"
    elif total < 60:
        decision = "观察"
    elif total < 75:
        decision = "只允许试错"
    elif total < 85:
        decision = "等待触发后确认"
    else:
        decision = "高质量候选，仍须条件触发"

    return {
        "mode": "entry",
        "setup": setup,
        "decision": decision,
        "score": total,
        "scores": parsed,
        "note": "评分只衡量证据完整度，不代表胜率或收益保证",
    }


def evaluate_position(data: dict[str, Any]) -> dict[str, Any]:
    signals = data.get("sell_signals", [])
    if not isinstance(signals, list):
        raise ValueError("sell_signals must be an array")
    unknown = set(signals) - set(SELL_SEVERITY)
    if unknown:
        raise ValueError(f"Unknown sell signals: {sorted(unknown)}")

    flow_state = data.get("flow_state", "neutral")
    if flow_state not in {"inflow", "neutral", "outflow"}:
        raise ValueError("flow_state must be inflow, neutral, or outflow")

    if not signals:
        return {
            "mode": "position",
            "decision": "继续观察",
            "severity": 0,
            "signals": [],
            "note": "没有卖出事件不等于应该继续持有；仍需核对原交易预期",
        }

    severity = max(SELL_SEVERITY[signal] for signal in signals)
    if flow_state == "outflow" and any(
        signal in {"limit_up_break", "price_volume_divergence", "sharp_corner"}
        for signal in signals
    ):
        severity = min(3, severity + 1)

    hard_combo = "support_break" in signals and flow_state == "outflow"
    continuation_combo = (
        "low_open_after_long_upper" in signals
        and data.get("price_action") == "continues_down"
    )

    if hard_combo or continuation_combo or severity >= 3:
        decision = "退出优先"
    elif severity == 2:
        decision = "至少减仓"
    else:
        decision = "分批锁利"

    return {
        "mode": "position",
        "decision": decision,
        "severity": severity,
        "signals": signals,
        "flow_state": flow_state,
        "note": "资金流入可调整减仓幅度，但不能取消已发生的风险事件",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="JSON file; omit to read stdin")
    args = parser.parse_args()
    try:
        data = load_input(args.input)
        mode = data.get("mode")
        if mode == "entry":
            result = evaluate_entry(data)
        elif mode == "position":
            result = evaluate_position(data)
        else:
            raise ValueError("mode must be entry or position")
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        json.dump({"error": str(exc)}, sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
