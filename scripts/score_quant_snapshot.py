#!/usr/bin/env python3
"""Q2.0 research candidate screening, not an execution/backtest engine.

Input: audited point-in-time ranks plus eight closed 5-minute bars.
See references/quant_strategy.md for units, provenance and trading rules.
No network, third-party packages, probability claims or automatic orders.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

VERSION = "Q2.0-research"
DATA_FLAGS = {
    "stale_quote", "timestamp_missing", "insufficient_data",
    "future_information", "membership_unverifiable",
}
VETO_FLAGS = {
    "theme_decline", "support_broken", "liquidity_trap", "major_event_risk",
    "auction_below_expectation", "final_day_no_entry",
}


class DataGate(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def obj(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise DataGate("insufficient_data", f"{name} must be an object")
    return value


def number(data: dict, name: str, low=None, high=None) -> float:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataGate("insufficient_data", f"{name} must be numeric, not missing/bool")
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) or (
        high is not None and value > high
    ):
        raise DataGate("insufficient_data", f"{name} is nonfinite or outside its domain")
    return value


def boolean(data: dict, name: str) -> bool:
    if not isinstance(data.get(name), bool):
        raise DataGate("insufficient_data", f"{name} must be bool")
    return data[name]


def stamp(value: Any, name: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise DataGate("timestamp_missing", f"{name} requires timezone-aware ISO8601")
    if result.utcoffset() != timedelta(hours=8):
        raise DataGate("timestamp_missing", f"{name} must use China time +08:00")
    return result


def trading_minute(end: datetime) -> int:
    minute = end.hour * 60 + end.minute
    if end.second or end.microsecond or minute % 5:
        raise DataGate("insufficient_data", "bar end is not on a 5-minute boundary")
    if 575 <= minute <= 690:  # 09:35 to 11:30 closed bars
        return minute - 570
    if 785 <= minute <= 900:  # 13:05 to 15:00; lunch excluded
        return minute - 780 + 120
    raise DataGate("insufficient_data", "bar end outside regular auction sessions")


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def response(decision: str, reasons: list[str], **extra: Any) -> dict:
    return {"version": VERSION, "decision": decision, "reasons": reasons, **extra}


def _score(data: dict) -> dict:
    if data.get("version") != VERSION:
        raise ValueError(f"version must be {VERSION}")
    setup = data.get("setup")
    if setup not in {"red_pullback", "green_reclaim"}:
        raise ValueError("Only red_pullback and green_reclaim are enabled")
    if data.get("data_level") not in {"L1", "L2"}:
        raise ValueError("data_level must be L1 or L2")
    flags = data.get("flags", [])
    if not isinstance(flags, list) or any(not isinstance(x, str) for x in flags):
        raise ValueError("flags must be an array of supported strings")
    unknown = set(flags) - DATA_FLAGS - VETO_FLAGS
    if unknown:
        raise ValueError(f"Unknown flags: {sorted(unknown)}")
    if set(flags) & DATA_FLAGS:
        return response("等待数据", sorted(set(flags) & DATA_FLAGS))
    asof = stamp(data.get("asof"), "asof")
    if set(flags) & VETO_FLAGS:
        return response("回避", sorted(set(flags) & VETO_FLAGS), asof=asof.isoformat())

    timestamps = obj(data.get("timestamps"), "timestamps")
    times = {name: stamp(timestamps.get(name), name) for name in (
        "daily_available_at", "membership_available_at", "snapshot_available_at",
        "support_frozen_at",
    )}
    for name, value in obj(timestamps.get("feature_available_at", {}),
                           "feature_available_at").items():
        times[f"feature:{name}"] = stamp(value, name)
    for name, value in times.items():
        if value > asof:
            raise DataGate("future_information", f"{name} was unavailable at asof")
    for name in ("daily_available_at", "membership_available_at"):
        if times[name].date() >= asof.date():
            raise DataGate("insufficient_data", f"{name} violates the prior-day frozen-pool rule")

    universe = obj(data.get("universe"), "universe")
    if not boolean(universe, "point_in_time_membership") or not str(
        universe.get("membership_source", "")
    ).strip():
        raise DataGate("membership_unverifiable", "No audited historical membership")
    counts = [number(universe, name, 1) for name in (
        "members", "valid_members", "ranking_themes",
    )]
    members, valid, themes = counts
    if any(x != int(x) for x in counts) or valid > members:
        raise DataGate("insufficient_data", "Invalid universe counts")
    if valid < 10 or valid / members < 0.9 or themes < 20:
        raise DataGate("insufficient_data", "Universe size/coverage below Q2 minimum")

    prior = obj(data.get("prior"), "prior")
    intraday = obj(data.get("intraday"), "intraday")
    atr = number(prior, "atr20", 0)
    prev_close = number(prior, "prev_close", 0)
    if atr <= 0 or prev_close <= 0:
        raise DataGate("insufficient_data", "ATR/previous close must be positive")
    liquidity = number(prior, "median_amount20", 0)
    T = (0.4 * number(prior, "theme_rs3_rank", 0, 100)
         + 25 * number(prior, "theme_breadth", 0, 1)
         + 20 * number(prior, "theme_above_ma10", 0, 1)
         + 15 * number(prior, "theme_persistence", 0, 1))
    rs5 = number(prior, "rs5_rank", 0, 100)
    rs1 = number(prior, "rs1_rank", 0, 100)
    attention = number(prior, "amount_share_rank", 0, 100)
    resilience = number(prior, "resilience_rank", 0, 100)
    I = 0.35 * rs5 + 0.15 * rs1 + 0.20 * attention + 0.30 * resilience
    theme_excess = number(intraday, "theme_excess_return")
    theme_breadth = number(intraday, "theme_breadth", 0, 1)
    market_breadth = number(intraday, "market_breadth", 0, 1)
    market_weak = boolean(intraday, "market_below_vwap_and_negative")
    relative = number(intraday, "relative_recovery_rank", 0, 100)

    raw_bars = data.get("bars")
    if not isinstance(raw_bars, list) or len(raw_bars) != 8:
        raise DataGate("insufficient_data", "Exactly 8 closed bars are required")
    bars = []
    ends = []
    minutes = []
    for index, raw in enumerate(raw_bars):
        bar = obj(raw, f"bars[{index}]")
        end = stamp(bar.get("end"), f"bars[{index}].end")
        if end > asof:
            raise DataGate("future_information", "Unclosed/future bar")
        if end.date() != asof.date():
            raise DataGate("insufficient_data", "Bars must be from the same decision day")
        b = {name: number(bar, name, 0) for name in (
            "open", "high", "low", "close", "volume", "seasonal_volume", "vwap",
        )}
        if min(b.values()) <= 0 or b["low"] > min(b["open"], b["close"]) or (
            b["high"] < max(b["open"], b["close"]) or b["low"] > b["high"]
        ):
            raise DataGate("insufficient_data", "Invalid OHLC/zero liquidity/seasonal template")
        bars.append(b)
        ends.append(end)
        minutes.append(trading_minute(end))
    if any(b - a != 5 for a, b in zip(minutes, minutes[1:])):
        raise DataGate("insufficient_data", "Missing/duplicate/noncontiguous bars")
    if asof - ends[-1] > timedelta(minutes=5):
        raise DataGate("stale_quote", "Decision snapshot older than one signal bar")
    if times["snapshot_available_at"] < ends[-1]:
        raise DataGate("future_information", "Snapshot claims availability before bar close")
    if times["support_frozen_at"] > ends[2]:
        raise DataGate("future_information", "Support chosen after pullback started")
    support = obj(data.get("support"), "support")
    S = number(support, "price", 0)
    if S <= 0 or not str(support.get("source", "")).strip():
        raise DataGate("insufficient_data", "Frozen support and source are required")
    if data.get("vwap_kind") not in {"actual", "proxy"}:
        raise DataGate("insufficient_data", "vwap_kind must be actual or proxy")

    U, D, R = bars[:3], bars[3:6], bars[6:]
    normalized = [b["volume"] / b["seasonal_volume"] for b in bars]
    x = mean(normalized[3:6]) / mean(normalized[:3])
    y = mean(normalized[6:]) / mean(normalized[3:6])
    stable_count = sum(b["low"] >= a["low"] - 0.05 * atr for a, b in zip(D, D[1:]))
    span = R[-1]["high"] - R[-1]["low"]
    if span <= 0:
        raise DataGate("insufficient_data", "Flat last bar cannot confirm recovery")
    clv = (R[-1]["close"] - R[-1]["low"]) / span
    sub = {
        "pullback_shrink": 100 * clip01((1.2 - x) / 0.6),
        "low_stability": 50 * stable_count,
        "close_recovery": 100 * clv,
        "rebound_volume": 100 * clip01((y - 0.7) / 0.5),
        "relative_recovery": relative,
    }
    A = sum(w * sub[name] for name, w in (
        ("pullback_shrink", .25), ("low_stability", .25), ("close_recovery", .20),
        ("rebound_volume", .15), ("relative_recovery", .15),
    ))
    roles = []
    if I >= 75 and rs5 >= 80 and resilience >= 60:
        roles.append("动量核心候选")
    if I >= 75 and attention >= 80 and rs5 >= 60:
        roles.append("容量核心候选")
    roles = roles or (["前排"] if I >= 70 else ["普通/跟随"])
    stop = min(b["low"] for b in D) - .10 * atr
    close = R[-1]["close"]
    metadata = {
        "asof": asof.isoformat(), "setup": setup, "data_level": "L1",
        "declared_data_level": data["data_level"],
        "scores": {"T": round(T, 4), "I": round(I, 4), "A": round(A, 4)},
        "A_components": {k: round(v, 4) for k, v in sub.items()},
        "identity_components": {"rs5_rank": rs5, "rs1_rank": rs1,
                                "amount_share_rank": attention, "resilience_rank": resilience},
        "roles": roles, "stop_price_unrounded": round(stop, 8),
        "ratios": {"pullback_volume": round(x, 6), "rebound_volume": round(y, 6)},
        "vwap_kind": data["vwap_kind"],
    }
    support_break = min(b["low"] for b in D) < S - .20 * atr or any(
        a["close"] < S - .10 * atr and b["close"] < S - .10 * atr
        for a, b in zip(D, D[1:])
    )
    if support_break or close <= stop:
        return response("回避", ["support_broken"], **metadata)
    if market_weak and market_breadth < .35:
        return response("回避", ["market_crisis"], **metadata)

    risk = obj(data.get("risk"), "risk")
    net_rr = number(risk, "net_rr", 0)
    if not boolean(risk, "overnight_risk_checked") or not boolean(
        risk, "events_permissions_checked"
    ):
        raise DataGate("insufficient_data", "Risk/events/permissions review unfinished")
    reasons = []
    checks = {
        "theme_score_below_65": T >= 65,
        "theme_intraday_weak": theme_excess >= 0 and theme_breadth >= .55,
        "illiquid": liquidity >= 100_000_000,
        "outside_entry_window": 610 <= (ends[-1].hour * 60 + ends[-1].minute) <= 860,
        "no_actual_pullback": D[-1]["close"] < U[-1]["close"],
        "recovery_not_confirmed": (R[-1]["close"] > R[0]["close"]
                                   and R[-1]["close"] >= R[0]["high"]
                                   and R[-1]["low"] >= min(b["low"] for b in D)),
        "pullback_not_shrinking": x <= .85,
        "rebound_volume_weak": y >= 1.10,
        "last_close_weak": clv >= .65,
        "relative_recovery_weak": relative >= 60,
        "risk_reward_below_2": net_rr >= 2,
        "stop_distance_invalid": 0.03 < close - stop <= .80 * atr,
    }
    if setup == "red_pullback":
        checks.update({
            "identity_below_70": I >= 70, "absorption_below_70": A >= 70,
            "low_stability_insufficient": stable_count >= 1,
            "initial_rally_missing": U[-1]["close"] > U[0]["open"],
            "not_low_red_zone": 0 <= close - prev_close <= .75 * atr,
            "vwap_not_reclaimed": all(b["close"] >= b["vwap"] for b in R),
            "overextended_from_vwap": close - R[-1]["vwap"] <= .35 * atr,
        })
    else:
        checks.update({
            "identity_below_80": I >= 80, "resilience_below_70": resilience >= 70,
            "absorption_below_80": A >= 80, "low_stability_insufficient": stable_count == 2,
            "not_shallow_green_zone": -.75 * atr <= close - prev_close < 0,
        })
    reasons.extend(name for name, passed in checks.items() if not passed)
    metadata["net_rr_input"] = net_rr
    if reasons:
        return response("观察/不入场", reasons, **metadata)
    return response("候选通过，核验下一bar成交与仓位", [], **metadata,
                    note="非订单、非胜率；净RR/排名/来源需外部审计，实际成交后重新核验风险与T+1")


def score_snapshot(data: Any) -> dict:
    try:
        return _score(obj(data, "input"))
    except DataGate as exc:
        return response("等待数据", [exc.reason], detail=str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Audited JSON snapshot; omit for stdin")
    args = parser.parse_args()
    try:
        raw = Path(args.input).read_text(encoding="utf-8-sig") if args.input else sys.stdin.read()
        result = score_snapshot(json.loads(raw))
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
        sys.stdout.write("\n")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        json.dump({"error": str(exc)}, sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
