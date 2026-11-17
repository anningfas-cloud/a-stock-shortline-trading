#!/usr/bin/env python3
"""Synthetic invariant tests ONLY; not historical performance validation."""
import copy
import unittest
from datetime import datetime, timedelta

from score_quant_snapshot import score_snapshot
from evaluate_setup import evaluate_entry


def fixture():
    prices = [
        (10.00, 10.12, 9.99, 10.10, 100),
        (10.10, 10.19, 10.08, 10.18, 100),
        (10.18, 10.24, 10.16, 10.22, 100),
        (10.22, 10.23, 10.10, 10.14, 60),
        (10.14, 10.17, 10.11, 10.15, 60),
        (10.15, 10.18, 10.12, 10.16, 60),
        (10.16, 10.24, 10.15, 10.21, 90),
        (10.21, 10.26, 10.20, 10.25, 90),
    ]
    first = datetime.fromisoformat("2026-09-10T09:35:00+08:00")
    bars = [dict(end=(first + timedelta(minutes=5 * i)).isoformat(),
                 open=o, high=h, low=l, close=c, volume=v * 1000,
                 seasonal_volume=100_000, vwap=10.10)
            for i, (o, h, l, c, v) in enumerate(prices)]
    return {
        "version": "Q2.0-research", "setup": "red_pullback", "data_level": "L1",
        "asof": bars[-1]["end"],
        "timestamps": {
            "daily_available_at": "2026-09-09T16:00:00+08:00",
            "membership_available_at": "2026-09-09T18:00:00+08:00",
            "snapshot_available_at": bars[-1]["end"],
            "support_frozen_at": bars[2]["end"],
        },
        "universe": {"members": 20, "valid_members": 20, "ranking_themes": 20,
                     "point_in_time_membership": True, "membership_source": "synthetic-only"},
        "prior": {"theme_rs3_rank": 85, "theme_breadth": .7,
                  "theme_above_ma10": .8, "theme_persistence": 2 / 3,
                  "rs5_rank": 90, "rs1_rank": 90, "amount_share_rank": 85,
                  "resilience_rank": 90, "atr20": .5, "prev_close": 10,
                  "median_amount20": 200_000_000},
        "intraday": {"theme_excess_return": .01, "theme_breadth": .7,
                     "market_breadth": .6, "market_below_vwap_and_negative": False,
                     "relative_recovery_rank": 90},
        "bars": bars, "support": {"price": 10.12, "source": "synthetic-frozen"},
        "vwap_kind": "actual",
        "risk": {"net_rr": 2.4, "overnight_risk_checked": True,
                 "events_permissions_checked": True},
    }


class QuantTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def test_known_formula_and_candidate(self):
        r = score_snapshot(self.data)
        self.assertTrue(r["decision"].startswith("候选通过"))
        self.assertAlmostEqual(r["scores"]["T"], 77.5)
        self.assertAlmostEqual(r["scores"]["I"], 89.0)
        self.assertAlmostEqual(r["scores"]["A"], 95.1667, places=4)
        self.assertAlmostEqual(r["stop_price_unrounded"], 10.05)
        self.assertNotIn("win_rate", r)

    def test_green_has_stricter_role_thresholds(self):
        self.data["setup"] = "green_reclaim"
        self.data["prior"]["prev_close"] = 10.35
        self.assertTrue(score_snapshot(self.data)["decision"].startswith("候选通过"))
        self.data["prior"]["resilience_rank"] = 65
        self.assertIn("resilience_below_70", score_snapshot(self.data)["reasons"])

    def test_missing_is_not_neutral(self):
        del self.data["prior"]["rs5_rank"]
        self.assertEqual(score_snapshot(self.data)["decision"], "等待数据")

    def test_bool_nan_and_out_of_range_are_rejected(self):
        for value in (True, float("nan"), float("inf"), 101, -1):
            d = copy.deepcopy(self.data)
            d["prior"]["rs5_rank"] = value
            self.assertEqual(score_snapshot(d)["decision"], "等待数据")

    def test_data_gate_precedes_high_scores(self):
        self.data["flags"] = ["stale_quote"]
        self.assertEqual(score_snapshot(self.data)["decision"], "等待数据")

    def test_hard_veto_without_score_invention(self):
        self.data["flags"] = ["liquidity_trap"]
        del self.data["prior"]
        self.assertEqual(score_snapshot(self.data)["decision"], "回避")
        self.assertEqual(evaluate_entry({"mode": "entry", "flags": ["support_broken"]})
                         ["decision"], "回避")

    def test_future_features(self):
        self.data["timestamps"]["feature_available_at"] = {
            "netflow": "2026-09-10T15:00:00+08:00"}
        self.assertIn("future_information", score_snapshot(self.data)["reasons"])

    def test_unclosed_future_bar(self):
        self.data["bars"][-1]["end"] = "2026-09-10T10:15:00+08:00"
        self.assertIn("future_information", score_snapshot(self.data)["reasons"])

    def test_support_cannot_be_backdrawn(self):
        self.data["timestamps"]["support_frozen_at"] = self.data["bars"][4]["end"]
        self.assertIn("future_information", score_snapshot(self.data)["reasons"])

    def test_today_membership_is_not_historical_membership(self):
        self.data["timestamps"]["membership_available_at"] = "2026-09-10T09:00:00+08:00"
        self.assertEqual(score_snapshot(self.data)["decision"], "等待数据")

    def test_universe_coverage(self):
        self.data["universe"]["valid_members"] = 17
        self.assertEqual(score_snapshot(self.data)["decision"], "等待数据")

    def test_membership_not_audited(self):
        self.data["universe"]["point_in_time_membership"] = False
        self.assertIn("membership_unverifiable", score_snapshot(self.data)["reasons"])

    def test_gap_or_duplicate_in_bars(self):
        self.data["bars"][4]["end"] = self.data["bars"][3]["end"]
        self.assertEqual(score_snapshot(self.data)["decision"], "等待数据")

    def test_zero_volume_or_template(self):
        for key in ("volume", "seasonal_volume"):
            d = copy.deepcopy(self.data)
            d["bars"][4][key] = 0
            self.assertEqual(score_snapshot(d)["decision"], "等待数据")

    def test_flat_recovery_is_not_confirmation(self):
        b = self.data["bars"][-1]
        b.update(open=10.25, high=10.25, low=10.25, close=10.25)
        self.assertEqual(score_snapshot(self.data)["decision"], "等待数据")

    def test_one_strong_factor_cannot_offset_other_weakness(self):
        self.data["prior"].update(theme_rs3_rank=0, theme_breadth=.2,
                                  theme_above_ma10=.2, theme_persistence=0)
        self.assertIn("theme_score_below_65", score_snapshot(self.data)["reasons"])

    def test_support_break_beats_scores(self):
        self.data["support"]["price"] = 10.5
        self.assertEqual(score_snapshot(self.data)["decision"], "回避")

    def test_net_rr_and_risk_review(self):
        self.data["risk"]["net_rr"] = 1.9
        self.assertIn("risk_reward_below_2", score_snapshot(self.data)["reasons"])
        self.data["risk"]["overnight_risk_checked"] = False
        self.assertEqual(score_snapshot(self.data)["decision"], "等待数据")

    def test_weak_market_veto(self):
        self.data["intraday"].update(market_breadth=.2, market_below_vwap_and_negative=True)
        self.assertEqual(score_snapshot(self.data)["decision"], "回避")

    def test_input_not_mutated_and_deterministic(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(score_snapshot(self.data), score_snapshot(self.data))
        self.assertEqual(self.data, before)

    def test_declared_l2_is_not_verified_l2(self):
        self.data["data_level"] = "L2"
        r = score_snapshot(self.data)
        self.assertEqual(r["data_level"], "L1")
        self.assertEqual(r["declared_data_level"], "L2")

    def test_missing_timezone(self):
        self.data["asof"] = "2026-09-10T10:10:00"
        self.assertIn("timestamp_missing", score_snapshot(self.data)["reasons"])

    def test_stale_intraday_snapshot(self):
        self.data["asof"] = "2026-09-10T10:20:00+08:00"
        self.assertIn("stale_quote", score_snapshot(self.data)["reasons"])

    def test_lunch_is_not_a_missing_bar(self):
        stamps = ["11:10", "11:15", "11:20", "11:25", "11:30", "13:05", "13:10", "13:15"]
        for bar, minute in zip(self.data["bars"], stamps):
            bar["end"] = f"2026-09-10T{minute}:00+08:00"
        self.data["asof"] = self.data["bars"][-1]["end"]
        self.data["timestamps"]["snapshot_available_at"] = self.data["asof"]
        self.data["timestamps"]["support_frozen_at"] = self.data["bars"][2]["end"]
        self.assertTrue(score_snapshot(self.data)["decision"].startswith("候选通过"))

    def test_unknown_flags_and_disabled_setup_fail_closed(self):
        for key, value in (("flags", ["invented_signal"]), ("setup", "deep_water")):
            d = copy.deepcopy(self.data)
            d[key] = value
            with self.assertRaises(ValueError):
                score_snapshot(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
