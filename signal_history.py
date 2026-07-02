"""
signal_history.py
─────────────────
Two additive layers on top of the untouched vpci_engine:

1. TRUE FRESH-SIGNAL DETECTION (position state machine)
   The engine's `fresh_signal` only checks "entry now, no entry on the
   previous bar". That mislabels re-entries: a stock that entered weeks ago,
   slipped to 6/7 for a bar, then re-passed 7/7 looks "fresh" even though the
   position never exited. Here we replay the FULL weekly history as a
   position state machine:

       • ENTRY  — full 7/7 entry fires (same gates + extension filter as the
                  engine, via vpci_engine._check_gates_at — no logic re-invented)
       • HOLD   — trailing stop ratchets up: trail = max(trail, close − 3×ATR)
       • SELL   — weekly close < trailing stop (the validated v3 exit)

   A signal on the latest bar is FRESH (fresh_v2) only if NO position was
   open coming into that bar — i.e. the stock has no prior buy history, or
   its last position was closed by a sell before this entry.

2. YOUNG-LISTING VPCI SCAN
   Stocks with fewer than MIN_BARS (42) weekly bars can't run the full
   7-gate analysis, but VPCI (5/20 windows) is computable from ~22 bars.
   `analyze_young_stock` produces a light accumulation read-out so promising
   new listings surface in their own tab instead of vanishing into "failed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vpci_engine import _check_gates_at, DEFAULT_PARAMS, MIN_BARS

YOUNG_MIN_BARS = 22   # VPCI long window (20) + rising check + signal line


# ═══════════════════════════════════════════════════════════════════════
# Shared indicator computation — IDENTICAL formulas & params to the engine
# (copied verbatim so the engine file itself stays untouched)
# ═══════════════════════════════════════════════════════════════════════
def _build_series(df: pd.DataFrame, params: dict) -> dict:
    close_pr = df["Close"]
    high_pr  = df["High"]
    low_pr   = df["Low"]
    vol      = df["Volume"]

    ef  = close_pr.ewm(span=params["ema_fast"], adjust=False).mean()
    es  = close_pr.ewm(span=params["ema_slow"], adjust=False).mean()
    s40 = close_pr.rolling(window=params["sma_long"]).mean()

    tr = pd.concat([high_pr - low_pr,
                    abs(high_pr - close_pr.shift(1)),
                    abs(low_pr - close_pr.shift(1))], axis=1).max(axis=1)
    atr_s = tr.rolling(window=params["atr_len"]).mean()

    vpciS, vpciL = params["vpci_short"], params["vpci_long"]
    cv = close_pr * vol
    vwma_L  = cv.rolling(vpciL).sum() / vol.rolling(vpciL).sum()
    vwma_S  = cv.rolling(vpciS).sum() / vol.rolling(vpciS).sum()
    sma_C_L = close_pr.rolling(vpciL).mean()
    sma_C_S = close_pr.rolling(vpciS).mean()
    sma_V_L = vol.rolling(vpciL).mean()
    sma_V_S = vol.rolling(vpciS).mean()
    vpc = vwma_L - sma_C_L
    vpr = vwma_S / sma_C_S
    vm  = sma_V_S / sma_V_L
    vpci_val = vpc * vpr * vm
    vpci_vol = vpci_val * vol
    vpci_sig = vpci_vol.rolling(vpciS).sum() / vol.rolling(vpciS).sum()

    hi_brk     = high_pr.rolling(params["brk_weeks"]).max().shift(1)
    ret_rs     = close_pr.pct_change(periods=params["rs_weeks"])
    candle_rng = high_pr - low_pr

    return {
        "close": close_pr, "high": high_pr, "low": low_pr, "vol": vol,
        "ef": ef, "es": es, "s40": s40, "atr": atr_s,
        "vpci": vpci_val, "vpci_sig": vpci_sig,
        "hi_brk": hi_brk, "ret_rs": ret_rs, "candle_rng": candle_rng,
        "sma_V_L": sma_V_L,
    }


# ═══════════════════════════════════════════════════════════════════════
# 1. POSITION STATE MACHINE — true fresh detection
# ═══════════════════════════════════════════════════════════════════════
def classify_fresh_v2(df: pd.DataFrame, params: dict = DEFAULT_PARAMS) -> dict:
    """
    Replay full history. Returns:
      fresh_v2          — 7/7 entry on the LAST bar AND no open position before it
      fresh_ext_v2      — 7/7-but-extended on the LAST bar AND no open position
      in_open_position  — a prior entry is still live (trail never hit)
      position_state    — human label for the UI
      last_entry_date / last_exit_date — for audit
    """
    out = {"fresh_v2": False, "fresh_ext_v2": False, "in_open_position": False,
           "position_state": "No signal history",
           "last_entry_date": None, "last_exit_date": None}
    try:
        n = len(df)
        if n < MIN_BARS:
            return out
        s = _build_series(df, params)
        ext_f      = params["ext_filter"]
        trail_mult = params["trail_mult"]
        dates = df.index

        # earliest bar where all rolling windows + slope + prev-bar exist
        warmup = max(params["sma_long"], params["vpci_long"],
                     params["atr_len"], params["brk_weeks"] + 1,
                     params["rs_weeks"], params["slope_weeks"]) + 1

        in_pos = False
        trail = None
        last_entry_date = None
        last_exit_date = None
        pos_open_before_last = False   # state coming INTO the last bar

        for i in range(warmup, n):
            neg = i - n   # negative index so _check_gates_at behaves exactly as in the engine
            if i == n - 1:
                pos_open_before_last = in_pos

            c   = float(s["close"].iloc[i])
            atr = float(s["atr"].iloc[i]) if pd.notna(s["atr"].iloc[i]) else c * 0.05

            # ── exit check first (a sell frees the stock to be fresh again) ──
            if in_pos:
                trail = max(trail, c - trail_mult * atr)
                if c < trail:
                    in_pos = False
                    trail = None
                    last_exit_date = dates[i]
                    continue   # exited this bar; a same-bar re-entry is not possible

            # ── entry check — same conditions as engine full_entry ──
            if not in_pos:
                try:
                    gates, gc = _check_gates_at(
                        s["close"], s["high"], s["ef"], s["es"], s["s40"],
                        s["vpci"], s["vpci_sig"], s["ret_rs"], s["hi_brk"],
                        params, idx=neg)
                except Exception:
                    continue
                candle = float(s["candle_rng"].iloc[i]) if pd.notna(s["candle_rng"].iloc[i]) else 0
                not_ext = True if ext_f == 0 else candle <= ext_f * atr
                if all(gates) and not_ext:
                    in_pos = True
                    trail = c - trail_mult * atr
                    last_entry_date = dates[i]

        # ── classify the LAST bar ──
        gates, gc = _check_gates_at(
            s["close"], s["high"], s["ef"], s["es"], s["s40"],
            s["vpci"], s["vpci_sig"], s["ret_rs"], s["hi_brk"], params, idx=-1)
        c_last   = float(s["close"].iloc[-1])
        atr_last = float(s["atr"].iloc[-1]) if pd.notna(s["atr"].iloc[-1]) else c_last * 0.05
        candle_l = float(s["candle_rng"].iloc[-1]) if pd.notna(s["candle_rng"].iloc[-1]) else 0
        not_ext_l = True if ext_f == 0 else candle_l <= ext_f * atr_last
        full_now  = all(gates) and not_ext_l
        ext_now   = all(gates) and not not_ext_l and ext_f > 0

        out["fresh_v2"]     = bool(full_now and not pos_open_before_last)
        out["fresh_ext_v2"] = bool(ext_now  and not pos_open_before_last)
        out["in_open_position"] = bool(pos_open_before_last)
        out["last_entry_date"] = (last_entry_date.date().isoformat()
                                  if last_entry_date is not None else None)
        out["last_exit_date"]  = (last_exit_date.date().isoformat()
                                  if last_exit_date is not None else None)

        if out["fresh_v2"] or out["fresh_ext_v2"]:
            out["position_state"] = ("🔥 First-ever signal" if last_exit_date is None
                                     else f"🔥 Fresh after sell ({out['last_exit_date']})")
        elif pos_open_before_last:
            out["position_state"] = (f"📌 Already in open position "
                                     f"(entered {out['last_entry_date']})")
        elif last_exit_date is not None:
            out["position_state"] = f"💤 Flat (last exit {out['last_exit_date']})"
        return out
    except Exception:
        return out


# ═══════════════════════════════════════════════════════════════════════
# 2. YOUNG-LISTING VPCI ACCUMULATION SCAN  (< MIN_BARS of history)
# ═══════════════════════════════════════════════════════════════════════
def analyze_young_stock(symbol: str, df: pd.DataFrame,
                        params: dict = DEFAULT_PARAMS) -> dict | None:
    """
    Light analysis for recently listed stocks (YOUNG_MIN_BARS ≤ bars < MIN_BARS).
    Same VPCI formulas/windows as the engine — just fewer gates, since the
    40-week SMA / 52w-high context doesn't exist yet.
    """
    try:
        n = len(df)
        if n < YOUNG_MIN_BARS or n >= MIN_BARS:
            return None
        s = _build_series(df, params)

        c      = float(s["close"].iloc[-1])
        l_vpci = float(s["vpci"].iloc[-1]) if pd.notna(s["vpci"].iloc[-1]) else 0
        p_vpci = float(s["vpci"].iloc[-2]) if pd.notna(s["vpci"].iloc[-2]) else 0
        l_sig  = float(s["vpci_sig"].iloc[-1]) if pd.notna(s["vpci_sig"].iloc[-1]) else 0
        l_ef   = float(s["ef"].iloc[-1])

        # G5-equivalent accumulation check (identical condition to the engine)
        accumulating = bool(l_vpci > 0 and l_vpci > l_sig and l_vpci > p_vpci)

        avg_vol   = float(s["sma_V_L"].iloc[-1]) if pd.notna(s["sma_V_L"].iloc[-1]) else 1
        vol_ratio = float(s["vol"].iloc[-1]) / avg_vol if avg_vol > 0 else 0

        hi_all  = float(s["high"].max())
        pct_off = (hi_all - c) / hi_all * 100 if hi_all > 0 else 0
        ret_since_listing = (c / float(s["close"].iloc[0]) - 1) * 100

        # simple 5-week momentum (rs_weeks=13 is impossible on short history)
        mom5 = (c / float(s["close"].iloc[-6]) - 1) * 100 if n >= 6 else np.nan

        return {
            "symbol": symbol, "bars": n, "close": round(c, 2),
            "vpci": round(l_vpci, 4), "vpci_signal": round(l_sig, 4),
            "vpci_rising": bool(l_vpci > p_vpci),
            "accumulating": accumulating,
            "above_13ema": bool(c > l_ef),
            "vol_ratio": round(vol_ratio, 2),
            "mom_5w_%": round(mom5, 1) if pd.notna(mom5) else None,
            "pct_off_high": round(pct_off, 1),
            "ret_since_listing_%": round(ret_since_listing, 1),
        }
    except Exception:
        return None
