import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("paper-bot")


# ==============================
# CONFIG  (unchanged from original)
# ==============================

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT",
    "AVAXUSDT", "MATICUSDT", "ATOMUSDT", "LTCUSDT", "DOGEUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT", "FILUSDT",
]

FEE_RATE      = 0.0004
SLIPPAGE      = 0.0004
BASE_RISK     = 0.01
MAX_RISK      = 0.02
PORT_CAP      = 0.2
MAX_POSITIONS = 4
ATR_PERIOD    = 14
EMA_PERIOD    = 50
STOP_ATR      = 2.0
ADX_THRESHOLD = 25
ATR_EXPANSION = 1.3
MULTIPLIER    = 1.015
INITIAL_CAPITAL = 100.0
INTERVAL      = "1h"
CANDLE_LIMIT  = 300
MIN_LOOKBACK  = 120

CC_SYMBOL_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BNBUSDT": "BNB",
    "LINKUSDT": "LINK",
    "AVAXUSDT": "AVAX",
    "MATICUSDT": "MATIC",
    "ATOMUSDT": "ATOM",
    "LTCUSDT": "LTC",
    "DOGEUSDT": "DOGE",
    "APTUSDT": "APT",
    "ARBUSDT": "ARB",
    "OPUSDT": "OP",
    "NEARUSDT": "NEAR",
    "FILUSDT": "FIL",
}
CC_HISTO_URL  = "https://min-api.cryptocompare.com/data/v2/histohour"
CC_PRICE_URL  = "https://min-api.cryptocompare.com/data/price"

STATE_PATH = Path(os.getenv("STATE_FILE", "state.json"))


# ==============================
# STATE  (unchanged)
# ==============================

def default_state() -> Dict[str, Any]:
    return {
        "equity": INITIAL_CAPITAL,
        "peak_equity": INITIAL_CAPITAL,
        "max_drawdown": 0.0,
        "trade_count": 0,
        "risk_multiplier": 1.0,
        "equity_curve": [],
        "positions": {},
        "last_attempted_candle": None,
        "last_processed_candle": None,
    }


def load_state() -> Dict[str, Any]:
    state = default_state()
    if not STATE_PATH.exists():
        logger.info("State file not found, starting fresh.")
        return state
    try:
        loaded = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse {STATE_PATH}: {exc}") from exc
    state.update({k: v for k, v in loaded.items() if k != "positions"})
    state["positions"] = loaded.get("positions", {})
    state["equity_curve"] = list(loaded.get("equity_curve", []))
    return state


def save_state(state: Dict[str, Any]) -> None:
    state["equity_curve"] = [float(x) for x in state.get("equity_curve", [])][-500:]
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


# ==============================
# HELPERS  (unchanged)
# ==============================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.utcnow()


def position_open_risk(state: Dict[str, Any]) -> float:
    return sum(float(p.get("risk", 0.0)) for p in state["positions"].values())


def append_equity_snapshot(state: Dict[str, Any]) -> float:
    equity = float(state["equity"])
    curve = state.setdefault("equity_curve", [])
    curve.append(equity)
    return float(np.mean(curve[-50:])) if len(curve) >= 50 else equity


def update_drawdown_state(state: Dict[str, Any]) -> Tuple[float, float]:
    equity = float(state["equity"])
    peak_equity = max(float(state["peak_equity"]), equity)
    drawdown = (equity - peak_equity) / peak_equity if peak_equity else 0.0
    if drawdown < -0.20:
        risk_multiplier = 0.5
    elif drawdown < -0.10:
        risk_multiplier = 0.75
    else:
        risk_multiplier = 1.0
    state["peak_equity"] = peak_equity
    state["risk_multiplier"] = risk_multiplier
    state["max_drawdown"] = min(float(state.get("max_drawdown", 0.0)), drawdown)
    return drawdown, risk_multiplier


# ==============================
# PAPER TRADING — live price fetch (no auth needed)
# ==============================

def get_live_price(symbol: str, fallback: float) -> float:
    """Fetch latest price via CryptoCompare public API."""
    try:
        coin = CC_SYMBOL_MAP.get(symbol, symbol.replace("USDT",""))
        r = requests.get(
            CC_PRICE_URL,
            params={"fsym": coin, "tsyms": "USDT"},
            headers={"User-Agent": "python-requests/2.31.0"},
            timeout=10,
        )
        data = r.json()
        if "USDT" in data:
            return float(data["USDT"])
    except Exception:
        pass
    return fallback


# ==============================
# DATA  (unchanged)
# ==============================

def fetch_asset_klines(symbol: str) -> pd.DataFrame:
    import time
    logger.info("Fetching 1h candles for %s via CryptoCompare", symbol)
    coin = CC_SYMBOL_MAP.get(symbol, symbol.replace("USDT", ""))
    rows = []
    for attempt in range(5):
        try:
            r = requests.get(
                CC_HISTO_URL,
                params={
                    "fsym": coin,
                    "tsym": "USDT",
                    "limit": CANDLE_LIMIT,
                    "aggregate": 1,
                },
                headers={"User-Agent": "python-requests/2.31.0"},
                timeout=30,
            )
            data = r.json()
            if data.get("Response") == "Success":
                candles = data["Data"]["Data"]
                if len(candles) >= MIN_LOOKBACK:
                    rows = candles
                    break
                else:
                    logger.warning("Attempt %d for %s: only %d candles", attempt+1, symbol, len(candles))
            else:
                logger.warning("Attempt %d for %s: %s", attempt+1, symbol, str(data)[:120])
            time.sleep(2)
        except Exception as e:
            logger.warning("Attempt %d for %s failed: %s", attempt+1, symbol, e)
            time.sleep(2)
    if not rows:
        raise RuntimeError(f"{symbol} could not be fetched from CryptoCompare.")

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "time": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volumefrom": "volume",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["close_time"] = df["timestamp"] + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    logger.info("Fetched %d candles for %s", len(df), symbol)
    return df[["timestamp", "open", "high", "low", "close", "volume", "close_time"]]


def prepare_asset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy().sort_values("timestamp")
    prepared["prev_close"] = prepared["close"].shift(1)
    prepared["tr"] = np.maximum(
        prepared["high"] - prepared["low"],
        np.maximum(
            (prepared["high"] - prepared["prev_close"]).abs(),
            (prepared["low"] - prepared["prev_close"]).abs(),
        ),
    )
    prepared["atr"] = prepared["tr"].ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    prepared["atr_median"] = prepared["atr"].rolling(100).median()
    prepared["ema_1h"] = prepared["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    df_4h = prepared.set_index("timestamp").resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    df_4h["ema_4h"] = df_4h["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    prepared = prepared.set_index("timestamp")
    prepared = prepared.merge(df_4h[["ema_4h"]], left_index=True, right_index=True, how="left")
    prepared["ema_4h"] = prepared["ema_4h"].ffill()

    plus_dm = prepared["high"].diff().copy()
    minus_dm = (-prepared["low"].diff()).copy()
    plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0
    tr_smooth = prepared["tr"].ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / tr_smooth)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    prepared["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return prepared.reset_index()


def build_market_data() -> Tuple[Dict[str, pd.DataFrame], pd.Timestamp]:
    import time
    raw_data: Dict[str, pd.DataFrame] = {}
    prepared_data: Dict[str, pd.DataFrame] = {}

    for symbol in ASSETS:
        raw_data[symbol] = fetch_asset_klines(symbol)
        prepared_data[symbol] = prepare_asset(raw_data[symbol])
        time.sleep(0.5)

    master = prepared_data["BTCUSDT"]["timestamp"]
    for symbol in ASSETS:
        frame = prepared_data[symbol].set_index("timestamp").reindex(master).ffill()
        prepared_data[symbol] = frame.reset_index()

    btc_raw = raw_data["BTCUSDT"].reset_index(drop=True)
    closed_mask = btc_raw["close_time"] <= utc_now()
    if not closed_mask.any():
        raise RuntimeError("No closed BTCUSDT candle available yet.")

    latest_closed_index = closed_mask[closed_mask].index[-1]
    latest_closed_timestamp = raw_data["BTCUSDT"].iloc[latest_closed_index]["timestamp"]

    if latest_closed_index < MIN_LOOKBACK:
        raise RuntimeError(f"Latest closed candle index {latest_closed_index} below required {MIN_LOOKBACK}.")

    return prepared_data, latest_closed_timestamp


# ==============================
# SIGNALS  (unchanged)
# ==============================

def row_is_usable(row: pd.Series) -> bool:
    required = ["close", "high", "low", "atr", "atr_median", "ema_1h", "ema_4h", "adx"]
    return (
        all(pd.notna(row[c]) for c in required)
        and float(row["atr"]) > 0
        and float(row["atr_median"]) > 0
    )


def compute_dynamic_risk(row: pd.Series, equity: float, equity_ma: float, risk_multiplier: float) -> Optional[float]:
    if not row_is_usable(row):
        return None
    regime_strength = clamp((float(row["adx"]) - 20.0) / 20.0, 0.0, 1.0)
    dynamic_risk = BASE_RISK + regime_strength * (MAX_RISK - BASE_RISK)
    dynamic_risk = min(dynamic_risk, MAX_RISK)
    vol_ratio = float(row["atr"]) / float(row["atr_median"])
    if vol_ratio <= 0:
        return None
    dynamic_risk *= clamp(1.0 / vol_ratio, 0.75, 1.25)
    if equity < equity_ma:
        dynamic_risk *= 0.5
    dynamic_risk *= risk_multiplier
    return dynamic_risk


def entry_signal(df: pd.DataFrame, index: int) -> bool:
    row = df.iloc[index]
    asset_up = (
        float(row["close"]) > float(row["ema_4h"])
        and float(row["ema_4h"]) > float(df["ema_4h"].iloc[index - 15])
    )
    if not asset_up:
        return False
    return (
        float(row["close"]) > float(row["ema_4h"]) * MULTIPLIER
        and float(row["ema_1h"]) > float(df["ema_1h"].iloc[index - 25])
        and float(row["adx"]) > ADX_THRESHOLD
        and float(row["atr"]) > ATR_EXPANSION * float(row["atr_median"])
        and (float(row["close"]) - float(df["close"].iloc[index - 10])) / float(df["close"].iloc[index - 10]) > 0.02
    )


# ==============================
# PAPER EXITS
# ==============================

def process_exits(state: Dict[str, Any], data: Dict[str, pd.DataFrame], index: int) -> None:
    for symbol in list(state["positions"].keys()):
        df = data[symbol]
        row = df.iloc[index]
        position = state["positions"][symbol]

        if not row_is_usable(row):
            continue

        initial_risk = float(position["entry"]) - float(position["stop"])
        if initial_risk <= 0:
            continue

        position["extreme"] = max(float(position["extreme"]), float(row["high"]))
        current_r = (float(row["close"]) - float(position["entry"])) / initial_risk
        trailing = float(position["stop"])

        if current_r >= 1.5:
            trailing = max(float(position["stop"]), float(position["extreme"]) - 2.5 * float(row["atr"]))

        if float(row["low"]) > trailing:
            continue

        exit_price = get_live_price(symbol, float(row["close"])) * (1 - SLIPPAGE)
        avg_entry = float(position.get("avg_entry", position["entry"]))
        qty = float(position["size"])
        pnl = (exit_price - avg_entry) * qty
        fee = (avg_entry * qty + exit_price * qty) * FEE_RATE
        state["equity"] = float(state["equity"]) + pnl - fee

        logger.info(
            "[PAPER EXIT] %s | qty=%.6f avg_entry=%.4f exit=%.4f pnl=%.4f fee=%.4f equity=%.4f",
            symbol, qty, avg_entry, exit_price, pnl, fee, state["equity"],
        )
        del state["positions"][symbol]
        save_state(state)


# ==============================
# PAPER ENTRIES
# ==============================

def process_entries(state: Dict[str, Any], data: Dict[str, pd.DataFrame], index: int, equity_ma: float) -> None:
    if len(state["positions"]) >= MAX_POSITIONS:
        return

    total_open_risk = position_open_risk(state)
    equity = float(state["equity"])

    if total_open_risk >= equity * PORT_CAP:
        return

    for symbol in ASSETS:
        if len(state["positions"]) >= MAX_POSITIONS:
            break
        if symbol in state["positions"]:
            continue

        df = data[symbol]
        row = df.iloc[index]

        if not row_is_usable(row) or index - 25 < 0:
            continue
        if not entry_signal(df, index):
            continue

        dynamic_risk = compute_dynamic_risk(row, equity, equity_ma, float(state["risk_multiplier"]))
        if dynamic_risk is None:
            continue

        trade_risk = equity * dynamic_risk
        if total_open_risk + trade_risk > equity * PORT_CAP:
            continue

        stop_distance = STOP_ATR * float(row["atr"])
        if stop_distance <= 0:
            continue

        entry_price = get_live_price(symbol, float(row["close"])) * (1 + SLIPPAGE)
        stop_price = entry_price - stop_distance
        size = trade_risk / stop_distance

        state["positions"][symbol] = {
            "entry": entry_price,
            "avg_entry": entry_price,
            "stop": stop_price,
            "size": size,
            "risk": trade_risk,
            "extreme": float(row["high"]),
            "adds": 0,
        }
        state["trade_count"] = int(state["trade_count"]) + 1
        total_open_risk += trade_risk

        logger.info(
            "[PAPER ENTRY] %s | size=%.6f entry=%.4f stop=%.4f risk=%.4f",
            symbol, size, entry_price, stop_price, trade_risk,
        )
        save_state(state)


# ==============================
# PAPER PYRAMIDING
# ==============================

def process_pyramiding(state: Dict[str, Any], data: Dict[str, pd.DataFrame], index: int, equity_ma: float) -> None:
    total_open_risk = position_open_risk(state)
    equity = float(state["equity"])

    for symbol in list(state["positions"].keys()):
        if total_open_risk >= equity * PORT_CAP:
            break

        position = state["positions"][symbol]
        df = data[symbol]
        row = df.iloc[index]

        if not row_is_usable(row):
            continue

        initial_risk = float(position["entry"]) - float(position["stop"])
        if initial_risk <= 0:
            continue

        current_r = (float(row["close"]) - float(position["entry"])) / initial_risk
        if current_r < 1.0 or int(position.get("adds", 0)) >= 1:
            continue

        dynamic_risk = compute_dynamic_risk(row, equity, equity_ma, float(state["risk_multiplier"]))
        if dynamic_risk is None:
            continue

        dynamic_risk *= 0.5
        trade_risk = equity * dynamic_risk
        if total_open_risk + trade_risk > equity * PORT_CAP:
            continue

        stop_distance = STOP_ATR * float(row["atr"])
        if stop_distance <= 0:
            continue

        add_price = get_live_price(symbol, float(row["close"])) * (1 + SLIPPAGE)
        new_size = trade_risk / stop_distance
        current_size = float(position["size"])
        avg_entry = float(position.get("avg_entry", position["entry"]))
        total_size = current_size + new_size
        new_avg = (avg_entry * current_size + add_price * new_size) / total_size

        position["size"] = total_size
        position["avg_entry"] = new_avg
        position["risk"] = float(position["risk"]) + trade_risk
        position["adds"] = int(position.get("adds", 0)) + 1
        state["trade_count"] = int(state["trade_count"]) + 1
        total_open_risk += trade_risk

        logger.info(
            "[PAPER PYRAMID] %s | add_size=%.6f add_price=%.4f new_avg=%.4f total_size=%.6f",
            symbol, new_size, add_price, new_avg, total_size,
        )
        save_state(state)


# ==============================
# HOURLY REPORT
# ==============================

def print_report(state: Dict[str, Any], data: Dict[str, pd.DataFrame], index: int) -> None:
    equity = float(state["equity"])
    peak = float(state["peak_equity"])
    pnl = equity - INITIAL_CAPITAL
    pnl_pct = (pnl / INITIAL_CAPITAL) * 100
    max_dd = float(state["max_drawdown"]) * 100
    trades = int(state["trade_count"])

    open_lines = []
    unrealised_total = 0.0
    for symbol, pos in state["positions"].items():
        live = get_live_price(symbol, float(pos["avg_entry"]))
        avg_entry = float(pos.get("avg_entry", pos["entry"]))
        qty = float(pos["size"])
        upnl = (live - avg_entry) * qty
        unrealised_total += upnl
        sign = "+" if upnl >= 0 else ""
        open_lines.append(
            f"  {symbol:<10} | Entry: ${avg_entry:>10.4f} | Live: ${live:>10.4f} | uPnL: {sign}${upnl:.4f}"
        )

    total_balance = equity + unrealised_total
    sign_pnl = "+" if pnl >= 0 else ""
    sign_upnl = "+" if unrealised_total >= 0 else ""

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  PAPER TRADING REPORT  —  {utc_now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(sep)
    print(f"  Starting Balance  : ${INITIAL_CAPITAL:>10.2f}")
    print(f"  Realised Equity   : ${equity:>10.2f}  ({sign_pnl}{pnl_pct:.2f}%)")
    print(f"  Unrealised PnL    : {sign_upnl}${unrealised_total:.4f}")
    print(f"  Total Balance     : ${total_balance:>10.2f}")
    print(f"  Max Drawdown      : {max_dd:.2f}%")
    print(f"  Total Trades      : {trades}")
    print(f"  Open Positions    : {len(state['positions'])}")
    print(sep)
    if open_lines:
        print("  OPEN POSITIONS:")
        for line in open_lines:
            print(line)
    else:
        print("  No open positions.")
    print(sep + "\n")


# ==============================
# MAIN
# ==============================

def main() -> None:
    logger.info("Starting PAPER trading bot (no real orders will be placed)")
    state = load_state()

    data, latest_closed_timestamp = build_market_data()
    latest_closed_iso = latest_closed_timestamp.isoformat()
    last_processed = state.get("last_processed_candle")
    last_attempted = state.get("last_attempted_candle")

    if last_processed == latest_closed_iso:
        logger.info("Candle %s already processed. Showing report.", latest_closed_iso)
        index = data["BTCUSDT"].index[data["BTCUSDT"]["timestamp"] == latest_closed_timestamp]
        print_report(state, data, int(index[0]) if len(index) else -1)
        return

    if last_attempted == latest_closed_iso and last_processed != latest_closed_iso:
        logger.warning("Candle %s attempted but not completed. Skipping.", latest_closed_iso)
        return

    index_arr = data["BTCUSDT"].index[data["BTCUSDT"]["timestamp"] == latest_closed_timestamp]
    if len(index_arr) == 0:
        raise RuntimeError("Failed to locate the latest closed candle.")

    candle_index = int(index_arr[0])
    if candle_index < MIN_LOOKBACK:
        raise RuntimeError(f"Candle index {candle_index} below required lookback {MIN_LOOKBACK}.")

    logger.info("Processing closed candle at %s", latest_closed_iso)
    state["last_attempted_candle"] = latest_closed_iso
    save_state(state)

    equity_ma = append_equity_snapshot(state)

    process_exits(state, data, candle_index)
    process_entries(state, data, candle_index, equity_ma)
    process_pyramiding(state, data, candle_index, equity_ma)

    update_drawdown_state(state)
    state["last_processed_candle"] = latest_closed_iso
    save_state(state)

    print_report(state, data, candle_index)


if __name__ == "__main__":
    main()
