"""
BIST hisseleri için sinyal botu. Kripto sistemiyle AYNI kurallarla çalışır:

1. STRATEJİ SİNYALİ: MA9/MA21 kesişimi + EMA200 trend + MACD + RSI + ADX +
   hacim onayının HEPSİNİN sağlandığı, sıkı filtreli AL/SAT sinyali.
2. ANORMAL HACİM ALARMI: Bir hissenin işlem hacmi kendi ortalamasının
   belirgin şekilde üzerine çıkarsa (büyük/kurumsal bir hareketin dolaylı
   işareti) bildirim gönderir. Gerçek emir defteri takibi DEĞİLDİR.

VERİ KAYNAĞI: Yahoo Finance (yfinance kütüphanesi, ücretsiz, anahtar
gerektirmez). BIST hisseleri Yahoo'da ".IS" uzantısıyla işlem görür
(örn. GARAN.IS). Not: Bu veri kaynağı bazen saatlik BIST verisinde
küçük aksaklıklar/boşluklar verebilir — bir hisse için veri çekilemezse
sistem o hisseyi o turda atlar, hata vermeden devam eder.

HİSSE LİSTESİ: Aşağıdaki liste, BIST'te işlem gören büyük/likit
şirketlerden oluşan bir seçkidir. BIST 100 endeksi yılda 4 kez
(Ocak/Nisan/Temmuz/Ekim) güncellendiği için bu liste MUTLAK GÜNCEL
resmi liste OLMAYABİLİR. Güncel resmi listeyi borsaistanbul.com'dan
kontrol edip BIST_TICKERS listesine ekleme/çıkarma yapabilirsiniz.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# AYARLAR
# ============================================================

NTFY_TOPIC = "buraya-kendi-gizli-bist-konu-adinizi-yazin-4471"

# Büyük/likit BIST hisseleri (siz ekleyip çıkarabilirsiniz, .IS uzantısı ile)
BIST_TICKERS = [
    "GARAN.IS", "AKBNK.IS", "ISCTR.IS", "YKBNK.IS", "VAKBN.IS", "HALKB.IS", "TSKB.IS",
    "KCHOL.IS", "SAHOL.IS", "SISE.IS", "DOAS.IS", "ALARK.IS", "ENKAI.IS", "TAVHL.IS",
    "EREGL.IS", "KRDMD.IS", "KOZAL.IS", "KOZAA.IS", "ISDMR.IS",
    "TUPRS.IS", "PETKM.IS", "AYGAZ.IS", "GUBRF.IS", "AKSEN.IS", "ODAS.IS", "ZOREN.IS", "ENJSA.IS",
    "ASELS.IS", "THYAO.IS", "TTKOM.IS", "TCELL.IS", "LOGO.IS", "NETAS.IS",
    "BIMAS.IS", "MGROS.IS", "SOKM.IS", "VESTL.IS", "ARCLK.IS", "TTRAK.IS",
    "ULKER.IS", "CCOLA.IS", "AEFES.IS", "TATGD.IS", "KENT.IS",
    "CIMSA.IS", "AKCNS.IS", "BUCIM.IS",
    "EKGYO.IS", "ISGYO.IS", "TRGYO.IS",
    "TURSG.IS", "ANHYT.IS", "AGESA.IS",
    "FROTO.IS", "TOASO.IS",
    "SASA.IS", "ASTOR.IS", "MAVI.IS", "PGSUS.IS", "TKFEN.IS", "HEKTS.IS", "KARSN.IS", "OTKAR.IS",
]

# --- Veri ayarları ---
YF_INTERVAL = "60m"       # Saatlik mum
YF_PERIOD = "90d"         # Yeterli geçmiş veri için (EMA200 hesaplanabilsin diye)

# --- Kesişim (tetikleyici) ---
MA_SHORT_PERIOD = 9
MA_LONG_PERIOD = 21

# --- Uzun vadeli trend filtresi ---
EMA_TREND_PERIOD = 200

# --- MACD ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- RSI ---
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- ADX (trend gücü) ---
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# --- Hacim onayı / anormal hacim alarmı ---
VOLUME_MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.2          # Strateji sinyali için hacim onay eşiği
BIG_VOLUME_MULTIPLIER = 3.0      # Anormal hacim alarmı için eşik

REQUEST_DELAY_SECONDS = 0.5

# ============================================================
# Buradan sonrasını değiştirmenize gerek yok
# ============================================================

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")


def get_stock_history(ticker: str) -> pd.DataFrame:
    data = yf.Ticker(ticker).history(period=YF_PERIOD, interval=YF_INTERVAL)
    if data is None or data.empty:
        raise RuntimeError("Veri boş döndü")
    data = data.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    return data[["open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _rsi(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


def _macd(closes: pd.Series):
    ema_fast = closes.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = closes.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd_line, signal_line


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-12)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def compute_strategy_signal(df: pd.DataFrame) -> dict:
    min_needed = max(EMA_TREND_PERIOD + 20, MACD_SLOW + MACD_SIGNAL, ADX_PERIOD * 3, VOLUME_MA_PERIOD) + 2
    if len(df) < min_needed:
        return {"signal": "HOLD", "price": float(df["close"].iloc[-1]) if len(df) else None, "detail": "Yeterli veri yok"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    crossed_up = prev["ma_short"] <= prev["ma_long"] and last["ma_short"] > last["ma_long"]
    crossed_down = prev["ma_short"] >= prev["ma_long"] and last["ma_short"] < last["ma_long"]

    if not (crossed_up or crossed_down):
        return {"signal": "HOLD", "price": float(last["close"]), "detail": "Kesişim yok"}

    direction = "BUY" if crossed_up else "SELL"

    checks = {}
    if direction == "BUY":
        checks["trend"] = last["close"] > last["ema_trend"]
        checks["macd"] = last["macd_line"] > last["macd_signal"]
        checks["rsi"] = 50 < last["rsi"] < RSI_OVERBOUGHT
        checks["adx"] = last["adx"] > ADX_THRESHOLD
        checks["volume"] = last["volume"] > last["volume_ma"] * VOLUME_MULTIPLIER
    else:
        checks["trend"] = last["close"] < last["ema_trend"]
        checks["macd"] = last["macd_line"] < last["macd_signal"]
        checks["rsi"] = RSI_OVERSOLD < last["rsi"] < 50
        checks["adx"] = last["adx"] > ADX_THRESHOLD
        checks["volume"] = last["volume"] > last["volume_ma"] * VOLUME_MULTIPLIER

    passed = sum(checks.values())
    total = len(checks)
    detail_parts = [f"{name}={'✓' if ok else '✗'}" for name, ok in checks.items()]
    detail = f"Kesişim + {passed}/{total} onay ({', '.join(detail_parts)}) RSI={last['rsi']:.1f} ADX={last['adx']:.1f}"

    if passed == total:
        return {"signal": direction, "price": float(last["close"]), "detail": detail}

    return {"signal": "HOLD", "price": float(last["close"]), "detail": f"Kesişim var ama filtre geçmedi: {detail}"}


def check_volume_spike(df: pd.DataFrame):
    if len(df) < VOLUME_MA_PERIOD + 2:
        return False, None, None
    last_vol = df["volume"].iloc[-1]
    last_vol_ma = df["volume_ma"].iloc[-1]
    last_price = float(df["close"].iloc[-1])
    if pd.isna(last_vol_ma) or last_vol_ma <= 0:
        return False, None, last_price
    ratio = last_vol / last_vol_ma
    return ratio >= BIG_VOLUME_MULTIPLIER, ratio, last_price


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def send_notification(title: str, message: str, priority: str = "default"):
    if "buraya-kendi" in NTFY_TOPIC:
        print("⚠️ UYARI: NTFY_TOPIC hâlâ varsayılan değerde. check_signal.py içinde değiştirin!")
        return
    import requests
    requests.post(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=message.encode("utf-8"),
        headers={"Title": title.encode("utf-8"), "Priority": priority},
        timeout=15,
    )


def main():
    state = load_state()
    print(f"{len(BIST_TICKERS)} BIST hissesi taranacak.")

    for ticker in BIST_TICKERS:
        pair_state = state.get(ticker, {})

        try:
            raw = get_stock_history(ticker)
            df = raw.copy()
            df["ma_short"] = df["close"].rolling(window=MA_SHORT_PERIOD).mean()
            df["ma_long"] = df["close"].rolling(window=MA_LONG_PERIOD).mean()
            df["ema_trend"] = df["close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
            df["rsi"] = _rsi(df["close"], RSI_PERIOD)
            df["macd_line"], df["macd_signal"] = _macd(df["close"])
            df["adx"] = _adx(df, ADX_PERIOD)
            df["volume_ma"] = df["volume"].rolling(window=VOLUME_MA_PERIOD).mean()
        except Exception as exc:
            print(f"{ticker}: veri alınamadı, atlanıyor -> {exc}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        # --- 1) STRATEJİ SİNYALİ ---
        result = compute_strategy_signal(df)
        print(f"{ticker}: {result['signal']} | {result['detail']}")

        if result["signal"] != "HOLD" and pair_state.get("last_signal") != result["signal"]:
            action_tr = "AL" if result["signal"] == "BUY" else "SAT"
            display_name = ticker.replace(".IS", "")
            send_notification(
                title=f"🔔 {action_tr} sinyali (sıkı filtre) — {display_name}",
                message=(
                    f"Fiyat: {result['price']:,.2f} TL\n{result['detail']}\n\n"
                    f"Yatırım kuruluşunuzun uygulamasından işlemi kendiniz yapabilirsiniz."
                ),
                priority="high",
            )
            print(f"  ✅ Strateji bildirimi gönderildi ({ticker}).")
            pair_state["last_signal"] = result["signal"]
        elif result["signal"] != "HOLD":
            print(f"  (Aynı sinyal zaten gönderilmişti: {ticker})")

        # --- 2) ANORMAL HACİM ALARMI ---
        is_spike, ratio, price = check_volume_spike(df)
        display_name = ticker.replace(".IS", "")
        if is_spike and not pair_state.get("volume_alert_active", False):
            send_notification(
                title=f"🏦 Anormal hacim artışı — {display_name}",
                message=(
                    f"İşlem hacmi ortalamanın {ratio:.1f} katına çıktı "
                    f"(olası büyük/kurumsal hareket). Fiyat: {price:,.2f} TL.\n\n"
                    f"Bu kesin bir sinyal değildir, dikkat çekici bir anormalliktir."
                ),
                priority="high",
            )
            print(f"  🏦 Hacim alarmı gönderildi ({ticker}, oran={ratio:.1f}).")
            pair_state["volume_alert_active"] = True
        elif not is_spike:
            pair_state["volume_alert_active"] = False

        pair_state["last_checked"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        state[ticker] = pair_state
        time.sleep(REQUEST_DELAY_SECONDS)

    save_state(state)
    print("Tarama tamamlandı.")


if __name__ == "__main__":
    main()
