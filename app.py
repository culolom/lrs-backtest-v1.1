# app.py — LRS 回測系統（含 Benchmark 對照）

import os
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import matplotlib.font_manager as fm
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 字型設定 ===
font_path = "./NotoSansTC-Bold.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans TC"
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# === Streamlit 頁面設定 ===
st.set_page_config(page_title="LRS 回測系統", page_icon="📈", layout="wide")
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 Leverage Rotation Strategy — SMA/EMA 回測系統（含 Benchmark）</h1>", unsafe_allow_html=True)

# === 自動補 .TW 的函式 ===
def normalize_symbol(symbol):
    s = symbol.strip().upper()
    if s.isdigit() or (not "." in s and (s.startswith("00") or s.startswith("23") or s.startswith("008"))):
        s += ".TW"
    return s

# === 使用者輸入 ===
col1, col2, col3 = st.columns(3)
with col1:
    raw_symbol = st.text_input("輸入代號（例：00631L.TW, QQQ, 0050, 2330）", "0050")
symbol = normalize_symbol(raw_symbol)

with col2:
    start = st.date_input("開始日期", pd.to_datetime("2013-01-01"))
with col3:
    end = st.date_input("結束日期", pd.to_datetime("2025-01-01"))

col4, col5, col6 = st.columns(3)
with col4:
    ma_type = st.selectbox("均線種類", ["SMA", "EMA"])
with col5:
    window = st.slider("均線天數", 10, 200, 200, 10)
with col6:
    initial_capital = st.number_input("投入本金（元）", 1000, 1_000_000, 10000, step=1000)

# === 新增 Benchmark 選項 ===
use_benchmark = st.checkbox("加入大盤 Benchmark 對照")
if use_benchmark:
    benchmark_raw = st.text_input("輸入 Benchmark 代號（例：SPY、VT、0050）", "SPY")
    benchmark_symbol = normalize_symbol(benchmark_raw)
else:
    benchmark_symbol = None

# === 主程式 ===
if st.button("開始回測 🚀"):
    start_early = pd.to_datetime(start) - pd.Timedelta(days=365)
    with st.spinner("資料下載中…（自動多抓一年暖機資料）"):
        df_raw = yf.download(symbol, start=start_early, end=end)
        if use_benchmark:
            df_bench = yf.download(benchmark_symbol, start=start_early, end=end)
        else:
            df_bench = None

        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        if df_bench is not None and isinstance(df_bench.columns, pd.MultiIndex):
            df_bench.columns = df_bench.columns.get_level_values(0)

    if df_raw.empty:
        st.error(f"⚠️ 無法下載 {symbol} 的資料，請確認代號或時間區間。")
        st.stop()

    df = df_raw.copy()
    df["MA"] = (
        df["Close"].rolling(window=window).mean()
        if ma_type == "SMA"
        else df["Close"].ewm(span=window, adjust=False).mean()
    )

    # === 生成訊號 ===
    df["Signal"] = 0
    df.loc[df.index[0], "Signal"] = 1
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["MA"].iloc[i] and df["Close"].iloc[i - 1] <= df["MA"].iloc[i - 1]:
            df.loc[df.index[i], "Signal"] = 1
        elif df["Close"].iloc[i] < df["MA"].iloc[i] and df["Close"].iloc[i - 1] >= df["MA"].iloc[i - 1]:
            df.loc[df.index[i], "Signal"] = -1
        else:
            df.loc[df.index[i], "Signal"] = 0

    # === 持倉 ===
    position, current = [], 1
    for sig in df["Signal"]:
        if sig == 1:
            current = 1
        elif sig == -1:
            current = 0
        position.append(current)
    df["Position"] = position

    # === 回報 ===
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Position"]

    # === 資金曲線 ===
    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()
    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    # === Benchmark ===
    if df_bench is not None and not df_bench.empty:
        df_bench["Return"] = df_bench["Close"].pct_change().fillna(0)
        df_bench["Equity_Bench"] = (1 + df_bench["Return"]).cumprod()
        df = df.join(df_bench["Equity_Bench"], how="inner")

    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    df["LRS_Capital"] = df["Equity_LRS"] * initial_capital
    df["BH_Capital"] = df["Equity_BuyHold"] * initial_capital
    if df_bench is not None and not df_bench.empty:
        df["Bench_Capital"] = df["Equity_Bench"] / df["Equity_Bench"].iloc[0] * initial_capital

    # === 指標 ===
    def calc(series):
        total = series.iloc[-1] - 1
        cagr = (1 + total) ** (1 / ((df.index[-1] - df.index[0]).days / 365)) - 1
        mdd = 1 - (series / series.cummax()).min()
        return total, cagr, mdd

    final_lrs, cagr_lrs, mdd_lrs = calc(df["Equity_LRS"])
    final_bh, cagr_bh, mdd_bh = calc(df["Equity_BuyHold"])
    if df_bench is not None and "Equity_Bench" in df:
        final_bench, cagr_bench, mdd_bench = calc(df["Equity_Bench"])
    else:
        final_bench = cagr_bench = mdd_bench = np.nan

    # === 圖表 ===
    st.markdown("<h2>📈 策略績效視覺化</h2>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("收盤價與均線（含買賣點）", "資金曲線比較"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="收盤價", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type}{window}", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_LRS"], name="LRS 策略", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_BuyHold"], name="Buy & Hold", line=dict(color="gray", dash="dot")), row=2, col=1)
    if df_bench is not None and "Equity_Bench" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["Equity_Bench"], name=f"Benchmark ({benchmark_symbol})", line=dict(color="purple", dash="dash")), row=2, col=1)
    fig.update_layout(height=800, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # === 報表 ===
    html_table = f"""
    <table style='width:100%; border-collapse:collapse; margin-top:1em; font-family:"Noto Sans TC";'>
    <thead><tr><th>指標名稱</th><th>LRS 策略</th><th>Buy & Hold</th><th>Benchmark</th></tr></thead>
    <tbody>
    <tr><td>總報酬</td><td>{final_lrs:.2%}</td><td>{final_bh:.2%}</td><td>{final_bench:.2%}</td></tr>
    <tr><td>年化報酬</td><td>{cagr_lrs:.2%}</td><td>{cagr_bh:.2%}</td><td>{cagr_bench:.2%}</td></tr>
    <tr><td>最大回撤</td><td>{mdd_lrs:.2%}</td><td>{mdd_bh:.2%}</td><td>{mdd_bench:.2%}</td></tr>
    </tbody></table>
    """
    st.markdown(html_table, unsafe_allow_html=True)

    st.success("✅ 回測完成！（含 Benchmark 對照）")
