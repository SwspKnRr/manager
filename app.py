import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="실전 포트폴리오", layout="wide")
st.markdown("<style>.big-font{font-size:52px !important;font-weight:bold;}.pos{color:#e62e2e;font-size:28px;font-weight:bold}.neg{color:#0066ff;font-size:28px;font-weight:bold}</style>", unsafe_allow_html=True)

# 포트폴리오 영구 저장
if "data" not in st.session_state:
    try:
        with open("p.json") as f:
            saved = json.load(f)
            st.session_state.data = pd.DataFrame(saved["h"])
            st.session_state.cash = float(saved["c"])
    except:
        st.session_state.data = pd.DataFrame(columns=["ticker","shares","avg_price"])
        st.session_state.cash = 10000.0

df = st.session_state.data

# ------------------------------- 사이드바 -------------------------------
st.sidebar.header("💼 포트폴리오 입력 (USD 기준)")

# 포트폴리오 DataFrame (항상 세션에 존재)
if "portfolio" not in st.session_state:
    try:
        with open("portfolio.json", "r") as f:
            data = json.load(f)
            st.session_state.portfolio = pd.DataFrame(data["holdings"])
            st.session_state.cash_usd = float(data["cash"])
    except:
        st.session_state.portfolio = pd.DataFrame(columns=["ticker", "shares", "avg_price"])
        st.session_state.cash_usd = 10000.0

df = st.session_state.portfolio

# 종목 추가/수정 폼
with st.sidebar.form(key="add_stock_form"):
    ticker = st.text_input("티커", placeholder="QQQ, TQQQ 등").upper().strip()
    shares = st.number_input("보유 주수", min_value=0, step=1, value=0)
    avg_price = st.number_input("평균 단가 (USD)", min_value=0.0, format="%.2f", value=0.0)
    
    if st.form_submit_button("✅ 추가/수정") and ticker:
        if ticker = ticker.upper().strip()
        if ticker in df["ticker"].values:
            df.loc[df.ticker == ticker, ["shares", "avg_price"]] = [shares, avg_price]
            st.success(f"{ticker} 수정 완료")
        else:
            new_row = pd.DataFrame([{"ticker": ticker, "shares": shares, "avg_price": avg_price}])
            df = pd.concat([df, new_row], ignore_index=True)
            st.success(f"{ticker} 추가 완료")
        
        # 저장
        st.session_state.portfolio = df
        with open("portfolio.json", "w") as f:
            json.dump({
                "holdings": df.to_dict("records"),
                "cash": float(st.session_state.cash_usd)
            }, f)
        st.rerun()

# 현금 잔고 입력 (실시간 저장)
st.sidebar.markdown("---")
current_cash = st.sidebar.number_input(
    "💰 현금 잔고 (USD)",
    min_value=0.0,
    value=float(st.session_state.cash_usd),
    step=500.0,
    format="%.2f"
)

# 현금 바뀌면 바로 저장
if abs(current_cash - st.session_state.cash_usd) > 0.01:
    st.session_state.cash_usd = current_cash
    with open("portfolio.json", "w") as f:
        json.dump({
            "holdings": st.session_state.portfolio.to_dict("records"),
            "cash": float(st.session_state.cash_usd)
        }, f)
    st.rerun()  # UI 즉시 반영

# 포트폴리오 초기화 버튼 (선택사항)
if st.sidebar.button("🗑️ 포트폴리오 초기화"):
    st.session_state.portfolio = pd.DataFrame(columns=["ticker", "shares", "avg_price"])
    st.session_state.cash_usd = 0.0
    with open("portfolio.json", "w") as f:
        json.dump({"holdings": [], "cash": 0.0}, f)
    st.success("초기화 완료")
    st.rerun()

if df.empty:
    st.warning("종목 추가해라 임마")
    st.stop()

tickers = df["ticker"].tolist()

# 데이터
@st.cache_data(ttl=180)
def load_data(t):
    return yf.download(t, period="5y", progress=False, auto_adjust=True)["Close"]

prices = load_data(tickers)
current = prices.iloc[-1]

# 계산
port = df.copy()
port["price"] = current.reindex(port["ticker"]).values
port["value"] = port["shares"] * port["price"]
port["profit"] = port["value"] - port["shares"]*port["avg_price"]
port["pct"] = port["profit"] / (port["shares"]*port["avg_price"]) * 100

total_value = port["value"].sum() + st.session_state.cash
total_ret = (total_value / (port["shares"]*port["avg_price"]).sum() + st.session_state.cash - st.session_state.cash) * 100

# 헤더
c1,_ = st.columns([1,3])
with c1:
    st.markdown(f'<p class="big-font">${total_value:,.0f}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="{"pos" if total_ret>=0 else "neg"}">{total_ret:+.2f}%</p>', unsafe_allow_html=True)

# 그래프
hist = prices.mul(port.set_index("ticker")["shares"], axis=1).sum(axis=1) + st.session_state.cash
fig = go.Figure(go.Scatter(x=hist.index, y=hist, line=dict(color="#e62e2e", width=3)))
fig.update_layout(height=320, margin=dict(t=20,b=0,l=0,r=0), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_visible=False, yaxis_visible=False)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# 테이블
disp = port[["ticker","shares","avg_price","price","value","pct"]].round(2)
disp.columns = ["티커","주수","평균단가","현재가","평가액","수익률%"]
disp["주수"] = disp["주수"].astype(int).astype(str)+"주"
st.dataframe(disp.style.format({"평균단가":"${:.2f}","현재가":"${:.2f}","평가액":"${:,.0f}","수익률%":"{:+.2f}%"}), use_container_width=True, hide_index=True)

# 탭
tab1, tab2, tab3 = st.tabs(["리밸런싱", "오늘 신호", "가격 예측"])

with tab1:
    target = st.selectbox("종목", tickers)
    if st.button("최적 전략 찾기"):
        with st.spinner("백테스팅 중..."):
            close = yf.download(target, period="5y", progress=False)["Close"]
            ret = close.pct_change().fillna(0)
            best = -1
            for up in np.arange(0.1, 0.4, 0.05):
                for down in np.arange(-0.3, -0.08, 0.05):
                    for ratio in [0.5, 0.8, 1.0]:
                        cash = 2000.0
                        shares = 10000 / close.iloc[0]
                        for i in range(1, len(close)):
                            if ret.iloc[i] >= up:
                                sell = shares * ratio
                                cash += sell * close.iloc[i]
                                shares -= sell
                            elif ret.iloc[i] <= down and cash > 500:
                                buy = cash * 0.8 / close.iloc[i]
                                shares += buy
                                cash -= buy * close.iloc[i]
                        final = shares * close.iloc[-1] + cash
                        cagr = (final/12000)**(1/5)-1
                        if cagr > best:
                            best = cagr
                            best_p = (up, down, ratio, final)
            u,d,r,f = best_p
            st.success(f"**최적**\n+{u:.1%} ↑ → {r:.0%} 매도\n{d:.1%} ↓ → 현금 80% 매수\n→ 5년 {f:,.0f}달러 (CAGR {best:.1%})")

with tab2:
    scores = {}
    for t in tickers:
        try:
            d = yf.download(t, period="1y", progress=False)
            c = d["Close"]
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rsi = 100 - 100/(1 + gain/loss.replace(0, 1e-10))
            rsi_val = rsi.iloc[-1]

            score = 50
            if rsi_val < 30: score += 35
            if rsi_val > 70: score -= 35
            if c.iloc[-1] < c.rolling(20).mean().iloc[-1] - 2*c.rolling(20).std().iloc[-1]: score += 25
            macd = c.ewm(12).mean() - c.ewm(26).mean()
            if macd.iloc[-1] > macd.ewm(9).mean().iloc[-1] and macd.iloc[-2] <= macd.ewm(9).mean().iloc[-2]: score += 20
            scores[t] = min(100, max(0, int(score)))
        except:
            scores[t] = 50
    sdf = pd.DataFrame(list(scores.items()), columns=["티커","점수"]).sort_values("점수", ascending=False)
    sdf["신호"] = pd.cut(sdf["점수"], bins=[0,40,65,85,100], labels=["🔴 매도","⚪ 관망","🟢 매수","🟢🟢 강력매수"])
    st.dataframe(sdf, use_container_width=True, hide_index=True)

with tab3:
    ticker = st.selectbox("예측 종목", tickers)
    if st.button("예측 시작"):
        with st.spinner("학습 중..."):
            raw = yf.download(ticker, period="5y", progress=False)
            train = pd.DataFrame({"ds": raw.index, "y": raw["Close"].values})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(train)
            future = m.make_future_dataframe(30)
            forecast = m.predict(future)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="실제"))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="예측", line=dict(color="#e62e2e")))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], line=dict(width=0), fill="tonexty", fillcolor="rgba(100,150,255,0.2)", name="구간"))
            st.plotly_chart(fig, use_container_width=True)
            curr = raw["Close"].iloc[-1]
            tmr = forecast[forecast["ds"] > train["ds"].iloc[-1]].iloc[0]["yhat"]
            w7 = forecast.iloc[-24]["yhat"]
            m30 = forecast.iloc[-1]["yhat"]
            st.metric("현재", f"${curr:.2f}")
            st.metric("내일 예상", f"${tmr:.2f}", f"{(tmr/curr-1)*100:+.2f}%")
            st.metric("+7일", f"${w7:.2f}", f"{(w7/curr-1)*100:+.2f}%")
            st.metric("+30일", f"${m30:.2f}", f"{(m30/curr-1)*100:+.2f}%", delta_color="normal")

st.caption("2025.11.22 — ")