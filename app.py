import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ------------------- 페이지 설정 & 토스증권 스타일 CSS -------------------
st.set_page_config(page_title="토스증권 스타일 포트폴리오", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .big-font {font-size:50px !important; font-weight:bold; color:#111111;}
    .profit-positive {color:#e62e2e; font-weight:bold;}
    .profit-negative {color:#0066ff; font-weight:bold;}
    .ticker-title {font-size:24px; font-weight:bold; margin-bottom:5px;}
    .metric-label {font-size:14px; color:#666;}
    section[data-testid="stSidebar"] {background-color:#f8f9fa;}
    .css-1d391kg {padding-top: 2rem;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ------------------- 포트폴리오 저장/로드 -------------------
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)
            return pd.DataFrame(data["holdings"]), float(data["cash_usd"])
    except:
        return pd.DataFrame(columns=["ticker", "shares", "avg_price"]), 10000.0  # 기본 현금 1만불

def save_portfolio(df, cash):
    data = {"holdings": df.to_dict("records"), "cash_usd": float(cash)}
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f)

holdings_df, cash_usd = load_portfolio()

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = holdings_df
    st.session_state.cash_usd = cash_usd

# ------------------- 사이드바 -------------------
st.sidebar.header("💼 내 포트폴리오 (USD 기준)")

with st.sidebar.form("add_stock"):
    ticker = st.text_input("티커 (예: QQQ)", "").upper().strip()
    shares = st.number_input("보유 주수", min_value=0, step=1, value=0)
    avg_price = st.number_input("평균 단가 (USD)", min_value=0.0, format="%.2f")
    add_btn = st.form_submit_button("추가/수정")

    if add_btn and ticker:
        if ticker in st.session_state.portfolio['ticker'].values:
            idx = st.session_state.portfolio[st.session_state.portfolio['ticker'] == ticker].index[0]
            st.session_state.portfolio.loc[idx, ['shares', 'avg_price']] = [shares, avg_price]
        else:
            new_row = pd.DataFrame([{"ticker": ticker, "shares": shares, "avg_price": avg_price}])
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
        save_portfolio(st.session_state.portfolio, st.session_state.cash_usd)
        st.success(f"{ticker} 저장 완료!")

st.sidebar.write("#### 현금 잔고 (USD)")
st.session_state.cash_usd = st.sidebar.number_input("", value=float(st.session_state.cash_usd), step=100.0, format="%.2f")

if st.sidebar.button("💾 전체 저장"):
    save_portfolio(st.session_state.portfolio, st.session_state.cash_usd)
    st.sidebar.success("포트폴리오 저장 완료!")

# 포트폴리오 없으면 강제 종료
if st.session_state.portfolio.empty:
    st.warning("좌측에서 포트폴리오를 먼저 입력해주세요!")
    st.stop()

# ------------------- 실시간 데이터 -------------------
tickers = st.session_state.portfolio['ticker'].tolist()
data = yf.download(tickers, period="5y", progress=False)['Adj Close']
current_prices = data.iloc[-1]

# 현재 평가액 계산
st.session_state.portfolio['current_price'] = st.session_state.portfolio['ticker'].map(current_prices)
st.session_state.portfolio['value'] = st.session_state.portfolio['shares'] * st.session_state.portfolio['current_price']
st.session_state.portfolio['cost'] = st.session_state.portfolio['shares'] * st.session_state.portfolio['avg_price']
st.session_state.portfolio['profit'] = st.session_state.portfolio['value'] - st.session_state.portfolio['cost']
st.session_state.portfolio['profit_pct'] = st.session_state.portfolio['profit'] / st.session_state.portfolio['cost'] * 100

total_value = st.session_state.portfolio['value'].sum() + st.session_state.cash_usd
total_cost = st.session_state.portfolio['cost'].sum() + st.session_state.cash_usd
total_return = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0

# ------------------- 헤더 -------------------
col1, col2 = st.columns([2, 3])
with col1:
    st.markdown(f'<p class="big-font">${total_value:,.2f}</p>', unsafe_allow_html=True)
    color = "profit-positive" if total_return >= 0 else "profit-negative"
    st.markdown(f'<p class="{color}">{total_return:+.2f}%</p>', unsafe_allow_html=True)

# ------------------- 포트폴리오 가치 그래프 -------------------
history_value = data * st.session_state.portfolio.set_index('ticker')['shares'].reindex(data.columns).fillna(0).values
history_value['Total'] = history_value.sum(axis=1) + st.session_state.cash_usd
history_value = history_value['Total'].resample('D').last().ffill()

fig = go.Figure()
fig.add_trace(go.Scatter(x=history_value.index, y=history_value.values, line=dict(color="#e62e2e", width=3)))
fig.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False, showticklabels=False)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ------------------- 보유 종목 테이블 -------------------
display_df = st.session_state.portfolio[['ticker', 'shares', 'avg_price', 'current_price', 'value', 'profit_pct']].copy()
display_df.columns = ['티커', '주수', '평균단가', '현재가', '평가액', '수익률(%)']
display_df['주수'] = display_df['주수'].astype(int).astype(str) + "주"
display_df = display_df.round(2)
st.dataframe(display_df.style.format({"평균단가": "${:.2f}", "현재가": "${:.2f}", "평가액": "${:,.0f}", "수익률(%)": "{:+.2f}%"}), use_container_width=True, hide_index=True)

# ------------------- 탭 -------------------
tab1, tab2, tab3 = st.tabs(["리밸런싱 가이드", "오늘 매수/매도 추천", "가격 예측"])

with tab1:
    st.write("#### 최적 리밸런싱 파라미터 (과거 5년 백테스트)")
    target = st.selectbox("대상 종목", tickers)
    if st.button("최고 파라미터 검색"):
        with st.spinner("백테스팅 중..."):
            price = yf.download(target, period="5y")['Adj Close']
            returns = price.pct_change().dropna()

            best_cagr = -999
            best_param = None

            for up in np.arange(0.10, 0.40, 0.05):
                for down in np.arange(-0.30, -0.08, 0.03):
                    for sell_ratio in [0.5, 0.7, 1.0]:
                        equity = 10000
                        cash = 2000
                        shares = equity / price.iloc[0]

                        for r in returns:
                            if r >= up:
                                sell_shares = shares * sell_ratio
                                cash += sell_shares * price.loc[r.name] * (1 + r)
                                shares -= sell_shares
                            elif r <= down:
                                buy_shares = cash * 0.8 / price.loc[r.name]
                                shares += buy_shares
                                cash -= buy_shares * price.loc[r.name]

                        final = shares * price.iloc[-1] + cash
                        cagr = (final / 12000) ** (1/5) - 1

                        if cagr > best_cagr:
                            best_cagr = cagr
                            best_param = (up, down, sell_ratio, final)

            up, down, ratio, final = best_param
            st.success(f"""
            **최적 파라미터 발견!**\n
            → {target}이 **+{up*100:.1f}%** 오르면 → 보유 주식의 **{ratio*100:.0f}% 매도**\n
            → {target}이 **{down*100:.1f}%** 내리면 → 현금의 80%로 물타기\n
            → 5년 백테스트 결과: 최종 자산 **${final:,.0f}** (CAGR {best_cagr*100:+.2f}%)
            """)

with tab2:
    st.write("#### 오늘 매수/매도 점수 (100점 만점)")
    scores = {}
    
    # 여기만 고쳤어요! tick64 → tickers
    for t in tickers:      # ← 이 줄 수정!
        df = yf.download(t, period="1y", progress=False)
        close = df['Close']

        # RSI 계산 (더 안정적인 방법)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()

        # 볼린저 밴드
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        score = 50

        # RSI 과매도/과매수
        if rsi.iloc[-1] < 30:   score += 30
        if rsi.iloc[-1] > 70:   score -= 35

        # 볼린저 하단 이탈
        if close.iloc[-1] < bb_lower.iloc[-1]: score += 25

        # MACD 골든크로스
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]: score += 20

        # 모멘텀 (20일 수익률)
        momentum = close.iloc[-1] / close.iloc[-21] - 1
        if momentum > 0.15: score += 15

        scores[t] = min(100, max(0, int(score)))

    score_df = pd.DataFrame(scores.items(), columns=["티커", "점수"])
    score_df = score_df.sort_values("점수", ascending=False)
    score_df["추천"] = score_df["점수"].apply(
        lambda x: "🟢🟢 강력 매수" if x >= 85 else
                  "🟢 매수 고려" if x >= 70 else
                  "🔴 매도 고려" if x <= 40 else
                  "⚪ 관망"
    )
    st.dataframe(score_df, use_container_width=True, hide_index=True)

with tab3:
    st.write("#### 가격 예측 (Prophet 기반)")
    ticker = st.selectbox("예측 종목", tickers, key="pred")
    if st.button("예측 실행"):
        df = yf.download(ticker, period="5y", progress=False)[['Close']].reset_index()
        df.columns = ['ds', 'y']

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="실제"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="예측"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode="lines", line_color="rgba(0,0,0,0)"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode="lines", fillcolor="rgba(100,100,255,0.2)", name="80% 구간"))
        st.plotly_chart(fig, use_container_width=True)

        curr = current_prices[ticker]
        tomorrow = forecast.iloc[-30]['yhat']
        week = forecast.iloc[-24]['yhat']
        month = forecast.iloc[-1]['yhat']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재", f"${curr:.2f}")
        c2.metric("내일", f"${tomorrow:.2f}", f"{(tomorrow/curr-1)*100:+.1f}%")
        c3.metric("7일후", f"${week:.2f}", f"{(week/curr-1)*100:+.1f}%")
        c4.metric("30일후", f"${month:.2f}", f"{(month/curr-1)*100:+.1f}%")

st.caption("2025년 11월 실전 버전 • 이제 진짜 됩니다")