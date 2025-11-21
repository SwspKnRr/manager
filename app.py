import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------- 캐시 ----------------------------------
@st.cache_data(ttl=180)
def get_price_data(tickers, period="5y"):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data

# ---------------------------------- 페이지 설정 ----------------------------------
st.set_page_config(page_title="실전 포트폴리오", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .big-font {font-size:52px !important; font-weight:bold; color:#111;}
    .profit-positive {color:#e62e2e; font-size:28px; font-weight:bold;}
    .profit-negative {color:#0066ff; font-size:28px; font-weight:bold;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------- 포트폴리오 로드 ----------------------------------
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)
            return pd.DataFrame(data["holdings"]), float(data["cash_usd"])
    except:
        return pd.DataFrame(columns=["ticker", "shares", "avg_price"]), 10000.0

def save_portfolio():
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump({"holdings": st.session_state.portfolio.to_dict("records"),
                   "cash_usd": float(st.session_state.cash_usd)}, f)

if 'portfolio' not in st.session_state:
    df, cash = load_portfolio()
    st.session_state.portfolio = df
    st.session_state.cash_usd = cash

# ---------------------------------- 사이드바 ----------------------------------
st.sidebar.header("💼 내 포트폴리오 (USD)")

with st.sidebar.form("add_form"):
    ticker = st.text_input("티커", placeholder="QQQ, TQQQ 등").upper().strip()
    shares = st.number_input("보유 주수", min_value=0, step=1, value=0)
    avg_price = st.number_input("평균 단가 (USD)", min_value=0.0, format="%.2f")
    if st.form_submit_button("추가/수정"):
        if ticker:
            if ticker in st.session_state.portfolio['ticker'].values:
                st.session_state.portfolio.loc[st.session_state.portfolio.ticker == ticker, ['shares', 'avg_price']] = [shares, avg_price]
            else:
                new = pd.DataFrame([{"ticker": ticker, "shares": shares, "avg_price": avg_price}])
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new], ignore_index=True)
            save_portfolio()
            st.success(f"{ticker} 저장 완료")
            st.rerun()

st.sidebar.number_input("현금 잔고 (USD)", min_value=0.0, value=float(st.session_state.cash_usd),
                        key="cash_usd", on_change=save_portfolio)

if st.session_state.portfolio.empty:
    st.warning("좌측 사이드바에서 종목을 추가해주세요!")
    st.stop()

tickers = st.session_state.portfolio['ticker'].tolist()

# ---------------------------------- 데이터 ----------------------------------
price_history = get_price_data(tickers, "5y")
if price_history.empty or price_history.iloc[-1].isna().all():
    st.error("티커 데이터 오류. 티커명을 확인해주세요 (예: QQQ, TSLA)")
    st.stop()

current_prices = price_history.iloc[-1].fillna(method='ffill')

# ---------------------------------- 포트폴리오 계산 ----------------------------------
p = st.session_state.portfolio.copy()
p['current_price'] = p['ticker'].map(current_prices)
p = p[p['current_price'].notna()]
p['value'] = p['shares'] * p['current_price']
p['cost'] = p['shares'] * p['avg_price']
p['profit'] = p['value'] - p['cost']
p['profit_pct'] = p['profit'] / p['cost'] * 100

total_value = p['value'].sum() + st.session_state.cash_usd
total_cost = p['cost'].sum() + st.session_state.cash_usd
total_return = (total_value - total_cost) / total_cost * 100

# ---------------------------------- 헤더 ----------------------------------
col1, _ = st.columns([1,2])
with col1:
    st.markdown(f'<p class="big-font">${total_value:,.0f}</p>', unsafe_allow_html=True)
    color = "profit-positive" if total_return >= 0 else "profit-negative"
    st.markdown(f'<p class="{color}">{total_return:+.2f}%</p>', unsafe_allow_html=True)

# ---------------------------------- 그래프 ----------------------------------
value_hist = price_history.multiply(p.set_index('ticker')['shares'], axis=1).sum(axis=1) + st.session_state.cash_usd
value_hist = value_hist.fillna(method='ffill').ffill()

fig = go.Figure()
fig.add_trace(go.Scatter(x=value_hist.index, y=value_hist.values, line=dict(color="#e62e2e", width=3)))
fig.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                 xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False))
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ---------------------------------- 보유 종목 ----------------------------------
disp = p[['ticker', 'shares', 'avg_price', 'current_price', 'value', 'profit_pct']].copy()
disp.columns = ['티커','주수','평균단가','현재가','평가액','수익률%']
disp['주수'] = disp['주수'].astype(int).astype(str) + "주"
disp = disp.round(2)
st.dataframe(disp.style.format({"평균단가":"${:.2f}","현재가":"${:.2f}","평가액":"${:,.0f}","수익률%":"{:+.2f}%"}),
             use_container_width=True, hide_index=True)

# ---------------------------------- 탭 ----------------------------------
tab1, tab2, tab3 = st.tabs(["리밸런싱 가이드", "오늘 매수/매도", "가격 예측"])

with tab1:
    target = st.selectbox("대상 종목", tickers)
    if st.button("최적 파라미터 검색"):
        with st.spinner("5년 백테스팅 중... (30초 정도 걸려요)"):
            df = yf.download(target, period="5y", progress=False)['Close']
            if len(df) < 200:
                st.error("데이터 부족")
            else:
                ret = df.pct_change().fillna(0)
                best_cagr = -999
                best_param = None
                for up in np.arange(0.08, 0.36, 0.04):
                    for down in np.arange(-0.30, -0.06, 0.04):
                        for ratio in [0.5, 0.75, 1.0]:
                            cash = 2000.0
                            shares = 10000.0 / df.iloc[0]
                            for i in range(1, len(df)):
                                r = ret.iloc[i]
                                price = df.iloc[i]
                                if r >= up:
                                    sell = shares * ratio
                                    cash += sell * price
                                    shares -= sell
                                elif r <= down and cash > 0:
                                    buy = cash * 0.8 / price
                                    shares += buy
                                    cash -= buy * price
                            final = shares * df.iloc[-1] + cash
                            cagr = (final / 12000) ** (1/5) - 1
                            if cagr > best_cagr:
                                best_cagr = cagr
                                best_param = (up, down, ratio, final)
                up, down, ratio, final = best_param
                st.success(f"""
                **최적 전략 발견**
                +{up:.1%} 상승 → 보유주식의 {ratio:.0%} 매도
                {down:.1%} 하락 → 현금 80% 물타기
                5년 백테스트 최종 금액 **${final:,.0f}** (CAGR {best_cagr:.1%})
                """)

with tab2:
    st.write("#### 오늘 매수/매도 강도 (0~100점)")
    scores = {}
    for t in tickers:
        df = yf.download(t, period="400d", progress=False, auto_adjust=True)
        if len(df) < 50:
            scores[t] = 50
            continue
        c = df['Close']

        # 완벽하게 안전한 RSI 계산 (이제 절대 안 걸림)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-10) 
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(loss != 0, 100)   # loss가 0이면 RSI = 100
        rsi = rsi.fillna(50)
        rsi_last = rsi.iloc[-1]

        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_lower = bb_mid - 2 * bb_std

        score = 50
        if rsi_last < 30: score += 35
        if rsi_last > 70: score -= 30
        if c.iloc[-1] < bb_lower.iloc[-1]: score += 25
        if len(macd) > 1 and macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]: score += 20
        if c.iloc[-1] > c.rolling(50).mean().iloc[-1]: score += 10

        scores[t] = min(100, max(0, int(score)))

    score_df = pd.DataFrame(list(scores.items()), columns=["티커", "점수"])
    score_df = score_df.sort_values("점수", ascending=False)
    score_df["추천"] = score_df["점수"].apply(lambda x: "강력 매수🟢🟢" if x>=85 else "매수🟢" if x>=70 else "매도🔴" if x<=40 else "관망⚪")
    st.dataframe(score_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("#### 🔮 Prophet 기반 가격 예측 (내일 ~ 30일 후)")
    ticker = st.selectbox("예측할 종목 선택", tickers, key="pred_ticker")

    if st.button("🚀 예측 시작", key="run_prophet"):
        with st.spinner(f"{ticker} 5년 데이터 불러와서 예측 중... (10~20초 소요)"):
            # ----- 안전하게 데이터 다운로드 -----
            data = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            if data.empty or 'Close' not in data.columns:
                st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
                st.stop()

            df = pd.DataFrame({
                'ds': data.index,
                'y': data['Close']
            }).reset_index(drop=True)

            # ----- Prophet 모델 -----
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            m.fit(df)

            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)

            # ----- 차트 -----
            fig = go.Figure()

            # 실제 가격
            fig.add_trace(go.Scatter(
                x=df['ds'], y=df['y'],
                mode='lines',
                name='실제 가격',
                line=dict(color='#1f77b4')
            ))

            # 예측선
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'],
                mode='lines',
                name='예측 가격',
                line=dict(color='#e62e2e', width=3)
            ))

            # 80% 신뢰구간
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(100, 150, 255, 0.2)',
                name='80% 신뢰구간'
            ))

            fig.update_layout(
                height=500,
                title=f"{ticker} 가격 예측 (Prophet)",
                xaxis_title="날짜",
                yaxis_title="가격 (USD)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ----- 숫자 예측 -----
            curr_price = data['Close'].iloc[-1]

            # 내일 (마지막 실제 날짜 다음 날)
            tomorrow_pred = forecast[forecast['ds'] > df['ds'].iloc[-1]].iloc[0]['yhat']
            
            # +7일 후 (대략)
            week_pred = forecast.iloc[-24]['yhat'] if len(forecast) > 24 else forecast.iloc[-1]['yhat']
            
            # +30일 후
            month_pred = forecast.iloc[-1]['yhat']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("현재가", f"${curr_price:.2f}")
            with col2:
                st.metric("내일 예상", f"${tomorrow_pred:.2f}", f"{(tomorrow_pred/curr_price-1)*100:+.2f}%")
            with col3:
                st.metric("+7일 예상", f"${week_pred:.2f}", f"{(week_pred/curr_price-1)*100:+.2f}%")
            with col4:
                st.metric("+30일 예상", f"${month_pred:.2f}", f"{(month_pred/curr_price-1)*100:+.2f}%")

            st.success(f"{ticker} 예측 완료!")