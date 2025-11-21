import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import math
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 페이지 설정 및 초기화
# ---------------------------------------------------------
st.set_page_config(page_title="Quant Portfolio", layout="wide")

if 'search_ticker' not in st.session_state:
    st.session_state['search_ticker'] = 'TQQQ'

def init_db():
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    
    # 테이블 생성
    c.execute('''CREATE TABLE IF NOT EXISTS holdings
                 (ticker TEXT PRIMARY KEY, shares INTEGER, avg_price REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS cash
                 (currency TEXT PRIMARY KEY, amount REAL)''')
    
    # [업데이트] 순서 변경을 위한 sort_order 컬럼 추가 (기존 DB 마이그레이션)
    try:
        c.execute("SELECT sort_order FROM holdings LIMIT 1")
    except sqlite3.OperationalError:
        # 컬럼이 없으면 추가
        c.execute("ALTER TABLE holdings ADD COLUMN sort_order INTEGER DEFAULT 99")
        
    conn.commit()
    conn.close()

def get_portfolio():
    conn = sqlite3.connect('portfolio.db')
    try:
        # [업데이트] sort_order 기준으로 정렬해서 가져오기
        df_holdings = pd.read_sql("SELECT * FROM holdings ORDER BY sort_order ASC, ticker ASC", conn)
        df_cash = pd.read_sql("SELECT * FROM cash", conn)
    except:
        df_holdings = pd.DataFrame()
        df_cash = pd.DataFrame()
    conn.close()
    return df_holdings, df_cash

def update_holding(ticker, shares, avg_price):
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    if shares == 0:
        c.execute("DELETE FROM holdings WHERE ticker=?", (ticker,))
    else:
        # 기존 sort_order 유지하면서 업데이트 (없으면 99)
        c.execute("SELECT sort_order FROM holdings WHERE ticker=?", (ticker,))
        res = c.fetchone()
        order = res[0] if res else 99
        c.execute("INSERT OR REPLACE INTO holdings VALUES (?, ?, ?, ?)", (ticker, shares, avg_price, order))
    conn.commit()
    conn.close()

def update_cash(amount):
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO cash VALUES (?, ?)", ('USD', amount))
    conn.commit()
    conn.close()

# [신규 기능] 순서 일괄 업데이트
def update_sort_orders(df_edited):
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    for index, row in df_edited.iterrows():
        c.execute("UPDATE holdings SET sort_order=? WHERE ticker=?", (row['sort_order'], row['ticker']))
    conn.commit()
    conn.close()

def set_ticker(ticker):
    st.session_state['search_ticker'] = ticker

init_db()

# ---------------------------------------------------------
# 2. 핵심 로직 함수들
# ---------------------------------------------------------

def run_backtest(df, initial_cash, mode, target_weight, trigger_up, sell_pct, trigger_down, buy_pct):
    cash = initial_cash
    start_price = df.iloc[0]['Close']
    
    initial_invest = (initial_cash * (target_weight / 100))
    shares = math.floor(initial_invest / start_price)
    cash -= shares * start_price
    
    last_rebal_price = start_price 
    
    history = [] 
    trade_log = []

    for date, row in df.iterrows():
        price = row['Close']
        stock_val = shares * price
        total_val = cash + stock_val
        current_weight = (stock_val / total_val * 100) if total_val > 0 else 0
        
        action_taken = False 
        
        should_sell = False
        if mode == 'VALUE': 
            if shares > 0 and price >= last_rebal_price * (1 + trigger_up/100):
                should_sell = True
        elif mode == 'WEIGHT': 
            if current_weight >= target_weight + trigger_up:
                should_sell = True
                
        if should_sell:
            sell_qty = math.floor(shares * (sell_pct / 100))
            if sell_qty > 0:
                shares -= sell_qty
                cash += sell_qty * price
                pct_diff = (price - last_rebal_price)/last_rebal_price*100
                trade_log.append({
                    "date": date, "type": "🔴 매도", "price": price, "qty": sell_qty, 
                    "cause": f"{'상승' if mode=='VALUE' else '비중초과'} (+{pct_diff:.1f}% / {current_weight:.1f}%)"
                })
                last_rebal_price = price 
                action_taken = True

        if not action_taken:
            should_buy = False
            if mode == 'VALUE':
                if price <= last_rebal_price * (1 - trigger_down/100) or (shares == 0 and cash > price):
                    should_buy = True
            elif mode == 'WEIGHT':
                if current_weight <= target_weight - trigger_down:
                    should_buy = True
            
            if should_buy:
                invest_amt = cash * (buy_pct / 100)
                buy_qty = math.floor(invest_amt / price)
                if buy_qty > 0:
                    shares += buy_qty
                    cash -= buy_qty * price
                    pct_diff = (price - last_rebal_price)/last_rebal_price*100
                    trade_log.append({
                        "date": date, "type": "🔵 매수", "price": price, "qty": buy_qty, 
                        "cause": f"{'하락' if mode=='VALUE' else '비중미달'} ({pct_diff:.1f}% / {current_weight:.1f}%)"
                    })
                    last_rebal_price = price

        total_asset = cash + (shares * price)
        history.append(total_asset)

    df['Strategy_Asset'] = history
    final_return = ((history[-1] - initial_cash) / initial_cash) * 100
    buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
    
    return df, trade_log, final_return, buy_hold_return

def optimize_params(df, fixed_b, fixed_d, target_w):
    if len(df) < 10:
        st.toast("❌ 데이터가 부족합니다.")
        return

    best_ret = -9999
    best_params = (st.session_state.get('mode', 'VALUE'), 
                   st.session_state.get('up_a', 10.0), 
                   st.session_state.get('down_c', 10.0))
    
    modes = ['VALUE', 'WEIGHT']
    search_ranges = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
    
    st.toast("🤖 모든 시나리오를 시뮬레이션 중입니다...")
    
    for m in modes:
        for a_val in search_ranges:
            for c_val in search_ranges:
                _, _, ret, _ = run_backtest(
                    df.copy(), 10000, 
                    m, target_w,
                    a_val, fixed_b, 
                    c_val, fixed_d
                )
                if ret > best_ret:
                    best_ret = ret
                    best_params = (m, a_val, c_val)
    
    st.session_state['mode'] = best_params[0]
    st.session_state['up_a'] = best_params[1]
    st.session_state['down_c'] = best_params[2]
    
    mode_kor = "평가액 변동" if best_params[0] == 'VALUE' else "비중 변동"
    st.toast(f"✅ 최적 전략: [{mode_kor}] +{best_params[1]}% / -{best_params[2]}%")


# ---------------------------------------------------------
# 3. UI 레이아웃
# ---------------------------------------------------------

col_main, col_side = st.columns([3, 1])

# --- [우측 패널] 내 투자 현황 ---
with col_side:
    st.subheader("내 투자")
    my_stocks, my_cash = get_portfolio()
    
    if my_cash.empty:
        current_cash = 0.0
    else:
        current_cash = my_cash.iloc[0]['amount']

    total_value = current_cash
    daily_pnl = 0.0
    
    # 보유 종목 목록
    if not my_stocks.empty:
        for index, row in my_stocks.iterrows():
            ticker = row['ticker']
            shares = row['shares']
            try:
                stock_data = yf.Ticker(ticker).history(period="5d")
                if len(stock_data) >= 2:
                    cur_price = stock_data['Close'].iloc[-1]
                    prev_close = stock_data['Close'].iloc[-2]
                    
                    val = cur_price * shares
                    total_value += val
                    daily_pnl += (cur_price - prev_close) * shares
                    
                    with st.container(border=True):
                        c1, c2 = st.columns([1.2, 1])
                        if c1.button(f"{ticker}", key=f"btn_{ticker}", use_container_width=True, on_click=set_ticker, args=(ticker,)):
                            pass
                        c1.caption(f"{shares}주")
                        
                        profit_pct = (cur_price - row['avg_price']) / row['avg_price'] * 100
                        color = "red" if profit_pct > 0 else "blue"
                        c2.markdown(f"${val:,.0f}")
                        c2.markdown(f":{color}[{profit_pct:.1f}%]")
            except:
                pass

    st.metric(label="총 자산 (USD)", value=f"${total_value:,.2f}", delta=f"${daily_pnl:,.2f} (오늘)")
    
    if st.button("📈 자산 추이 (Simulation)", use_container_width=True):
        if not my_stocks.empty:
            with st.spinner("계산 중..."):
                tickers = my_stocks['ticker'].tolist()
                data = yf.download(tickers, period="1y")['Close']
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers[0])
                
                portfolio_hist = pd.Series(current_cash, index=data.index)
                for index, row in my_stocks.iterrows():
                    if row['ticker'] in data.columns:
                        portfolio_hist += data[row['ticker']] * row['shares']
                
                with st.expander("내 포트폴리오 가치 변화 (1년)", expanded=True):
                    fig_total = go.Figure()
                    fig_total.add_trace(go.Scatter(x=portfolio_hist.index, y=portfolio_hist, fill='tozeroy', line=dict(color='#8b5cf6')))
                    fig_total.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                    st.plotly_chart(fig_total, use_container_width=True)

    st.divider()
    
    # 탭: 현금 / 주식 / 순서변경
    tab_edit1, tab_edit2, tab_edit3 = st.tabs(["💵 현금", "✏️ 주식", "≡ 순서"])
    
    with tab_edit1:
        new_cash = st.number_input("보유 현금 ($)", value=float(current_cash), step=100.0)
        if st.button("현금 업데이트"):
            update_cash(new_cash)
            st.rerun()
            
    with tab_edit2:
        input_ticker = st.text_input("티커").upper()
        input_shares = st.number_input("수량", min_value=0, step=1)
        input_avg = st.number_input("평단가 ($)", min_value=0.0)
        if st.button("주식 저장"):
            update_holding(input_ticker, input_shares, input_avg)
            st.rerun()
    
    # [신규 기능] 종목 순서 변경
    with tab_edit3:
        if not my_stocks.empty:
            st.caption("숫자가 작을수록 위로 올라갑니다.")
            # 데이터 프레임 에디터로 순서 편집
            edited_df = st.data_editor(
                my_stocks[['ticker', 'sort_order']], 
                column_config={
                    "ticker": st.column_config.TextColumn("종목", disabled=True),
                    "sort_order": st.column_config.NumberColumn("순서", min_value=1, step=1)
                },
                hide_index=True,
                use_container_width=True
            )
            if st.button("순서 저장"):
                update_sort_orders(edited_df)
                st.rerun()
        else:
            st.info("보유 종목이 없습니다.")

# --- [좌측 패널] 차트 및 분석 ---
with col_main:
    c_search, c_int, c_refresh = st.columns([2, 1, 0.5])
    with c_search:
        search_ticker = st.text_input("종목 검색", key='search_ticker').upper()
    
    with c_int:
        interval_map = {'1m': '1분', '5m': '5분', '1d': '일봉', '1wk': '주봉', '1mo': '월봉'}
        sel_interval = st.selectbox("주기", options=list(interval_map.keys()), format_func=lambda x: interval_map[x], index=2)
    with c_refresh:
        st.write("") 
        st.write("")
        if st.button("🔄"):
            st.rerun()

    stock = yf.Ticker(search_ticker)
    period_map = {'1m': '5d', '5m': '1mo', '1d': '2y', '1wk': '5y', '1mo': '10y'}
    
    try:
        hist_chart = stock.history(period=period_map[sel_interval], interval=sel_interval)
    except:
        hist_chart = pd.DataFrame()
    
    if hist_chart.empty:
        st.error(f"'{search_ticker}' 데이터를 불러올 수 없습니다.")
    else:
        last_price = hist_chart['Close'].iloc[-1]
        prev_price = hist_chart['Close'].iloc[-2]
        change = last_price - prev_price
        pct_change = (change / prev_price) * 100
        
        st.markdown(f"## {search_ticker} ${last_price:.2f} <span style='color:{'red' if change>0 else 'blue'}'>({pct_change:.2f}%)</span>", unsafe_allow_html=True)

        # [신규 기능] 차트 구간 선택기 (분봉일 때만 표시)
        zoom_range = None
        if sel_interval in ['1m', '5m']:
            # 가로 탭 형태로 표시
            range_cols = st.columns([1, 1, 1, 1, 6]) # 버튼 4개 + 여백
            with range_cols[0]: 
                if st.button("1H", use_container_width=True): zoom_range = 1
            with range_cols[1]: 
                if st.button("2H", use_container_width=True): zoom_range = 2
            with range_cols[2]: 
                if st.button("4H", use_container_width=True): zoom_range = 4
            with range_cols[3]: 
                if st.button("ALL", use_container_width=True): zoom_range = 0 # 전체
            
            # Session State에 줌 상태 저장 (버튼 클릭 유지 효과)
            if 'chart_zoom' not in st.session_state: st.session_state['chart_zoom'] = 4 # 기본 4시간
            
            if zoom_range is not None:
                st.session_state['chart_zoom'] = zoom_range
            
            current_zoom = st.session_state['chart_zoom']

        # 차트 생성
        fig = go.Figure(data=[go.Candlestick(x=hist_chart.index,
                    open=hist_chart['Open'], high=hist_chart['High'],
                    low=hist_chart['Low'], close=hist_chart['Close'])])
        
        # [기능 구현] x축 범위(Range) 동적 설정
        if sel_interval in ['1m', '5m'] and st.session_state.get('chart_zoom', 0) > 0:
            end_time = hist_chart.index[-1]
            start_time = end_time - timedelta(hours=st.session_state['chart_zoom'])
            fig.update_xaxes(range=[start_time, end_time])
        
        # RangeSlider 제거 및 여백 최소화
        fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["🔄 전략 시뮬레이터", "📢 매매 신호", "📈 추세 예측"])
        
        # === Tab 1: 리밸런싱 ===
        with tab1:
            st.markdown("### 🛠️ 과거 데이터 검증 (Backtest)")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input("시작일", value=datetime.now() - timedelta(days=365))
            with col_d2:
                end_date = st.date_input("종료일", value=datetime.now())
            
            hist_back = stock.history(start=start_date, end=end_date, interval="1d")
            st.divider()
            
            col_inputs, col_results = st.columns([1, 2])
            
            with col_inputs:
                st.markdown("#### ⚙️ 규칙 설정")
                
                if 'mode' not in st.session_state: st.session_state['mode'] = 'VALUE'
                if 'target_w' not in st.session_state: st.session_state['target_w'] = 50
                if 'up_a' not in st.session_state: st.session_state['up_a'] = 10.0
                if 'sell_b' not in st.session_state: st.session_state['sell_b'] = 50
                if 'down_c' not in st.session_state: st.session_state['down_c'] = 10.0
                if 'buy_d' not in st.session_state: st.session_state['buy_d'] = 50

                with st.container(border=True):
                    mode_options = {'VALUE': '📊 평가액 변동 기준', 'WEIGHT': '⚖️ 비중 변동 기준'}
                    selected_mode = st.radio("매매 기준", options=list(mode_options.keys()), format_func=lambda x: mode_options[x], key='mode')
                    
                    if selected_mode == 'WEIGHT':
                        st.slider("목표 주식 비중 (%)", 10, 90, key='target_w', step=10)
                    
                    lbl_up = "A: 상승폭 (+%)" if selected_mode == 'VALUE' else "A: 비중 초과 (+%p)"
                    lbl_down = "C: 하락폭 (-%)" if selected_mode == 'VALUE' else "C: 비중 미달 (-%p)"

                    st.markdown("**매도(Sell)**")
                    in_up_A = st.slider(lbl_up, 1.0, 30.0, key='up_a', step=0.5)
                    in_sell_B = st.slider("B: 매도량 (%)", 10, 100, key='sell_b', step=10)
                    
                    st.markdown("**매수(Buy)**")
                    in_down_C = st.slider(lbl_down, 1.0, 30.0, key='down_c', step=0.5)
                    in_buy_D = st.slider("D: 매수량 (%)", 10, 100, key='buy_d', step=10)

                st.button("✨ 최적 파라미터 찾기", on_click=optimize_params, args=(hist_back, in_sell_B, in_buy_D, st.session_state['target_w']))
                
            with col_results:
                if len(hist_back) > 0:
                    df_res, logs, final_ret, bh_ret = run_backtest(
                        hist_back.copy(), 10000, 
                        st.session_state['mode'], st.session_state['target_w'],
                        in_up_A, in_sell_B, in_down_C, in_buy_D
                    )
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("전략 수익률", f"{final_ret:.2f}%", delta=f"{final_ret - bh_ret:.2f}%p")
                    m2.metric("단순 보유", f"{bh_ret:.2f}%")
                    m3.metric("매매 횟수", f"{len(logs)}회")
                    
                    fig_back = go.Figure()
                    fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Strategy_Asset'], mode='lines', name='전략', line=dict(color='#ef4444', width=2)))
                    norm_factor = 10000 / df_res['Close'].iloc[0]
                    fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Close']*norm_factor, mode='lines', name='보유', line=dict(color='#e5e7eb', dash='dot')))
                    
                    buy_pts = df_res.loc[[x['date'] for x in logs if '매수' in x['type']]]
                    sell_pts = df_res.loc[[x['date'] for x in logs if '매도' in x['type']]]
                    
                    fig_back.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts['Strategy_Asset'], mode='markers', name='매수', marker=dict(color='blue', symbol='triangle-up', size=8)))
                    fig_back.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts['Strategy_Asset'], mode='markers', name='매도', marker=dict(color='red', symbol='triangle-down', size=8)))

                    fig_back.update_layout(title="자산 추이", margin=dict(t=30, b=0, l=0, r=0), legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
                    st.plotly_chart(fig_back, use_container_width=True)
                    
                    with st.expander("상세 기록"):
                        if logs: st.dataframe(pd.DataFrame(logs), use_container_width=True)

        # === Tab 2: 지표 ===
        with tab2:
            st.write("### 투자 심리 & 지표")
            if len(hist_chart) > 15:
                delta = hist_chart['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                col_i1, col_i2 = st.columns(2)
                col_i1.metric("RSI (14)", f"{rsi:.2f}")
                msg = "🟢 과매도 (매수 기회?)" if rsi < 30 else "🔴 과매수 (과열 주의)" if rsi > 70 else "⚪ 중립"
                col_i1.info(msg)
            else:
                st.warning("데이터가 부족하여 지표를 계산할 수 없습니다.")

        # === Tab 3: 예측 ===
        with tab3:
            st.write("### 통계적 변동성 예측")
            daily_vol = hist_chart['Close'].pct_change().std()
            st.info(f"내일 예상 변동폭: ±${last_price * daily_vol:.2f} ({daily_vol*100:.2f}%)")