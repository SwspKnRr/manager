import os
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def make_features_from_pv(total_pv: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """포트폴리오 평가액 시계열 → 피처 X, 라벨 y 생성"""

    total_pv = total_pv.dropna()
    returns = total_pv.pct_change().dropna()

    df = pd.DataFrame(index=returns.index)
    df["r_1"] = returns.shift(1)
    df["r_3"] = returns.rolling(3).mean().shift(1)
    df["r_5"] = returns.rolling(5).mean().shift(1)
    df["r_10"] = returns.rolling(10).mean().shift(1)

    df["vol_5"] = returns.rolling(5).std().shift(1)
    df["vol_20"] = returns.rolling(20).std().shift(1)

    ma_5 = total_pv.rolling(5).mean()
    ma_20 = total_pv.rolling(20).mean()
    df["ma_gap"] = (ma_5 - ma_20) / ma_20

    dd = (total_pv / total_pv.cummax() - 1)
    df["drawdown"] = dd

    # 라벨: 내일이 플러스인지?
    y = (returns.shift(-1) > 0).astype(int)

    # 피처/라벨에서 NaN 제거
    data = df.join(y.rename("y")).dropna()
    X = data.drop(columns=["y"])
    y = data["y"]

    return X, y

def train_direction_model(X: pd.DataFrame, y: pd.Series):
    """
    RandomForest로 방향 예측 모델 학습.
    (여기선 hyperparameter 튜닝 없이 기본값 사용)
    """
    if len(X) < 200:
        return None, None, None  # 데이터 너무 적음

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # 시계열이라 시간 순서 유지
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, (X_test.index[0], X_test.index[-1])


def predict_next_prob(model, X: pd.DataFrame):
    """
    가장 최근 row 하나를 넣어서 '내일 상승 확률' 계산.
    """
    if model is None or X.empty:
        return None
    last_x = X.iloc[[-1]]  # 마지막 행 1개
    prob = model.predict_proba(last_x)[0][1]  # 클래스 1(상승)의 확률
    return prob


# ---------------------- 기본 설정 ---------------------- #
st.set_page_config(page_title="포트폴리오 트레이딩 봇", layout="wide")
st.title("📊 포트폴리오 트레이딩 봇 (MVP 버전)")
st.markdown("---")

PORTFOLIO_FILE = "portfolio.json"


# ---------------------- 유틸 함수들 ---------------------- #
def load_portfolio() -> pd.DataFrame:
    """저장된 포트폴리오 불러오기 (없으면 빈 DF 리턴)"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            df = pd.read_json(PORTFOLIO_FILE, orient="records")
            return df
        except Exception:
            pass

    cols = ["ticker", "shares", "avg_price", "currency"]
    return pd.DataFrame(columns=cols)


def save_portfolio(df: pd.DataFrame):
    """포트폴리오 JSON 저장"""
    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=["ticker"])
    clean_df["shares"] = pd.to_numeric(clean_df["shares"], errors="coerce").fillna(0.0)
    clean_df["avg_price"] = pd.to_numeric(clean_df["avg_price"], errors="coerce").fillna(0.0)
    clean_df["currency"] = clean_df["currency"].fillna("USD")
    clean_df.to_json(PORTFOLIO_FILE, orient="records", force_ascii=False)


@st.cache_data
def fetch_price_history(tickers, start, end):
    """yfinance로 Adj Close 받아오기"""
    if len(tickers) == 0:
        return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    # 단일 티커일 때는 Series가 나오므로 DF로 변환
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data


def compute_portfolio_value(price_df: pd.DataFrame, portfolio_df: pd.DataFrame):
    """
    price_df : 날짜 x 티커
    portfolio_df : ticker, shares
    """
    # 티커 이름 정리
    tickers = [t for t in portfolio_df["ticker"].unique() if isinstance(t, str)]
    price_df = price_df[tickers]

    shares_map = portfolio_df.groupby("ticker")["shares"].sum().to_dict()
    # 각 티커별 수량 곱해서 포트폴리오 평가액 시계열 계산
    pv = price_df.copy()
    for t in tickers:
        pv[t] = pv[t] * shares_map.get(t, 0.0)

    total = pv.sum(axis=1)
    return total, pv


def simple_direction_stats(portfolio_value: pd.Series):
    """
    과거 포트폴리오 수익률 기반으로 '내일 수익률이 플러스일 확률' 같은 것 계산
    (아주 단순한 통계 버전, 예시용)
    """
    returns = portfolio_value.pct_change().dropna()
    if len(returns) < 10:
        return None

    # 수익률이 양수인 비율
    prob_up = (returns > 0).mean()
    avg_up = returns[returns > 0].mean()
    avg_down = returns[returns <= 0].mean()

    # 최근 30일 동안의 prob_up
    recent = returns.tail(30)
    prob_up_recent = (recent > 0).mean() if len(recent) > 0 else np.nan

    return {
        "prob_up_all": prob_up,
        "avg_up": avg_up,
        "avg_down": avg_down,
        "prob_up_recent": prob_up_recent,
    }


def dummy_rule_search(portfolio_value: pd.Series):
    """
    운용 규칙 최적화 부분은 나중에 진짜 백테스트 로직 넣을 거고,
    일단은 틀만 잡기 위해 간단한 예시 결과 리턴 (추측/샘플입니다)
    """
    if len(portfolio_value) < 50:
        return []

    # 예시 규칙 몇 개 가정 (실제로는 여기서 grid search 들어가야 함)
    rules = [
        {"name": "룰 A", "desc": "각 종목 +5% 시 20% 매도, -5% 시 20% 매수"},
        {"name": "룰 B", "desc": "각 종목 +10% 시 30% 매도, -7% 시 20% 매수"},
        {"name": "룰 C", "desc": "리밸런싱 없는 buy&hold"},
    ]

    # 임의로 성과 넣는 더미 (나중에 실제 백테스트로 교체)
    results = []
    for i, r in enumerate(rules):
        results.append(
            {
                "rule_name": r["name"],
                "description": r["desc"],
                "cagr": 0.10 + 0.02 * i,   # 가짜 값
                "mdd": -0.15 - 0.05 * i,   # 가짜 값
                "final_value": 1.5 + 0.3 * i,
            }
        )

    return results


# ---------------------- 사이드바: 포트폴리오 입력 ---------------------- #
st.sidebar.header("포트폴리오 설정")

if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = load_portfolio()

st.sidebar.markdown("**보유 종목/수량/평단 입력**")

edited_df = st.sidebar.data_editor(
    st.session_state["portfolio_df"],
    num_rows="dynamic",
    key="portfolio_editor",
    column_config={
        "ticker": st.column_config.TextColumn("티커 (예: AAPL, TSLA, 005930.KS)"),
        "shares": st.column_config.NumberColumn("보유 수량", step=1),
        "avg_price": st.column_config.NumberColumn("평단가"),
        "currency": st.column_config.TextColumn("통화 (USD/KRW 등)"),
    },
)

if st.sidebar.button("💾 포트폴리오 저장"):
    save_portfolio(edited_df)
    st.session_state["portfolio_df"] = edited_df
    st.sidebar.success("저장 완료! 다음 접속 때 자동으로 불러옵니다.")


# ---------------------- 메인 탭 ---------------------- #
tab1, tab2, tab3 = st.tabs(["📂 포트폴리오", "📈 수익 방향 예측", "⚙️ 운용 규칙 최적화"])

portfolio_df = st.session_state["portfolio_df"].copy()
portfolio_df = portfolio_df.dropna(subset=["ticker"])
portfolio_df["shares"] = pd.to_numeric(portfolio_df["shares"], errors="coerce").fillna(0.0)
portfolio_df = portfolio_df[portfolio_df["shares"] > 0]


# ---------------------- 탭 1: 포트폴리오 ---------------------- #
with tab1:
    st.subheader("현재 포트폴리오")

    if portfolio_df.empty:
        st.warning("포트폴리오가 비어 있습니다. 왼쪽 사이드바에서 종목을 추가하세요.")
    else:
        # 가격 불러오기 (1년치 예시)
        end = date.today()
        start = end - timedelta(days=365)
        tickers = portfolio_df["ticker"].tolist()
        price_df = fetch_price_history(tickers, start, end)

        if price_df.empty:
            st.error("가격 데이터를 가져오지 못했습니다. 티커를 확인해 보세요.")
        else:
            last_prices = price_df.ffill().iloc[-1]
            portfolio_df["last_price"] = portfolio_df["ticker"].map(last_prices.to_dict())
            portfolio_df["value"] = portfolio_df["shares"] * portfolio_df["last_price"]

            total_value = portfolio_df["value"].sum()
            portfolio_df["weight"] = portfolio_df["value"] / total_value * 100

            st.write("총 평가액 (대략):", f"{total_value:,.2f}")
            st.dataframe(portfolio_df, use_container_width=True)

            # 간단한 비중 파이차트
            st.write("종목별 비중")
            st.bar_chart(
                portfolio_df.set_index("ticker")["weight"]
            )


with tab2:
    st.subheader("포트폴리오 수익 방향 (ML + 통계)")

    if portfolio_df.empty:
        st.warning("포트폴리오가 비어 있어 수익 방향을 계산할 수 없습니다.")
    else:
        horizon_years = st.slider("과거 몇 년 데이터로 학습할지", 1, 10, 3)
        end = date.today()
        start = end - timedelta(days=365 * horizon_years)
        tickers = portfolio_df["ticker"].tolist()

        price_df = fetch_price_history(tickers, start, end)
        if price_df.empty:
            st.error("가격 데이터를 가져오지 못했습니다.")
        else:
            total_pv, pv_detail = compute_portfolio_value(price_df, portfolio_df)
            st.line_chart(total_pv, height=300)

            # 1) 통계 기반 지표 (기존 함수)
            stats = simple_direction_stats(total_pv)
            if stats is not None:
                st.markdown("### 통계 기반 분위기")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "전체 기간 기준, 다음 날 플러스일 확률 (추정)",
                        f"{stats['prob_up_all']*100:,.1f}%",
                    )
                with col2:
                    st.metric(
                        "최근 30일 기준, 다음 날 플러스일 확률 (추정)",
                        f"{stats['prob_up_recent']*100:,.1f}%",
                    )

            # 2) RandomForest 기반 방향 예측
            st.markdown("---")
            st.markdown("### ML(RandomForest) 기반 방향 예측")

            X, y = make_features_from_pv(total_pv)
            st.write(f"학습 가능한 데이터 포인트 수: {len(X)}")

            if len(X) < 200:
                st.info("데이터가 200일 미만이라 간단 통계만 사용합니다.")
            else:
                if st.button("🤖 모델 학습 및 평가"):
                    model, acc, (test_start, test_end) = train_direction_model(X, y)
                    if model is None:
                        st.error("모델 학습에 실패했습니다.")
                    else:
                        st.success(
                            f"테스트 구간({test_start.date()} ~ {test_end.date()}) "
                            f"정확도: {acc*100:,.1f}%"
                        )
                        prob_next = predict_next_prob(model, X)
                        if prob_next is not None:
                            st.metric(
                                "현재 기준 내일 상승할 확률 (모델 추정)",
                                f"{prob_next*100:,.1f}%",
                            )
                        st.caption("※ 단순 RandomForest 분류 모델이며, 과최적화/과거 데이터 편향 위험이 있습니다.")



# ---------------------- 탭 3: 운용 규칙 최적화 ---------------------- #
with tab3:
    st.subheader("간단 운용 규칙 탐색 (데모)")

    if portfolio_df.empty:
        st.warning("포트폴리오가 비어 있어 규칙 테스트를 할 수 없습니다.")
    else:
        horizon_years = st.slider("백테스트 기간 (년)", 1, 10, 5, key="rule_years")
        end = date.today()
        start = end - timedelta(days=365 * horizon_years)
        tickers = portfolio_df["ticker"].tolist()

        price_df = fetch_price_history(tickers, start, end)
        if price_df.empty:
            st.error("가격 데이터를 가져오지 못했습니다.")
        else:
            total_pv, _ = compute_portfolio_value(price_df, portfolio_df)

            if st.button("🚀 규칙 탐색 실행 (데모)"):
                results = dummy_rule_search(total_pv)
                if not results:
                    st.warning("데이터가 부족하거나 규칙 탐색 결과가 없습니다.")
                else:
                    res_df = pd.DataFrame(results)
                    st.dataframe(res_df, use_container_width=True)
                    st.info(
                        "※ 현재는 '형식만 갖춘 데모 결과'입니다. "
                        "진짜 규칙 최적화 로직은 너랑 상의해서 백테스트 넣자."
                    )
