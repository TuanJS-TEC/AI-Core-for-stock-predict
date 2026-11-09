"""Demo Streamlit: chọn mã -> nhãn dự đoán T+2 và T+5 kèm xác suất -> đồ thị giá dự báo và giá thực tế.

Chạy từ thư mục gốc của project (cần kho model outputs/models/, tạo bởi `python -m src.pipeline`):
  streamlit run demo.py

Phần trên: nhãn và xác suất của nhãn cho phiên mục tiêu T = 2 và T = 5 sau phiên cuối của dữ liệu, mỗi T hai model
(L = 30 và L = 90), tính như `python -m src.predict` (không ghi file).
Phần dưới: với mỗi phiên gốc d, model dự báo giá đóng cửa tại phiên thứ T sau d (đầu hồi quy, bước cuối) từ L phiên
kết thúc tại d; đồ thị đặt dự báo tại phiên mục tiêu, cạnh giá thật và baseline repeat (giá đóng cửa của d), từ phiên
mục tiêu C.DEMO_CHART_START đến hôm nay. Dự báo có phiên mục tiêu chưa có giá thật được vẽ riêng.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C, metrics as M
from src.data import build_panel, raw_windows
from src.predict import (WEEKDAY_NAMES, build, forward, invalid_text, load_artifact, market_holidays,
                         missing_sessions, model_path, predict, session_after)
from src.report import to_price

LABEL_MARK = {"Giảm": "▼", "Đi ngang": "▶", "Tăng": "▲"}


def mtimes(paths) -> tuple:
    """Dấu thời gian sửa file: cache được làm mới khi sync dữ liệu hoặc pipeline cập nhật model."""
    return tuple(p.stat().st_mtime_ns for p in paths if p.exists())


@st.cache_resource(show_spinner="Đang đọc dữ liệu ...")
def load_panel(stamp: tuple):
    return build_panel(C.TICKERS)


@st.cache_resource(show_spinner=False)
def load_model(ticker: str, T: int, L: int, stamp: tuple):
    a = load_artifact(model_path(ticker, T, L))
    return a, build(a)


@st.cache_data(show_spinner="Đang dự đoán ...")
def latest_predictions(ticker: str, data_stamp: tuple, model_stamp: tuple, now: pd.Timestamp) -> pd.DataFrame:
    return predict([ticker], load_panel(data_stamp), now=now)


def day_name(d) -> str:
    d = pd.Timestamp(d)
    return f"{WEEKDAY_NAMES[d.dayofweek]} {d:%d/%m/%Y}"


def forecast_curve(a: dict, model, panel, holidays: np.ndarray, start: pd.Timestamp) -> pd.DataFrame:
    """Dự báo giá đóng cửa tại phiên thứ T sau mỗi phiên gốc, cho mọi phiên mục tiêu từ start trở đi."""
    t, T, L = a["ticker"], a["T"], a["L"]
    s, n = panel.log_close[t], len(panel.dates)
    ends = np.arange(max(L - 1, int(panel.dates.searchsorted(start)) - T), n)
    reg, z = forward(a, model, raw_windows(s, ends, L))
    probs, _, scores = M.apply_calibration(z, a["temperature"], np.asarray(a["prior"]), a["alpha"], a["shrink"])
    labels = [a["label_names"][k] for k in scores.argmax(1)]
    has_true = ends + T < n
    table = pd.DataFrame({
        "base_date": panel.dates[ends],
        "target_date": [panel.dates[e + T] if ok else session_after(panel.dates[e], T, holidays)
                        for e, ok in zip(ends, has_true)],
        "pred": to_price(a, reg[:, -1]),
        "repeat": np.exp(s[ends]),
        "actual": np.where(has_true, np.exp(s[np.minimum(ends + T, n - 1)]), np.nan),
        "label": labels,
        "p_label": probs[np.arange(len(ends)), scores.argmax(1)],
        **dict(zip(C.PROB_COLS, probs.T)),
    })
    return table[table["target_date"] >= start].reset_index(drop=True)


def label_card(row: pd.Series) -> None:
    problem = invalid_text(row)
    if problem:                                                     # dữ liệu không hợp lệ: không có nhãn/xác suất
        with st.container(border=True):
            st.markdown(f"**L = {row['L']} phiên** · {row['model']}")
            grey = C.DEMO_LABEL_COLORS["Đi ngang"]
            st.markdown(f"<div style='font-size:1.4rem;font-weight:700;color:{grey}'>{problem}</div>",
                        unsafe_allow_html=True)
            st.caption("Dữ liệu thiếu phiên hoặc có ngày chưa giải thích được; chạy `python sync_stock_parquet.py` "
                       "hoặc khai báo ngày nghỉ rồi tải lại trang.")
        return
    label = row["label"]
    p = row[C.PROB_COLS[C.LABEL_NAMES.index(label)]]
    color = C.DEMO_LABEL_COLORS[label]
    with st.container(border=True):
        st.markdown(f"**L = {row['L']} phiên** · {row['model']}")
        st.markdown(f"<div style='font-size:1.9rem;font-weight:700;color:{color}'>{LABEL_MARK[label]} {label}"
                    f"<span style='font-size:1.1rem;font-weight:500'> &nbsp;xác suất {p:.1%}</span></div>",
                    unsafe_allow_html=True)
        st.caption(" · ".join(f"{name} {row[col]:.1%}" for name, col in zip(C.LABEL_NAMES, C.PROB_COLS)))
        st.caption(f"Tăng nếu giá thay đổi > {row['up_above_pct']:+.2f} %, Giảm nếu < {row['down_below_pct']:+.2f} %"
                   f" · Macro-F1 val {row['val_f1_macro']:.3f}")


def chart(curve: pd.DataFrame, panel, ticker: str, start: pd.Timestamp, end: pd.Timestamp, T: int,
          show_repeat: bool) -> go.Figure:
    dates = panel.dates[(panel.dates >= start) & (panel.dates <= end)]
    actual = np.exp(panel.log_close[ticker][panel.dates.isin(dates)])
    hover = ("Phiên mục tiêu %{x|%d/%m/%Y}<br>Giá dự báo %{y:.2f}<br>Phiên gốc %{customdata[0]|%d/%m/%Y}"
             "<br>Nhãn %{customdata[1]} (%{customdata[2]:.1%})<extra></extra>")
    fig = go.Figure()
    fig.add_scatter(x=dates, y=actual, name="Giá thực tế", mode="lines+markers",
                    line=dict(color=C.DEMO_ACTUAL_COLOR, width=2), marker=dict(size=5),
                    hovertemplate="%{x|%d/%m/%Y}<br>Giá thực tế %{y:.2f}<extra></extra>")
    done, ahead = curve[curve["actual"].notna()], curve[curve["actual"].isna()]
    for part, name, dash, symbol in ((done, f"Giá dự báo T+{T}", "dash", "circle"),
                                     (ahead, f"Dự báo T+{T} (chưa có giá thật)", "dot", "circle-open")):
        if part.empty:
            continue
        fig.add_scatter(x=part["target_date"], y=part["pred"], name=name, mode="lines+markers",
                        line=dict(color=C.DEMO_PRED_COLOR, width=2, dash=dash),
                        marker=dict(size=8, symbol=symbol, color=C.DEMO_PRED_COLOR,
                                    line=dict(width=2, color=[C.DEMO_LABEL_COLORS[x] for x in part["label"]])),
                        customdata=part[["base_date", "label", "p_label"]], hovertemplate=hover)
    if show_repeat:
        fig.add_scatter(x=curve["target_date"], y=curve["repeat"], name="Baseline repeat", mode="lines",
                        line=dict(color=C.DEMO_REPEAT_COLOR, width=1.5, shape="hv"),
                        customdata=curve[["base_date"]],
                        hovertemplate="Phiên mục tiêu %{x|%d/%m/%Y}<br>Repeat %{y:.2f} (giá phiên gốc "
                                      "%{customdata[0]|%d/%m/%Y})<extra></extra>")
    fig.add_vline(x=panel.dates[-1], line=dict(color=C.DEMO_REPEAT_COLOR, width=1, dash="dot"))
    fig.update_layout(height=460, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title="Giá đóng cửa (thang giá gốc)",
                      xaxis=dict(range=[start - pd.Timedelta(days=1), max(end, curve["target_date"].max())
                                        + pd.Timedelta(days=1)],
                                 rangebreaks=[dict(bounds=["sat", "mon"])]))
    return fig


def error_table(curve: pd.DataFrame) -> pd.DataFrame:
    done = curve[curve["actual"].notna()]
    return pd.DataFrame([{"Phương pháp": name, "Số phiên": len(done),
                          "MAE (thang giá gốc)": float(np.abs(done[col] - done["actual"]).mean()),
                          "MSE (thang giá gốc)": float(((done[col] - done["actual"]) ** 2).mean())}
                         for name, col in (("Model (đầu hồi quy)", "pred"), ("Baseline repeat", "repeat"))])


def main() -> None:
    st.set_page_config(page_title="Dự đoán xu hướng cổ phiếu", page_icon="📈", layout="wide")
    st.title("Dự đoán xu hướng giá cổ phiếu")
    ticker = st.selectbox("Chọn mã cổ phiếu", C.TICKERS, index=None, placeholder="Chọn một mã ...")
    if ticker is None:
        st.info(f"Chọn một trong {len(C.TICKERS)} mã để xem nhãn dự đoán và đồ thị giá dự báo.")
        st.stop()

    data_stamp = mtimes(sorted(C.STOCK_DIR.glob("*_stock.parquet")))
    model_files = [model_path(ticker, T, L) for T in C.HORIZONS for L in C.LOOKBACKS]
    if not all(p.exists() for p in model_files):
        st.error("Chưa có đủ model cho mã này trong outputs/models/; chạy `python -m src.pipeline` trước.")
        st.stop()
    model_stamp = mtimes(model_files)
    panel = load_panel(data_stamp)
    holidays = market_holidays(panel)
    now = pd.Timestamp.now().floor("min")
    last = panel.dates[-1]

    # ---- Phần trên: nhãn dự đoán kèm xác suất cho T+2 và T+5 ----
    st.header(f"{ticker}: nhãn dự đoán")
    lag = missing_sessions(last, now, holidays)
    if lag:
        st.warning(f"Dữ liệu mới đến phiên {day_name(last)}, thiếu {len(lag)} phiên lẽ ra đã có "
                   f"({', '.join(f'{d:%d/%m}' for d in lag)}). Chạy `python sync_stock_parquet.py` rồi tải lại trang.")
    st.caption(f"Dự đoán lúc {now:%H:%M %d/%m/%Y} từ phiên cuối của dữ liệu {day_name(last)}, giá đóng cửa "
               f"{np.exp(panel.log_close[ticker][-1]):.2f}. Nhãn so giá đóng cửa tại phiên mục tiêu với phiên này; "
               "xác suất đã hiệu chỉnh trên val. Nhãn chọn theo argmax(p / prior^alpha) như khi đánh giá, nên có thể "
               "khác lớp có xác suất lớn nhất.")
    table = latest_predictions(ticker, data_stamp, model_stamp, now)
    for col, (T, rows) in zip(st.columns(len(C.HORIZONS)), table.groupby("T")):
        with col:
            first = rows.iloc[0]
            note = {"đã qua": " (đã qua, dữ liệu chưa cập nhật)",
                    "dự kiến": " (dự kiến, lịch nghỉ chưa khai báo)"}.get(first["target_note"], "")
            st.subheader(f"T+{T}: {day_name(first['target_date'])}{note}")
            for _, row in rows.sort_values("L").iterrows():
                label_card(row)

    # ---- Phần dưới: đồ thị giá dự báo và giá thực tế ----
    st.header(f"{ticker}: giá dự báo và giá thực tế")
    c1, c2, c3 = st.columns([1, 1, 2])
    T = c1.radio("Tầm dự báo", C.HORIZONS, format_func=lambda x: f"T+{x}", horizontal=True)
    L = c2.radio("Độ dài cửa sổ", C.LOOKBACKS, format_func=lambda x: f"L = {x}", horizontal=True)
    show_repeat = c3.checkbox("Hiện baseline repeat (giá đóng cửa của phiên gốc)", value=True)
    start, end = pd.Timestamp(C.DEMO_CHART_START), now.normalize()
    a, model = load_model(ticker, T, L, mtimes([model_path(ticker, T, L)]))
    curve = forecast_curve(a, model, panel, holidays, start)
    st.plotly_chart(chart(curve, panel, ticker, start, end, T, show_repeat), width="stretch")
    st.caption(f"Model {a['model']} (wd = {a['wd']}, seed = {a['seed']}). Mỗi điểm dự báo đặt tại phiên mục tiêu, "
               f"tính từ {L} phiên kết thúc tại phiên gốc T = {T} phiên trước đó; viền điểm có màu của nhãn dự đoán "
               "(đỏ Giảm, xám Đi ngang, xanh Tăng). Đường chấm dọc: phiên cuối của dữ liệu.")
    if curve["actual"].notna().any():
        st.dataframe(error_table(curve), hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.4f")
                                    for c in ("MAE (thang giá gốc)", "MSE (thang giá gốc)")})
    with st.expander("Bảng chi tiết"):
        show = curve.assign(base_date=curve["base_date"].dt.date, target_date=curve["target_date"].dt.date)
        st.dataframe(show.rename(columns={"base_date": "Phiên gốc", "target_date": "Phiên mục tiêu",
                                          "pred": "Giá dự báo", "repeat": "Repeat", "actual": "Giá thực tế",
                                          "label": "Nhãn", "p_label": "Xác suất nhãn"}),
                     hide_index=True)


main()
