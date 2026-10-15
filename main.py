import os
os.environ["PYTHONUTF8"] = "1"
os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(page_title="进销存预测", layout="wide")

# 标题居中样式
st.markdown("""
<style>
h1 {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# ========== session_state初始化 ==========
if "page_mode" not in st.session_state:
    st.session_state["page_mode"] = "home"

if "predict_history" not in st.session_state:
    st.session_state["predict_history"] = []

if "view_history_item" not in st.session_state:
    st.session_state["view_history_item"] = None

MAX_HISTORY = 10

def clean_spec(val):
    s = str(val)
    s = re.sub(r"[('),\"]", "", s)
    return s.strip()

def weighted_ma3_predict(series: np.ndarray, alpha: float):
    hist = np.asarray(series, dtype=float)
    if len(hist) == 0:
        return 0, np.array([0, 0, 0])
    if len(hist) >= 3:
        last3 = hist[-3:]
    elif len(hist) == 2:
        last3 = np.array([hist[0], hist[1], hist[1]])
    else:
        last3 = np.array([hist[0], hist[0], hist[0]])

    w1 = alpha
    w2 = (1 - alpha) * 0.5
    w3 = (1 - alpha) * 0.3
    w_sum = w1 + w2 + w3
    base = (w1 * last3[2] + w2 * last3[1] + w3 * last3[0]) / w_sum
    return int(max(0, round(base))), last3

def rolling_predict(history_series: np.ndarray, pred_months: int, alpha: float, w_yoy: float, w_qoq: float):
    hist = history_series.copy().tolist()
    pred_list = []
    for _ in range(pred_months):
        arr_hist = np.array(hist)
        base_pred, _ = weighted_ma3_predict(arr_hist, alpha)
        qoq_rate = 1.0
        yoy_rate = 1.0
        if len(arr_hist) >= 2:
            curr = arr_hist[-1]
            prev = arr_hist[-2]
            qoq_rate = curr / prev if prev > 1e-6 else 1.0
        if len(arr_hist) >= 13:
            yoy_rate = arr_hist[-1] / arr_hist[-13] if arr_hist[-13] > 1e-6 else 1.0
        f_yoy = 1.0 + w_yoy * (yoy_rate - 1)
        f_qoq = 1.0 + w_qoq * (qoq_rate - 1)
        this_pred = int(max(0, round(base_pred * f_yoy * f_qoq)))
        pred_list.append(this_pred)
        hist.append(this_pred)
    return pred_list

def detect_outlier_index(seq: np.ndarray, th_mult=3.0):
    idx_list = []
    n = len(seq)
    if n < 6:
        return idx_list
    mean_val = np.mean(seq[-12:]) if n >= 12 else np.mean(seq)
    for i in range(n):
        v = seq[i]
        cond = mean_val > 1e-6 and v > mean_val * th_mult
        if cond:
            prev_ok = True if i == 0 else seq[i - 1] <= mean_val * th_mult
            next_ok = True if i >= n - 1 else seq[i + 1] <= mean_val * th_mult
            if prev_ok and next_ok:
                idx_list.append(i)
    return idx_list

def replace_outlier(seq: np.ndarray, out_idx):
    arr = seq.copy()
    mask = np.ones(len(arr), dtype=bool)
    for ix in out_idx:
        mask[ix] = False
    normal_mean = np.mean(arr[mask]) if np.any(mask) else np.mean(arr)
    for ix in out_idx:
        arr[ix] = normal_mean
    return arr

def load_ship_data(file):
    df_raw = pd.read_excel(file)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df_raw["商品规格"] = df_raw["商品规格"].apply(clean_spec)
    df_raw["商品类别"] = df_raw["商品类别"].fillna("未知系列").astype(str)
    # 适配你的出货表：订单部门作为区域
    if "订单部门" in df_raw.columns:
        df_raw["区域"] = df_raw["订单部门"].fillna("未知区域").astype(str)
    else:
        df_raw["区域"] = "未知区域"
    df_raw["出货日期"] = pd.to_datetime(df_raw["出货日期"], errors="coerce")
    df_raw = df_raw.dropna(subset=["出货日期", "商品规格", "出货数量"])
    df_raw["出货数量"] = df_raw["出货数量"].clip(lower=0)
    df_raw["年月"] = df_raw["出货日期"].dt.to_period("M")
    df_raw["年月_dt"] = df_raw["年月"].dt.to_timestamp()
    df_raw["中文月份"] = df_raw["年月_dt"].dt.strftime("%Y年%m月")
    monthly = df_raw.groupby(["商品规格", "商品类别", "区域", "年月", "年月_dt", "中文月份"])["出货数量"].sum().reset_index()
    monthly.rename(columns={"出货数量": "qty"}, inplace=True)
    return monthly

def fill_full_panel(g):
    min_dt = g["年月_dt"].min()
    max_dt = g["年月_dt"].max()
    all_month = pd.date_range(start=min_dt, end=max_dt, freq="MS")
    full_df = pd.DataFrame({"年月_dt": all_month})
    full_df = pd.merge(full_df, g[["年月_dt", "qty", "中文月份"]], how="left", on="年月_dt")
    full_df["qty"] = full_df["qty"].fillna(0)
    full_df["中文月份"] = full_df["年月_dt"].dt.strftime("%Y年%m月")
    return full_df

def run_predict_logic(monthly_df, pred_months, alpha, w_yoy, w_qoq, outlier_th, do_replace_outlier, isp_df, proj_df, series_factor):
    all_month_info = monthly_df[["年月_dt", "中文月份"]].drop_duplicates().sort_values("年月_dt")
    latest_3_month = all_month_info.tail(3).reset_index(drop=True)
    m1_dt, m2_dt, m3_dt = latest_3_month["年月_dt"].iloc[0], latest_3_month["年月_dt"].iloc[1], latest_3_month["年月_dt"].iloc[2]
    m1_name, m2_name, m3_name = latest_3_month["中文月份"].iloc[0], latest_3_month["中文月份"].iloc[1], latest_3_month["中文月份"].iloc[2]
    last_year_m3_dt = m3_dt - pd.DateOffset(years=1)

    all_specs = sorted(monthly_df["商品规格"].unique())
    detail_rows = []
    outlier_log = []
    future_dt = pd.date_range(start=m3_dt + pd.DateOffset(months=1), periods=pred_months, freq="MS")
    future_col = [d.strftime("%Y年%m月预测") for d in future_dt]

    for spec in all_specs:
        g_all_region = monthly_df[monthly_df["商品规格"] == spec].copy()
        cate = g_all_region["商品类别"].iloc[0]
        full_panel_all = fill_full_panel(g_all_region.groupby(["年月_dt", "中文月份"])["qty"].sum().reset_index())
        qty_raw_total = full_panel_all["qty"].values.copy()

        q_m1 = full_panel_all[full_panel_all["年月_dt"] == m1_dt]["qty"].sum()
        q_m2 = full_panel_all[full_panel_all["年月_dt"] == m2_dt]["qty"].sum()
        q_m3 = full_panel_all[full_panel_all["年月_dt"] == m3_dt]["qty"].sum()
        q_m3_lastyear = full_panel_all[full_panel_all["年月_dt"] == last_year_m3_dt]["qty"].sum()

        out_idx = detect_outlier_index(qty_raw_total, th_mult=outlier_th)
        qty_for_model = qty_raw_total.copy()
        if len(out_idx) > 0:
            for ix in out_idx:
                outlier_log.append({
                    "商品规格": spec,
                    "商品类别": cate,
                    "异常月份": full_panel_all.iloc[ix]["中文月份"],
                    "当月出货": float(qty_raw_total[ix])
                })
            if do_replace_outlier:
                qty_for_model = replace_outlier(qty_raw_total, out_idx)

        total_pred_list = rolling_predict(qty_for_model, pred_months, alpha, w_yoy, w_qoq)
        if isp_df is not None and spec in isp_df.index:
            upper = float(isp_df.loc[spec, "预测上限"])
            lower = float(isp_df.loc[spec, "预测下限"])
            total_pred_list = [np.clip(v, lower, upper) for v in total_pred_list]
        sf = series_factor.get(cate, 1.0)
        total_pred_list = [int(max(0, round(v * sf))) for v in total_pred_list]

        # 读取项目增量，没有则0
        proj_inc = 0
        if proj_df is not None and spec in proj_df.index:
            proj_inc = int(round(float(proj_df.loc[spec])))

        # 模型预测 + 项目增量
        total_pred_list = [int(max(0, round(v + proj_inc))) for v in total_pred_list]

        if q_m2 > 1e-6:
            qoq_text = f"{(q_m3 - q_m2) / q_m2:.1%}"
        else:
            qoq_text = "‑"
        if q_m3_lastyear > 1e-6:
            yoy_text = f"{(q_m3 - q_m3_lastyear) / q_m3_lastyear:.1%}"
        else:
            yoy_text = "‑"

        row = {
            "商品规格": spec,
            "商品类别(系列)": cate,
            f"{m1_name}(真实出货)": q_m1,
            f"{m2_name}(真实出货)": q_m2,
            f"{m3_name}(真实出货)": q_m3,
            f"环比({m3_name}对比{m2_name})": qoq_text,
            f"同比({m3_name}对比{last_year_m3_dt.strftime('%Y年%m月')})": yoy_text,
            "项目增量": proj_inc,
            "存在标记突增": len(out_idx) > 0
        }
        for idx, col_name in enumerate(future_col):
            row[col_name] = total_pred_list[idx]
        detail_rows.append(row)

    df_detail = pd.DataFrame(detail_rows)
    grp_sum = df_detail.groupby("商品类别(系列)").agg({
        f"{m1_name}(真实出货)": "sum",
        f"{m2_name}(真实出货)": "sum",
        f"{m3_name}(真实出货)": "sum",
        "项目增量":"sum",
        **{c: "sum" for c in future_col}
    }).reset_index()

    df_log = pd.DataFrame(outlier_log) if len(outlier_log) > 0 else pd.DataFrame()
    meta_info = {
        "m1_name": m1_name,
        "m2_name": m2_name,
        "m3_name": m3_name,
        "future_col": future_col
    }
    return df_detail, grp_sum, df_log, meta_info


# =====================【首页】=====================
if st.session_state.page_mode == "home":
    st.title("进销存预测")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("数据预测", use_container_width=True):
            st.session_state.page_mode = "new_predict"
            st.rerun()
    with col2:
        if st.button("历史记录", use_container_width=True):
            st.session_state.page_mode = "history_page"
            st.rerun()
    st.stop()

# =====================【历史记录页面】=====================
if st.session_state.page_mode == "history_page":
    st.title("历史记录")
    if st.button("返回首页"):
        st.session_state.page_mode = "home"
        st.rerun()
    st.divider()
    hist_list = st.session_state["predict_history"]
    if len(hist_list) > 0:
        opt_text = [
            f"{i}｜{x['run_time']}｜{x['mode_name']}｜文件:{x['upload_filename']}"
            for i, x in enumerate(hist_list)
        ]
        sel_idx = st.selectbox("选择历史记录", options=range(len(hist_list)), format_func=lambda i: opt_text[i])
        sel_item = hist_list[sel_idx]
        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            if st.button("打开选中记录"):
                st.session_state["view_history_item"] = sel_item
                st.session_state.page_mode = "history_detail"
                st.rerun()
        with col_h2:
            if st.button("删除选中记录"):
                hist_list.pop(sel_idx)
                st.session_state["predict_history"] = hist_list
                st.rerun()
        if st.button("清空全部历史记录"):
            st.session_state["predict_history"] = []
            st.rerun()
    else:
        st.info("暂无历史记录")
    st.stop()

# =====================【历史记录详情页面】=====================
if st.session_state.page_mode == "history_detail":
    item = st.session_state["view_history_item"]
    st.title(f"历史结果查看｜{item['mode_name']}｜{item['run_time']}")
    if st.button("返回历史记录列表"):
        st.session_state["view_history_item"] = None
        st.session_state.page_mode = "history_page"
        st.rerun()
    df_detail = item["df_detail"]
    df_cate_summary = item["df_summary"]
    df_outlier_log = item["df_log"]
    meta = item["meta"]
    future_col = meta["future_col"]
    st.divider()
    st.subheader("1、型号明细预测")
    cate_options = ["全部系列"] + sorted(df_detail["商品类别(系列)"].unique())
    pick_cate = st.selectbox("筛选商品类别", cate_options)
    if pick_cate != "全部系列":
        df_detail_show = df_detail[df_detail["商品类别(系列)"] == pick_cate]
    else:
        df_detail_show = df_detail
    st.dataframe(df_detail_show, width="stretch", height=420)
    st.subheader("2、按产品类别汇总预测")
    st.dataframe(df_cate_summary, width="stretch")
    if not df_outlier_log.empty:
        st.subheader("3、突增异常标记日志")
        st.dataframe(df_outlier_log, width="stretch")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_detail.to_excel(writer, sheet_name="型号明细预测", index=False)
        df_cate_summary.to_excel(writer, sheet_name="按类别汇总", index=False)
        if not df_outlier_log.empty:
            df_outlier_log.to_excel(writer, sheet_name="异常标记日志", index=False)
    out_bytes = output.getvalue()
    st.download_button(
        "下载历史结果Excel",
        data=out_bytes,
        file_name=f"{item['mode_name']}_历史结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.stop()

# =====================【数据预测：选择预测类型】=====================
if st.session_state.page_mode == "new_predict":
    st.title("数据预测")
    if st.button("返回首页"):
        st.session_state.page_mode = "home"
        st.rerun()
    st.divider()
    st.markdown("请选择预测类型")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("整体预测", width="stretch"):
            st.session_state.page_mode = "total"
            st.rerun()
    with col2:
        if st.button("部门预测", width="stretch"):
            st.session_state.page_mode = "dept_select"
            st.rerun()
    st.stop()

# =====================【部门预测：选择部门】=====================
if st.session_state.page_mode == "dept_select":
    st.title("部门预测")
    if st.button("返回数据预测"):
        st.session_state.page_mode = "new_predict"
        st.rerun()
    st.divider()
    st.markdown("请选择部门")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("华东", width="stretch"):
            st.session_state.page_mode = "huadong"
            st.rerun()
    with c2:
        if st.button("华南", width="stretch"):
            st.session_state.page_mode = "huanan"
            st.rerun()
    with c3:
        if st.button("华北", width="stretch"):
            st.session_state.page_mode = "huabei"
            st.rerun()
    with c4:
        if st.button("渠道", width="stretch"):
            st.session_state.page_mode = "channel"
            st.rerun()
    st.stop()

# =====================【预测页面：整体 / 四个部门】=====================
mode_title_map = {
    "total": "整体预测",
    "huadong": "华东部门预测",
    "huanan": "华南部门预测",
    "huabei": "华北部门预测",
    "channel": "渠道部门预测"
}

current_mode = st.session_state.page_mode
st.title(mode_title_map[current_mode])

if current_mode == "total":
    back_text = "返回数据预测"
    back_target = "new_predict"
else:
    back_text = "返回部门选择"
    back_target = "dept_select"

if st.button(back_text):
    st.session_state.page_mode = back_target
    st.rerun()

# ----------------------侧边栏配置----------------------
with st.sidebar:
    st.header("参数设置")
    upload_ship = st.file_uploader(f"上传{mode_title_map[current_mode]}数据表(excel)", type=["xlsx"])

    st.divider()
    st.subheader("简易模式参数")
    pred_months = st.slider("预测未来几个月", min_value=1, max_value=3, value=3)
    alpha = st.slider("看重最近1个月程度", min_value=0.1, max_value=0.9, value=0.6, step=0.05)
    w_yoy = st.slider("同比参考强度", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    w_qoq = st.slider("环比参考强度", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    with st.expander("展开高级模式"):
        st.markdown("附加文件上传（可选）")
        upload_isp = st.file_uploader("ISP参数表(规格,预测上限,预测下限)", type=["xlsx"])
        upload_project = st.file_uploader("项目备货增量表(支持老格式/华东报备表)", type=["xlsx"])

        st.markdown("异常订单处理")
        outlier_th = st.number_input("突增倍数阈值", min_value=1.5, max_value=10.0, value=3.0, step=0.25)
        do_replace_outlier = st.checkbox("启用：将识别到的一次性突增大单替换后再训练模型", value=False)

        st.markdown("系列行情系数")
        series_text = st.text_area("格式：系列名=系数，一行一个", value="伺服‑A6=1.0\n伺服‑E7=1.0")
        series_factor = {}
        for line in series_text.splitlines():
            line = line.strip()
            if "=" in line:
                k, vstr = line.split("=")
                try:
                    series_factor[k.strip()] = float(vstr.strip())
                except Exception:
                    pass

# 加载ISP参数表
isp_df = None
if upload_isp is not None:
    isp_df = pd.read_excel(upload_isp)
    isp_df["商品规格"] = isp_df["商品规格"].apply(clean_spec)
    isp_df = isp_df.set_index("商品规格")[["预测上限", "预测下限"]]

# ========= 适配两种项目表格式：老格式(商品规格/项目增量)、华东报备(完整型号/项目需求数量) =========
proj_df = None
if upload_project is not None:
    proj_raw = pd.read_excel(upload_project)
    proj_raw.columns = [c.strip() for c in proj_raw.columns]
    # 识别型号列
    spec_col = None
    for cname in ["商品规格", "完整型号", "规格"]:
        if cname in proj_raw.columns:
            spec_col = cname
            break
    # 识别数量列
    qty_col = None
    for cname in ["项目增量", "项目需求数量", "需求数量"]:
        if cname in proj_raw.columns:
            qty_col = cname
            break
    if spec_col is not None and qty_col is not None:
        proj_raw[spec_col] = proj_raw[spec_col].apply(clean_spec)
        proj_raw[qty_col] = pd.to_numeric(proj_raw[qty_col], errors="coerce").fillna(0)
        # 同型号求和
        proj_df = proj_raw.groupby(spec_col)[qty_col].sum()

if upload_ship is not None:
    with st.spinner("正在处理出货数据..."):
        monthly_df_raw = load_ship_data(upload_ship)

    # 部门筛选，适配订单部门字段的值：华东区、华南区、华北区、渠道部
    if current_mode == "huadong":
        monthly_df = monthly_df_raw[monthly_df_raw["区域"] == "华东区"].copy()
    elif current_mode == "huanan":
        monthly_df = monthly_df_raw[monthly_df_raw["区域"] == "华南区"].copy()
    elif current_mode == "huabei":
        monthly_df = monthly_df_raw[monthly_df_raw["区域"] == "华北区"].copy()
    elif current_mode == "channel":
        monthly_df = monthly_df_raw[monthly_df_raw["区域"] == "渠道部"].copy()
    else:
        monthly_df = monthly_df_raw.copy()

    if len(monthly_df) <= 0:
        st.error("当前上传的数据中，没有找到该部门对应的出货记录，请核对Excel的【订单部门】字段！")
        st.stop()

    st.success(f"数据读取完成，本模块一共 {monthly_df['商品规格'].nunique()} 个型号")

    df_total_detail, df_cate_summary, df_outlier_log, meta = run_predict_logic(
        monthly_df, pred_months, alpha, w_yoy, w_qoq, outlier_th, do_replace_outlier,
        isp_df, proj_df, series_factor
    )

    new_hist_item = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode_name": mode_title_map[current_mode],
        "upload_filename": upload_ship.name,
        "df_detail": df_total_detail.copy(),
        "df_summary": df_cate_summary.copy(),
        "df_log": df_outlier_log.copy(),
        "meta": meta
    }
    hist_arr = st.session_state["predict_history"]
    hist_arr.append(new_hist_item)
    if len(hist_arr) > MAX_HISTORY:
        hist_arr.pop(0)
    st.session_state["predict_history"] = hist_arr

    st.divider()
    st.subheader("1、型号明细预测")
    cate_options = ["全部系列"] + sorted(df_total_detail["商品类别(系列)"].unique())
    pick_cate = st.selectbox("筛选商品类别", cate_options)
    if pick_cate != "全部系列":
        df_detail_show = df_total_detail[df_total_detail["商品类别(系列)"] == pick_cate]
    else:
        df_detail_show = df_total_detail
    st.dataframe(df_detail_show, width="stretch", height=420)

    st.subheader("2、按产品类别汇总预测")
    st.dataframe(df_cate_summary, width="stretch")

    if not df_outlier_log.empty:
        st.subheader("3、突增异常标记日志")
        st.dataframe(df_outlier_log, width="stretch")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_total_detail.to_excel(writer, sheet_name="型号明细预测", index=False)
        df_cate_summary.to_excel(writer, sheet_name="按类别汇总", index=False)
        if not df_outlier_log.empty:
            df_outlier_log.to_excel(writer, sheet_name="异常标记日志", index=False)
    out_bytes = output.getvalue()
    st.download_button(
        "下载预测结果Excel",
        data=out_bytes,
        file_name=f"{mode_title_map[current_mode]}_结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("左侧上传对应模块的出货数据表后开始预测")
