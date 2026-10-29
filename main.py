import os
os.environ["PYTHONUTF8"] = "1"
os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
import warnings
from datetime import datetime, timedelta
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

# ===================== 解决pyarrow OverflowError清洗函数 =====================
def clean_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    num_cols = df_out.select_dtypes(include=["number"]).columns
    for c in num_cols:
        df_out[c] = df_out[c].astype(float)
    return df_out

# ===================== ISP上限逻辑【新版：异常型号踢出额度池，正常型号总额封顶】 =====================
def apply_isp_series_cap(df_detail: pd.DataFrame, isp_raw: pd.DataFrame, future_col_list):
    df = df_detail.copy()
    isp_raw.index = [str(idx).strip() for idx in isp_raw.index]
    isp_raw.columns = [str(c).strip() for c in isp_raw.columns]

    isp_month_map = dict()
    for col in isp_raw.columns:
        col_origin = col
        try:
            num = float(col)
            dt = datetime(1899, 12, 30) + timedelta(days=num)
            isp_month_map[(dt.year, dt.month)] = col_origin
        except (ValueError, TypeError):
            m = re.search(r"(\d{4})年(\d{1,2})月", col)
            if m:
                y = int(m.group(1))
                mth = int(m.group(2))
                isp_month_map[(y, mth)] = col_origin

    for pred_col in future_col_list:
        m = re.search(r"(\d{4})年(\d{1,2})月预测", pred_col)
        if not m:
            continue
        pred_year = int(m.group(1))
        pred_mth = int(m.group(2))
        key = (pred_year, pred_mth)
        if key not in isp_month_map:
            continue
        isp_col_name = isp_month_map[key]

        # 按系列分组处理
        for series_name, group_df in df.groupby("商品类别(系列)"):
            if series_name not in isp_raw.index:
                continue
            try:
                cap_total = float(isp_raw.loc[series_name, isp_col_name])
            except Exception:
                continue
            if cap_total <= 1e-6:
                continue

            # 拆分：异常突增型号 / 正常型号
            abnormal_mask = group_df["存在标记突增"] == True
            normal_mask = group_df["存在标记突增"] == False

            normal_df = group_df[normal_mask]
            if len(normal_df) == 0:
                continue

            sum_normal = normal_df[pred_col].sum()
            if sum_normal <= cap_total + 1e-6:
                # 正常型号总和没超上限，不用压缩
                continue

            # 需要等比例压缩正常型号
            ratio = cap_total / sum_normal
            for idx in normal_df.index:
                old_val = df.at[idx, pred_col]
                new_val = old_val * ratio
                df.at[idx, pred_col] = int(round(new_val))
    return df

# ========== session_state初始化 ==========
if "page_mode" not in st.session_state:
    st.session_state["page_mode"] = "home"

if "predict_history" not in st.session_state:
    st.session_state["predict_history"] = []

if "view_history_item" not in st.session_state:
    st.session_state["view_history_item"] = None

if "isp_config" not in st.session_state:
    st.session_state["isp_config"] = None

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
    w2 = (1.0 - alpha) / 2.0
    w3 = (1.0 - alpha) / 2.0
    base = last3[2] * w1 + last3[1] * w2 + last3[0] * w3
    return int(max(0, round(base))), last3

# ===================== 滚动预测函数【增加地板保护层 + 单型号环比暴涨保护】 =====================
def rolling_predict(history_series: np.ndarray, pred_months: int, alpha: float, w_yoy: float, w_qoq: float, max_gain_rate:float):
    hist = history_series.copy().tolist()
    pred_list = []

    # 保护层：取真实历史最后3个月最小值 ×0.2作为地板
    real_last_3 = np.asarray(history_series[-3:], dtype=float)
    floor_val = float(np.min(real_last_3)) * 0.2

    for _ in range(pred_months):
        arr_hist = np.array(hist, dtype=float)
        base_pred, _ = weighted_ma3_predict(arr_hist, alpha)

        qoq_rate = 1.0
        yoy_rate = 1.0
        if len(arr_hist) >= 2:
            curr = arr_hist[-1]
            prev = arr_hist[-2]
            qoq_rate = curr / prev if prev > 1e-6 else 1.0
        if len(arr_hist) >= 13:
            curr = arr_hist[-1]
            prev_year = arr_hist[-13]
            yoy_rate = curr / prev_year if prev_year > 1e-6 else 1.0

        f_yoy = 1.0 + w_yoy * (yoy_rate - 1.0)
        f_qoq = 1.0 + w_qoq * (qoq_rate - 1.0)

        raw_result = base_pred * f_yoy * f_qoq

        # ===== 新增：暴涨限制，不能超过上一期 * max_gain_rate =====
        prev_pred_val = arr_hist[-1]
        if prev_pred_val > 1e-6:
            raw_result = min(raw_result, prev_pred_val * max_gain_rate)

        # 地板保护，不能低于floor_val，不能小于0
        protected_result = max(raw_result, floor_val, 0.0)
        this_pred = int(round(protected_result))

        pred_list.append(this_pred)
        hist.append(this_pred)  # 继续把预测值加入窗口，维持你原本滚动逻辑
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
            prev_ok = True if i == 0 else seq[i-1] <= mean_val * th_mult
            next_ok = True if i >= n-1 else seq[i+1] <= mean_val * th_mult
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
    df_raw = pd.read_excel(file, engine="openpyxl")
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    df_raw["商品规格"] = df_raw["商品规格"].apply(clean_spec)
    df_raw["商品类别"] = df_raw["商品类别"].fillna("未知系列").astype(str)
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
    monthly.rename(columns={"出货数量":"qty"}, inplace=True)
    return monthly

def fill_full_panel(g):
    min_dt = g["年月_dt"].min()
    max_dt = g["年月_dt"].max()
    all_month = pd.date_range(start=min_dt, end=max_dt, freq="MS")
    full_df = pd.DataFrame({"年月_dt": all_month})
    full_df = pd.merge(full_df, g[["年月_dt","qty","中文月份"]], how="left", on="年月_dt")
    full_df["qty"] = full_df["qty"].fillna(0)
    full_df["中文月份"] = full_df["年月_dt"].dt.strftime("%Y年%m月")
    return full_df

def run_predict_logic(monthly_df, pred_months, alpha, w_yoy, w_qoq, outlier_th, do_replace_outlier, proj_dict, series_factor, max_gain_rate):
    all_month_info = monthly_df[["年月_dt","中文月份"]].drop_duplicates().sort_values("年月_dt")
    latest_3_month = all_month_info.tail(3).reset_index(drop=True)
    m1_dt, m2_dt, m3_dt = latest_3_month["年月_dt"].iloc[0], latest_3_month["年月_dt"].iloc[1], latest_3_month["年月_dt"].iloc[2]
    m1_name, m2_name, m3_name = latest_3_month["中文月份"].iloc[0], latest_3_month["中文月份"].iloc[1], latest_3_month["中文月份"].iloc[2]
    last_year_m3_dt = m3_dt - pd.DateOffset(years=1)

    all_specs = sorted(monthly_df["商品规格"].unique())
    detail_rows = []
    outlier_log = []
    future_dt = pd.date_range(start=m3_dt + pd.DateOffset(months=1), periods=pred_months, freq="MS")
    future_col = [d.strftime("%Y年%m月预测") for d in future_dt]
    future_month_num = [d.month for d in future_dt]

    for spec in all_specs:
        g_all_region = monthly_df[monthly_df["商品规格"] == spec].copy()
        cate = g_all_region["商品类别"].iloc[0]
        full_panel_all = fill_full_panel(g_all_region.groupby(["年月_dt","中文月份"])["qty"].sum().reset_index())
        qty_raw_total = full_panel_all["qty"].values

        q_m1 = full_panel_all.loc[full_panel_all["年月_dt"] == m1_dt, "qty"].sum()
        q_m2 = full_panel_all.loc[full_panel_all["年月_dt"] == m2_dt, "qty"].sum()
        q_m3 = full_panel_all.loc[full_panel_all["年月_dt"] == m3_dt, "qty"].sum()
        q_m3_lastyear = full_panel_all.loc[full_panel_all["年月_dt"] == last_year_m3_dt, "qty"].sum()

        out_idx = detect_outlier_index(qty_raw_total, outlier_th)
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

        total_pred_list = rolling_predict(qty_for_model, pred_months, alpha, w_yoy, w_qoq, max_gain_rate)
        sf = series_factor.get(cate, 1.0)
        total_pred_list = [int(max(0, round(v * sf))) for v in total_pred_list]

        proj_add_list = []
        for idx in range(pred_months):
            mn = future_month_num[idx]
            mn_str = str(mn)
            if proj_dict is not None and spec in proj_dict and mn_str in proj_dict[spec]:
                proj_add_list.append(proj_dict[spec][mn_str])
            else:
                proj_add_list.append(0)
        final_pred = [int(max(0, round(total_pred_list[i] + proj_add_list[i]))) for i in range(pred_months)]
        sum_proj_add = sum(proj_add_list)

        if q_m2 > 1e-6:
            qoq_text = f"{(q_m3 - q_m2)/q_m2:.1%}"
        else:
            qoq_text = "-"
        if q_m3_lastyear > 1e-6:
            yoy_text = f"{(q_m3 - q_m3_lastyear)/q_m3_lastyear:.1%}"
        else:
            yoy_text = "-"

        row = {
            "商品规格": spec,
            "商品类别(系列)": cate,
            f"{m1_name}(真实出货)": q_m1,
            f"{m2_name}(真实出货)": q_m2,
            f"{m3_name}(真实出货)": q_m3,
            f"环比({m3_name}对比{m2_name})": qoq_text,
            f"同比({m3_name}对比{last_year_m3_dt.strftime('%Y年%m月')})": yoy_text,
            "项目剩余增量": sum_proj_add,
            "存在标记突增": len(out_idx) > 0
        }
        for idx, col_name in enumerate(future_col):
            row[col_name] = final_pred[idx]
        detail_rows.append(row)

    df_detail = pd.DataFrame(detail_rows)
    sum_col_list = [c for c in df_detail.columns if "(真实出货)" in c or c == "项目剩余增量" or "预测" in c]
    grp_sum = df_detail.groupby("商品类别(系列)")[sum_col_list].sum().reset_index()
    df_log = pd.DataFrame(outlier_log) if len(outlier_log) > 0 else pd.DataFrame()
    meta_info = {
        "m1_name": m1_name,
        "m2_name": m2_name,
        "m3_name": m3_name,
        "future_col": future_col
    }
    return df_detail, grp_sum, df_log, meta_info, future_col

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
        opt_text = [f"{i}｜{x['run_time']}｜{x['mode_name']}｜文件:{x['upload_filename']}" for i,x in enumerate(hist_list)]
        sel_idx = st.selectbox("选择历史记录", options=range(len(hist_list)), format_func=lambda i: opt_text[i])
        sel_item = hist_list[sel_idx]
        c1,c2 = st.columns(2)
        with c1:
            if st.button("打开选中记录"):
                st.session_state["view_history_item"] = sel_item
                st.session_state.page_mode = "history_detail"
                st.rerun()
        with c2:
            if st.button("删除选中记录"):
                hist_list.pop(sel_idx)
                st.session_state["predict_history"] = hist_list
                st.rerun()
        if st.button("清空全部历史"):
            st.session_state["predict_history"] = []
            st.rerun()
    else:
        st.info("暂无历史记录")
    st.stop()

# =====================【历史详情】=====================
if st.session_state.page_mode == "history_detail":
    item = st.session_state["view_history_item"]
    st.title(f"历史结果｜{item['mode_name']}｜{item['run_time']}")
    if st.button("返回历史列表"):
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
    cate_opt = ["全部系列"] + sorted(df_detail["商品类别(系列)"].unique())
    pick_cate = st.selectbox("筛选商品类别", cate_opt)
    if pick_cate != "全部系列":
        df_show = df_detail[df_detail["商品类别(系列)"] == pick_cate]
    else:
        df_show = df_detail
    df_show = clean_for_arrow(df_show)
    st.dataframe(df_show, width=1200, height=420)

    st.subheader("2、按产品类别汇总预测")
    df_sum_show = clean_for_arrow(df_cate_summary)
    st.dataframe(df_sum_show, width=1200)

    if not df_outlier_log.empty:
        st.subheader("3、异常标记日志")
        st.dataframe(clean_for_arrow(df_outlier_log))

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_detail.to_excel(writer, sheet_name="型号明细预测", index=False)
        df_cate_summary.to_excel(writer, sheet_name="按类别汇总", index=False)
        if not df_outlier_log.empty:
            df_outlier_log.to_excel(writer, sheet_name="异常标记日志", index=False)
    out_bytes = buf.getvalue()
    st.download_button("下载Excel结果", data=out_bytes, file_name=f"{item['mode_name']}_结果.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.stop()

# =====================【选择预测类型】=====================
if st.session_state.page_mode == "new_predict":
    st.title("数据预测")
    if st.button("返回首页"):
        st.session_state.page_mode = "home"
        st.rerun()
    st.divider()
    c1,c2 = st.columns(2)
    with c1:
        if st.button("整体预测", use_container_width=True):
            st.session_state.page_mode = "total"
            st.rerun()
    with c2:
        if st.button("部门预测", use_container_width=True):
            st.session_state.page_mode = "dept_select"
            st.rerun()
    st.stop()

# =====================【选择部门】=====================
if st.session_state.page_mode == "dept_select":
    st.title("部门预测")
    if st.button("返回数据预测"):
        st.session_state.page_mode = "new_predict"
        st.rerun()
    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        if st.button("华东", use_container_width=True):
            st.session_state.page_mode = "huadong"
            st.rerun()
    with c2:
        if st.button("华南", use_container_width=True):
            st.session_state.page_mode = "huanan"
            st.rerun()
    with c3:
        if st.button("华北", use_container_width=True):
            st.session_state.page_mode = "huabei"
            st.rerun()
    with c4:
        if st.button("渠道", use_container_width=True):
            st.session_state.page_mode = "channel"
            st.rerun()
    st.stop()

# =====================【预测主页面】=====================
mode_title_map = {
    "total":"整体预测",
    "huadong":"华东部门预测",
    "huanan":"华南部门预测",
    "huabei":"华北部门预测",
    "channel":"渠道部门预测"
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

# --------侧边栏--------
with st.sidebar:
    st.header("参数设置")
    upload_ship = st.file_uploader(f"上传{mode_title_map[current_mode]}数据表", type=["xlsx"])
    st.divider()
    st.subheader("简易参数")
    pred_months = st.slider("预测未来月数", min_value=1, max_value=3, value=3)
    alpha = st.slider("近期权重alpha", min_value=0.1, max_value=0.9, value=0.6, step=0.05)
    w_yoy = st.slider("同比参考权重w_yoy", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    w_qoq = st.slider("环比参考权重w_qoq", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    with st.expander("高级设置"):
        upload_isp = st.file_uploader("ISP上限表", type=["xlsx"])
        upload_project = st.file_uploader("项目增量表", type=["xlsx"])
        outlier_th = st.number_input("异常倍数阈值", min_value=1.5, max_value=10.0, value=3.0, step=0.25)
        max_gain_rate = st.number_input("单型号最大环比增幅倍数", min_value=1.0, value=2.0, step=0.1, help="限制单型号单月暴涨，2=下月最多是上月2倍")
        do_replace_outlier = st.checkbox("替换异常值", value=False)

# ISP加载存入session
if upload_isp is not None:
    isp_raw = pd.read_excel(upload_isp, engine="openpyxl")
    # 修复：数字表头转字符串再strip
    isp_raw.columns = [str(c).strip() for c in isp_raw.columns]
    first_col = isp_raw.columns[0]
    isp_cache = isp_raw.set_index(first_col)
    st.session_state["isp_config"] = isp_cache
    st.info("✅ ISP已加载到会话缓存")

# 项目增量解析
proj_dict = {}
if upload_project is not None:
    proj_raw = pd.read_excel(upload_project, engine="openpyxl")
    proj_raw.columns = [str(c).strip() for c in proj_raw.columns]
    need_cols = {"完整型号","需求月份","项目需求数量","已出数量"}
    if need_cols.issubset(set(proj_raw.columns)):
        proj_raw["项目需求数量"] = pd.to_numeric(proj_raw["项目需求数量"], errors="coerce").fillna(0)
        proj_raw["已出数量"] = pd.to_numeric(proj_raw["已出数量"], errors="coerce").fillna(0)
        proj_raw["剩余增量"] = proj_raw["项目需求数量"] - proj_raw["已出数量"]
        proj_raw.loc[proj_raw["剩余增量"] < 0, "剩余增量"] = 0
        pat = re.compile(r"(\d{1,2})月份")
        expand_list = []
        for _,row in proj_raw.iterrows():
            spec = clean_spec(str(row["完整型号"]))
            add = int(round(row["剩余增量"]))
            if add <= 0:
                continue
            matches = pat.findall(str(row["需求月份"]))
            for m_str in matches:
                expand_list.append((spec, m_str, add))
        proj_dict = {}
        for spec, m_str, add in expand_list:
            if spec not in proj_dict:
                proj_dict[spec] = {}
            proj_dict[spec][m_str] = proj_dict[spec].get(m_str,0) + add
        st.info(f"✅ 项目增量加载完成，共{len(proj_dict)}个型号")
    else:
        st.error("❌项目表表头不对，需要：完整型号、需求月份、项目需求数量、已出数量")

series_factor = {}
if upload_ship is not None:
    with st.spinner("读取数据，生成系列系数滑块……"):
        monthly_df_raw = load_ship_data(upload_ship)
    if current_mode == "huadong":
        monthly_df = monthly_df_raw[monthly_df_raw["区域"]=="华东区"].copy()
    elif current_mode == "huanan":
        monthly_df = monthly_df_raw[monthly_df_raw["区域"]=="华南区"].copy()
    elif current_mode == "huabei":
        monthly_df = monthly_df_raw[monthly_df_raw["区域"]=="华北区"].copy()
    elif current_mode == "channel":
        monthly_df = monthly_df_raw[monthly_df_raw["区域"]=="渠道部"].copy()
    else:
        monthly_df = monthly_df_raw.copy()

    if len(monthly_df) <= 0:
        st.error("该部门没有出货数据，请检查【订单部门】字段")
        st.stop()
    st.success(f"读取成功，型号总数：{monthly_df['商品规格'].nunique()}")

    # 【重点】自动读出所有系列，生成滑块
    all_series = sorted(monthly_df["商品类别"].unique())
    st.sidebar.subheader("📊 系列缩放系数（拖动滑块调整）")
    for s_name in all_series:
        sf_val = st.sidebar.slider(f"{s_name}", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
        series_factor[s_name] = sf_val

    df_origin, df_cate_sum, df_outlier_log, meta, future_col = run_predict_logic(
        monthly_df, pred_months, alpha, w_yoy, w_qoq, outlier_th, do_replace_outlier, proj_dict, series_factor, max_gain_rate
    )
    df_show = df_origin.copy()
    isp_cache = st.session_state.get("isp_config")
    if isp_cache is not None:
        df_show = apply_isp_series_cap(df_show, isp_cache, future_col)

    item_save = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode_name": mode_title_map[current_mode],
        "upload_filename": upload_ship.name,
        "df_detail": clean_for_arrow(df_show.copy()),
        "df_summary": clean_for_arrow(df_cate_sum.copy()),
        "df_log": clean_for_arrow(df_outlier_log.copy()),
        "meta": meta
    }
    st.session_state["predict_history"].append(item_save)
    if len(st.session_state["predict_history"]) > MAX_HISTORY:
        st.session_state["predict_history"].pop(0)

    st.divider()
    st.subheader("1、型号明细预测结果")
    cate_opt = ["全部系列"] + sorted(df_show["商品类别(系列)"].unique())
    pick_cate = st.selectbox("筛选系列", cate_opt)
    if pick_cate != "全部系列":
        df_view = df_show[df_show["商品类别(系列)"] == pick_cate]
    else:
        df_view = df_show
    df_view = clean_for_arrow(df_view)
    st.dataframe(df_view, width=1300, height=420)

    st.subheader("2、按系列汇总预测")
    sum_cols = [c for c in df_view.columns if "(真实出货)" in c or c == "项目剩余增量" or "预测" in c]
    df_sum_view = df_view.groupby("商品类别(系列)")[sum_cols].sum().reset_index()
    df_sum_view = clean_for_arrow(df_sum_view)
    st.dataframe(df_sum_view, width=1300)
else:
    st.info("请在侧边栏上传出货数据表开始预测")
