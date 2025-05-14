import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from scipy.optimize import minimize

# モデル読み込み（キャッシュ付き）
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)  # ← app/ フォルダの絶対パス

    model_visc = joblib.load(os.path.join(base_path, "models", "model_viscosity.joblib"))
    model_abs  = joblib.load(os.path.join(base_path, "models", "model_absorption.joblib"))
    model_str  = joblib.load(os.path.join(base_path, "models", "model_strength.joblib"))

    return model_visc, model_abs, model_str

model_visc, model_abs, model_str = load_models()

# タイトルとUI
st.title("demo")

st.sidebar.header("原材料・製造条件の入力")
input_a = st.sidebar.slider("小麦A比率（%）", 0.0, 1.0, 0.5)
input_b = st.sidebar.slider("小麦B比率（%）", 0.0, 1.0, 0.5)
water_ratio = st.sidebar.slider("水分比（%）", 0.4, 0.7, 0.55)
kneading_time = st.sidebar.slider("混練時間（分）", 0.0, 60.0, 30.0)
final_temp = st.sidebar.slider("最終温度（℃）", 20.0, 50.0, 35.0)

# 特徴量正規化
X = np.array([[input_a, input_b, final_temp / 50, kneading_time / 60]])

# 結果変数（初期値None）
results = None

# タブUI
tab1, tab2 = st.tabs(["予測結果", "原因分析・対策"])

with tab1:
    st.subheader("予測結果")

    if st.button("予測実行"):
        # 予測と判定
        pred_visc = model_visc.predict(X)[0]
        pred_abs  = model_abs.predict(X)[0]
        pred_str  = model_str.predict(X)[0]

        judge_visc = "合格" if pred_visc >= 60 else "不合格"
        judge_abs  = "合格" if 55 <= pred_abs <= 65 else "不合格"
        judge_str  = "合格" if pred_str >= 70 else "不合格"

        results = pd.DataFrame({
            "品質特性": ["粘度", "吸水率", "焼成強度"],
            "予測値": [round(pred_visc, 2), round(pred_abs, 2), round(pred_str, 2)],
            "判定": [judge_visc, judge_abs, judge_str]
        })

        def color_judge(val):
            return "color: green" if val == "合格" else "color: red"

        st.success("予測完了")
        st.dataframe(results.style.applymap(color_judge, subset=["判定"]))

        # ログ出力
        os.makedirs("logs", exist_ok=True)
        log_entry = {
            "時刻": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "小麦A": input_a, "小麦B": input_b,
            "水分比": water_ratio, "混練時間": kneading_time, "温度": final_temp,
            "粘度": pred_visc, "粘度_判定": judge_visc,
            "吸水率": pred_abs, "吸水率_判定": judge_abs,
            "焼成強度": pred_str, "焼成強度_判定": judge_str
        }
        df_log = pd.DataFrame([log_entry])
        log_path = "logs/predict_log.csv"
        df_log.to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))

with tab2:
    st.subheader("不合格時の原因分析・対策ページ")

    if results is None:
        st.info("まずは「予測実行」ボタンを押して品質予測を実行してください。")
    elif "不合格" in results["判定"].values:
        st.warning("不合格の品質特性が検出されました。以下の条件と参考値を確認してください。")

        # 合格の参考値（仮定）
        ref_values = {
            "小麦A": 0.6,
            "小麦B": 0.4,
            "水分比": 0.55,
            "混練時間": 30.0,
            "温度": 35.0
        }

        input_values = {
            "小麦A": input_a,
            "小麦B": input_b,
            "水分比": water_ratio,
            "混練時間": kneading_time,
            "温度": final_temp
        }

        df_compare = pd.DataFrame({
            "項目": list(ref_values.keys()),
            "現在値": list(input_values.values()),
            "参考値": list(ref_values.values())
        })
        st.dataframe(df_compare)

        # 再入力促し
        if st.button("条件を変更して再予測する"):
            st.experimental_rerun()
    else:
        st.success("すべて合格です。原因分析は不要です。")