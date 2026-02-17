import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import re

import re

# 여기 아래에 네 helper 블록 붙여넣기
# =========================
# Pretty labels (C1–C9) for plots
# =========================
COND_SHORT_NR = {
    1: "C1 Lab closed",
    2: "C2 Lab TV",
    3: "C3 Lab open window",
    4: "C4 Lab open door",
    5: "C5 Lab Office",
    6: "C6 MAR entrance",
    7: "C7 Campus",
    8: "C8 U-Bahn",
    9: "C9 Mensa",
}

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def pretty_xticklabels(group_col, groups):
    if group_col == "Condition Nr.":
        out = []
        for g in groups:
            try:
                n = int(g)
                out.append(COND_SHORT_NR.get(n, f"C{n}"))
            except Exception:
                out.append(str(g))
        return out
    return [str(g) for g in groups]

def rotation_for(group_col):
    return 30 if group_col in ["Condition", "Condition Nr."] else 45

def figsize_for(group_col):
    return (10.5, 4.5) if group_col in ["Condition", "Condition Nr."] else (6.8, 4.2)


# =========================
# 0) Load
# =========================
supervised_df = pd.read_csv("analysis_results.csv")
crowdsourced_df = pd.read_csv("batchTotal_report_final_recDevIncl.csv")

OUTDIR = "fig_3_3"
os.makedirs(OUTDIR, exist_ok=True)


# =========================
# 1) Harmonize column names
# =========================
# Supervised columns already: ASL, Percentage of Speech, Duration, Loudness, Bandwidth, SNR, Condition, Device
# Crowdsourced columns: ActiveSpeechLevelP56, speechPercentageP56, fileDurationInSec, LoudnessITU-R1770-4,
#                       BandwiseBandwidthEstimation, deviceAndMicrophone

supervised_df = supervised_df.rename(columns={
    "Percentage of Speech": "SpeechPercentage",
    # keep ASL, Duration, Loudness, Bandwidth, Device as-is
})

crowdsourced_df = crowdsourced_df.rename(columns={
    "ActiveSpeechLevelP56": "ASL",
    "speechPercentageP56": "SpeechPercentage",
    "fileDurationInSec": "Duration",
    "LoudnessITU-R1770-4": "Loudness",
    "BandwiseBandwidthEstimation": "Bandwidth",
    "deviceAndMicrophone": "Device",
})

# 1) R번호 추출 (raw)
crowdsourced_df["R_raw"] = (
    crowdsourced_df["fileName"].astype(str)
    .str.extract(r"_R(\d{1,2})_")[0]
)
crowdsourced_df["R_raw"] = pd.to_numeric(crowdsourced_df["R_raw"], errors="coerce")
crowdsourced_df = crowdsourced_df.dropna(subset=["R_raw"]).copy()
crowdsourced_df["R_raw"] = crowdsourced_df["R_raw"].astype(int)

# 2) R08, R10 제거
crowdsourced_df = crowdsourced_df[~crowdsourced_df["R_raw"].isin([8, 10])].copy()

# 3) 재라벨 매핑: R09->8, R11->9 (나머지는 그대로)
R_TO_CONDNR = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 9:8, 11:9}

# 4) ConditionNr를 1~9로 "덮어쓰기"
crowdsourced_df["ConditionNr"] = crowdsourced_df["R_raw"].map(R_TO_CONDNR).astype("Int64")
crowdsourced_df = crowdsourced_df.dropna(subset=["ConditionNr"]).copy()

# 5) 체크
print("CS R_raw unique:", sorted(crowdsourced_df["R_raw"].unique().tolist()))
print("CS ConditionNr (aligned) unique:", sorted(crowdsourced_df["ConditionNr"].dropna().astype(int).unique().tolist()))


# =========================
# 2) Speech 0-1 -> percent (both datasets)
# =========================
def ensure_speech_percent(df, col="SpeechPercentage"):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().max() <= 1.5:
            df[col] = s * 100.0
    return df

supervised_df = ensure_speech_percent(supervised_df)
crowdsourced_df = ensure_speech_percent(crowdsourced_df)

# =========================
# 3) Bandwidth label cleanup
# =========================
def norm_bw(x):
    if pd.isna(x):
        return "empty"
    x = str(x).strip()
    # unify case
    if x.lower() in ["", "nan", "none"]:
        return "empty"
    if x.lower() == "empty":
        return "empty"
    return x.upper()

supervised_df["Bandwidth"] = supervised_df["Bandwidth"].apply(norm_bw)
crowdsourced_df["Bandwidth"] = crowdsourced_df["Bandwidth"].apply(norm_bw)

BW_ALL = ["NB", "WB", "SWB", "FB", "EMPTY"]       # 계산/집계용(EMPTY 포함)
BW_VALID = ["NB", "WB", "SWB", "FB"]

def bw_to_order(x):
    x = x.upper()
    if x == "EMPTY":
        return "EMPTY"
    if x in ["NB", "WB", "SWB", "FB"]:
        return x
    return "EMPTY"

supervised_df["Bandwidth"] = supervised_df["Bandwidth"].apply(bw_to_order)
crowdsourced_df["Bandwidth"] = crowdsourced_df["Bandwidth"].apply(bw_to_order)


# =========================
# A) DeviceGroup: "mic"로 분류하지 말고, 실제 라벨을 정확히 매핑
# =========================
# (1) Crowdsourced device strings (정확히 4개)
crowd_device_map = {
    "Smartphone, built-in mic": "Smartphone",
    "Laptop, built-in mic": "Laptop",
    "Laptop with a headset or external mic": "Headset",
    "Smartphone with a headset or external mic": "Headset",
}
crowdsourced_df["DeviceGroup"] = crowdsourced_df["Device"].map(crowd_device_map).fillna("Other")

# (2) Supervised device strings (너 파일에서 실제로 4개)
sup_device_map = {
    "Smartphone (iPhone 11 Pro)": "Smartphone",
    "Laptop (Lenovo ThinkPad X1 Carbon)": "Laptop",
    "AirPods": "Headset",
    "Headset (Sennheiser PXC 550)": "Headset",
}
supervised_df["DeviceGroup"] = supervised_df["Device"].map(sup_device_map).fillna("Other")

DEVICE_ORDER = ["Smartphone", "Laptop", "Headset"]
supervised_df  = supervised_df[supervised_df["DeviceGroup"].isin(DEVICE_ORDER)].copy()
crowdsourced_df = crowdsourced_df[crowdsourced_df["DeviceGroup"].isin(DEVICE_ORDER)].copy()


# =========================
# B) MatchedCondition: supervised는 키워드 말고 "명시적 매핑"으로 확정
# =========================
sup_cond_map = {
    "In the lab doors/windows closed": "Home_Silent",
    "In the lab TV noise": "Home_TV",
    "In the lab open windows": "Home_OpenWindow",
    "In the lab open door": "Home_OtherNoise",
    "In the lab office": "Office",
    "In front of MAR": "Street",
    "Campus": "Park",
    "U-Bhf": "SubwayStation",
    "Mensa": "CafeRestaurant",
}
supervised_df["MatchedCondition"] = supervised_df["Condition"].map(sup_cond_map)

# crowdsourced: R 추출 + (R08 shopping mall, R10 car) 제외 + 매핑
crowdsourced_df["R"] = crowdsourced_df["fileName"].str.extract(r"_R(\d{2})_")[0].astype(int)
crowdsourced_df = crowdsourced_df[~crowdsourced_df["R"].isin([8, 10])].copy()

crowd_cond_map = {
    1: "Home_Silent",
    2: "Home_TV",
    3: "Home_OpenWindow",
    4: "Home_OtherNoise",
    5: "Office",
    6: "Street",
    7: "Park",
    9: "SubwayStation",
    11: "CafeRestaurant",
}
crowdsourced_df["MatchedCondition"] = crowdsourced_df["R"].map(crowd_cond_map)

MATCHED_ORDER = [
    "Home_Silent", "Home_TV", "Home_OpenWindow", "Home_OtherNoise",
    "Office", "Street", "Park", "SubwayStation", "CafeRestaurant"
]
# ===== Pretty labels for condition plots (MatchedCondition order -> C1..C9) =====
COND_LABELS = [COND_SHORT_NR[i] for i in range(1, len(MATCHED_ORDER) + 1)]
COND_ROT = rotation_for("Condition Nr.")      # 30
COND_FIGSIZE = figsize_for("Condition Nr.")   # (10.5, 4.5)

# =========================
# C) df_cond / df_dev를 분리해서 그리기 (이게 중요!)
# =========================
supervised_df["Dataset"] = "Supervised"
crowdsourced_df["Dataset"] = "Crowdsourced"

df_all = pd.concat([supervised_df, crowdsourced_df], ignore_index=True)

DATASET_ORDER = ["Supervised", "Crowdsourced"]

DATASET_LABEL = {
    "Supervised": "Simulation CS",
    "Crowdsourced": "General CS",
}

DS_TICKLABELS = [DATASET_LABEL.get(ds, ds) for ds in DATASET_ORDER]


# =========================
# (NEW) Clean DeviceGroup labels + remove Other + drop empty categories
# =========================
# 1) 라벨 통일
df_all["DeviceGroup"] = df_all["DeviceGroup"].replace({"Headset/External": "Headset"})

# 2) Other 제거 (원하는 3개만)
DEVICE_ORDER = ["Smartphone", "Laptop", "Headset"]
df_dev = df_all[df_all["DeviceGroup"].isin(DEVICE_ORDER)].copy()

# 3) 데이터 없는 카테고리는 x축에서 제거
DEVICE_ORDER = [g for g in DEVICE_ORDER if df_dev["DeviceGroup"].eq(g).any()]

df_cond = df_all[df_all["MatchedCondition"].isin(MATCHED_ORDER)].copy()
df_dev  = df_all[df_all["DeviceGroup"].isin(DEVICE_ORDER)].copy()
df_all = df_all[df_all["DeviceGroup"].isin(["Smartphone", "Laptop", "Headset"])].copy()
DEVICE_ORDER = ["Smartphone", "Laptop", "Headset"]

# =========================
# D) 실행 전에 sanity check (이게 0이면 그래프가 비는 게 정상)
# =========================
print("== DeviceGroup counts ==")
print(pd.crosstab(df_dev["Dataset"], df_dev["DeviceGroup"]))
print("\n== MatchedCondition counts ==")
print(pd.crosstab(df_cond["Dataset"], df_cond["MatchedCondition"]))


# =========================
# 4) Add Dataset column & merge (for plotting convenience)
# =========================
supervised_df["Dataset"] = "Supervised"
crowdsourced_df["Dataset"] = "Crowdsourced"
df_all = pd.concat([supervised_df, crowdsourced_df], ignore_index=True)

DATASET_ORDER = ["Supervised", "Crowdsourced"]

DATASET_LABEL = {
    "Supervised": "Simulation CS",
    "Crowdsourced": "General CS",
}

DS_TICKLABELS = [DATASET_LABEL.get(ds, ds) for ds in DATASET_ORDER]


# =========================
# 5) Common plot utils (match 3.1 느낌)
# =========================
YLABEL_FONTSIZE = 16   # <- 원하는 크기로 조절 (예: 14~18)
YLABEL_PAD = 8

def finalize(ax, ylabel, rotation=0):
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE, labelpad=YLABEL_PAD)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    if rotation:
        plt.setp(
            ax.get_xticklabels(),
            rotation=rotation,
            ha="right",              # ✅ 핵심: 오른쪽 정렬
            rotation_mode="anchor",  # ✅ 핵심: 회전 기준 고정
            fontsize=16,
        )
    else:
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)

    ax.tick_params(axis="y", labelsize=12)

def save(fig, out_png, dpi=300):
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved: {out_png}")

# =========================
# (NEW) Overall unified style (violin + box with fixed colors)
# =========================
SUP_COLOR = "#4C78A8"  # blue
CRD_COLOR = "#72B7B2"  # teal
MEDIAN_COLOR = "#FF7F0E"

def finalize_overall(ax, ylabel):
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE, labelpad=YLABEL_PAD)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(axis="x", labelsize=12, pad=6)
    ax.tick_params(axis="y", labelsize=12)

def plot_violin_box_overall(df, metric, out_png, ylabel, ylim=None):
    # 안전장치: 데이터가 비면 violinplot이 깨질 수 있음
    data = []
    for ds in DATASET_ORDER:  # ["Supervised","Crowdsourced"]
        vals = pd.to_numeric(df.loc[df["Dataset"] == ds, metric], errors="coerce").dropna().values
        if len(vals) == 0:
            print(f"[WARN] No data for {metric} in dataset={ds}. Skip {out_png}")
            return
        data.append(vals)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    # violin (shape only)
    vp = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    colors = [SUP_COLOR, CRD_COLOR]
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.28)
        body.set_linewidth(0.8)

    # box (summary)
    bp = ax.boxplot(
        data, widths=0.22, showfliers=False, patch_artist=True
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.55)
        box.set_edgecolor("black")
        box.set_linewidth(1.0)

    for med in bp["medians"]:
        med.set_color(MEDIAN_COLOR)
        med.set_linewidth(1.6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(DS_TICKLABELS)

    if ylim is not None:
        ax.set_ylim(ylim)

    finalize_overall(ax, ylabel=ylabel)

    legend_handles = [
        Patch(facecolor=SUP_COLOR, alpha=0.55, label=f"{DATASET_LABEL['Supervised']} (left)"),
        Patch(facecolor=CRD_COLOR, alpha=0.55, label=f"{DATASET_LABEL['Crowdsourced']} (right)"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),  # ✅ 그래프 오른쪽 바깥
        frameon=True,
        fontsize=12
    )

    save(fig, os.path.join(OUTDIR, out_png))

# =========================
# 6) ASL: Mean bar + 95% CI (Supervised vs Crowdsourced)
# =========================
def plot_mean_ci95(df, metric, out_png, ylabel):
    g = df.groupby("Dataset")[metric]
    means = g.mean().reindex(DATASET_ORDER)
    sem = g.sem().reindex(DATASET_ORDER)
    ci95 = 1.96 * sem

    x = np.arange(len(DATASET_ORDER))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x, means.values, yerr=ci95.values, capsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(DS_TICKLABELS)
    finalize(ax, ylabel=ylabel, rotation=0)
    save(fig, os.path.join(OUTDIR, out_png))

# =========================
# 7) Violin + Box (Supervised vs Crowdsourced)
# =========================
def plot_violin_box(df, metric, out_png, ylabel):
    data = [
        pd.to_numeric(df.loc[df["Dataset"] == ds, metric], errors="coerce").dropna().values
        for ds in DATASET_ORDER
    ]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    ax.boxplot(data, widths=0.18, showfliers=False)

    ax.set_xticks(range(1, len(DATASET_ORDER) + 1))
    ax.set_xticklabels(DS_TICKLABELS)
    finalize(ax, ylabel=ylabel, rotation=0)
    save(fig, os.path.join(OUTDIR, out_png))

# =========================
# 8) Speech: Boxplot (Supervised vs Crowdsourced)
# =========================
def plot_box(df, metric, out_png, ylabel):
    data = [
        pd.to_numeric(df.loc[df["Dataset"] == ds, metric], errors="coerce").dropna().values
        for ds in DATASET_ORDER
    ]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.set_xticklabels(DS_TICKLABELS)
    finalize(ax, ylabel=ylabel, rotation=0)
    save(fig, os.path.join(OUTDIR, out_png))

# =========================
# 9) Bandwidth: stacked bar share% (Supervised vs Crowdsourced)
# =========================
def plot_bw_stacked_share(df, out_png):
    # 1) counts
    tab = pd.crosstab(df["Dataset"], df["Bandwidth"]).reindex(index=DATASET_ORDER, fill_value=0)

    # BW_ALL 기준으로 컬럼 보정
    for c in BW_ALL:
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[BW_ALL]

    # 2) EMPTY 비율 따로 계산/저장 (투명성)
    total = tab.sum(axis=1).replace(0, np.nan)
    empty_share = (tab["EMPTY"] / total * 100.0).fillna(0.0)
    empty_share.to_csv(os.path.join(OUTDIR, "BW_EMPTY_share_overall.csv"), header=["EMPTY (%)"])

    # 3) 그림/분석용: EMPTY 제외 후 재정규화(=NB/WB/SWB/FB만 100%)
    tab_valid = tab[BW_VALID]
    denom = tab_valid.sum(axis=1).replace(0, np.nan)
    share = (tab_valid.div(denom, axis=0) * 100.0).fillna(0.0)

    # 4) plot
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    bottom = np.zeros(len(DATASET_ORDER))
    x = np.arange(len(DATASET_ORDER))
    bar_w = 0.4

    color_map = {
        "NB": "#0072B2",
        "WB": "#E69F00",
        "SWB": "#009E73",
        "FB": "#CC79A7",
        # "EMPTY": "#D9D9D9"  # 이제 그림에서는 안 씀
    }

    for bw in BW_VALID:
        ax.bar(
            x,
            share[bw].values,
            bottom=bottom,
            width=bar_w,
            color=color_map[bw],
            edgecolor="white",
            linewidth=0.6,
            label=bw
        )
        bottom += share[bw].values

    ax.set_xticks(x)
    ax.set_xticklabels(DS_TICKLABELS, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)  # <- y축 tick 글자 크기
    ax.set_ylabel("Share", fontsize=14, labelpad=YLABEL_PAD)  # <- y축 라벨 크기
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.legend(title="Bandwidth", loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=True)
    fig.subplots_adjust(right=0.78)

    save(fig, os.path.join(OUTDIR, out_png))


# =========================
# 3.5) Extract R/S from crowdsourced fileName + filter R08/R10
# =========================
# R, S 추출
crowdsourced_df["R"] = crowdsourced_df["fileName"].str.extract(r"_R(\d{2})_").astype(float).astype("Int64")
crowdsourced_df["S"] = crowdsourced_df["fileName"].str.extract(r"_S(\d{2})_").astype(float).astype("Int64")

# shopping mall(R08), car(R10) 제외
crowdsourced_df = crowdsourced_df[~crowdsourced_df["R"].isin([8, 10])].copy()

# (선택) 문장도 맞춰서 공정 비교하고 싶으면:
# crowdsourced_df = crowdsourced_df[crowdsourced_df["S"].isin([1,2,3,4,5])].copy()

# =========================
# 3.6) DeviceGroup 만들기 (crowd / supervised 둘 다)
# =========================
def device_group_from_crowd(s):
    s = str(s).lower()
    if "smartphone" in s and ("headset" in s or "external" in s):
        return "Headset"
    if "laptop" in s and ("headset" in s or "external" in s):
        return "Headset"
    if "smartphone" in s:
        return "Smartphone"
    if "laptop" in s:
        return "Laptop"
    return "Other"

def device_group_from_supervised(s):
    s = str(s).lower()
    # 너 supervised Device 라벨에 맞춰 키워드만 조정하면 됨
    if "head" in s or "airpod" in s or "earbud" in s or "external" in s or "mic" in s:
        return "Headset"
    if "phone" in s or "smart" in s:
        return "Smartphone"
    if "laptop" in s or "thinkpad" in s:
        return "Laptop"
    return "Other"

crowdsourced_df["DeviceGroup"] = crowdsourced_df["Device"].apply(device_group_from_crowd)
supervised_df["DeviceGroup"] = supervised_df["Device"].apply(device_group_from_supervised)

# =========================
# 3.7) MatchedCondition 만들기 (9조건 매칭용)
# =========================
# crowd R -> matched label (R08/R10 이미 제외)
crowd_match = {
    1: "Home_Silent",
    2: "Home_TV",
    3: "Home_OpenWindow",
    4: "Home_OtherNoise",
    5: "Office",
    6: "Street",
    7: "Park",
    9: "SubwayStation",
    11:"CafeRestaurant"
}
crowdsourced_df["MatchedCondition"] = crowdsourced_df["R"].map(crowd_match)

# supervised Condition -> matched label
# ⚠️ 여기만 네 supervised 실제 Condition 값에 맞게 9개를 정확히 채우면 끝!
# 예시는 너가 이전에 쓴 9환경 이름 기준으로 적어둠.
sup_match = {
    "lab_closed": "Home_Silent",
    "lab_tv": "Home_TV",
    "lab_open_window": "Home_OpenWindow",
    "lab_open_door": "Home_OtherNoise",
    "lab_office_speech": "Office",
    "building_entrance": "Street",
    "campus_courtyard": "Park",
    "ubahn": "SubwayStation",
    "mensa": "CafeRestaurant",
}
supervised_df["MatchedCondition"] = supervised_df["Condition"].map(sup_match)

# 매칭 안 된 행 제거(라벨 mismatch 방지)
supervised_df = supervised_df[supervised_df["MatchedCondition"].notna()].copy()
crowdsourced_df = crowdsourced_df[crowdsourced_df["MatchedCondition"].notna()].copy()



# =========================
# 10) Export summary stats (for writing numbers in 3.3)
# =========================
def summarize_metric(df, metric):
    x = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(x) == 0:
        return pd.Series({"N": 0, "Mean": np.nan, "SD": np.nan, "Median": np.nan, "Q1": np.nan, "Q3": np.nan, "IQR": np.nan})
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    return pd.Series({
        "N": len(x),
        "Mean": x.mean(),
        "SD": x.std(ddof=1),
        "Median": x.median(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1
    })

metrics = [
    ("ASL", "ASL"),
    ("Duration", "Duration"),
    ("SpeechPercentage", "Speech activity factor"),
    ("Loudness", "Loudness (LUFS)"),
]

rows = []
for ds in DATASET_ORDER:
    sub = df_all[df_all["Dataset"] == ds]
    for m, _ in metrics:
        if m in sub.columns:
            s = summarize_metric(sub, m)
            s["Dataset"] = ds
            s["Metric"] = m
            rows.append(s)

summary_df = pd.DataFrame(rows)
summary_df = summary_df[["Dataset", "Metric", "N", "Mean", "SD", "Median", "Q1", "Q3", "IQR"]]
summary_df.to_csv(os.path.join(OUTDIR, "summary_supervised_vs_crowdsourced.csv"), index=False)

# =========================
# (NEW) DeviceGroup별 summary (Median/Q1/Q3/IQR + Mean/SD)
#  -> 그래프(박스플롯)와 동일한 df_dev를 사용하므로 수치가 정확히 대응됨
# =========================
rows_dev = []

for ds in DATASET_ORDER:
    for dev in DEVICE_ORDER:
        sub = df_dev[(df_dev["Dataset"] == ds) & (df_dev["DeviceGroup"] == dev)]
        for metric, metric_label in metrics:
            if metric not in sub.columns:
                continue
            s = summarize_metric(sub, metric)
            s["Dataset"] = ds
            s["DatasetLabel"] = DATASET_LABEL.get(ds, ds)
            s["DeviceGroup"] = dev
            s["Metric"] = metric
            s["MetricLabel"] = metric_label
            rows_dev.append(s)

summary_by_device = pd.DataFrame(rows_dev)

# 보기 좋은 컬럼 순서
summary_by_device = summary_by_device[
    ["Dataset", "DatasetLabel", "DeviceGroup", "Metric", "MetricLabel",
     "N", "Mean", "SD", "Median", "Q1", "Q3", "IQR"]
]

out_path = os.path.join(OUTDIR, "summary_by_devicegroup_supervised_vs_crowdsourced.csv")
summary_by_device.to_csv(out_path, index=False)
print("saved:", out_path)


# =========================
# 11) Make plots (core set for 3.3)
# =========================
plot_violin_box_overall(df_all, "ASL", "VS_ASL_overall.png", "ASL")
plot_violin_box_overall(df_all, "Duration", "VS_Duration_overall.png", "Duration")
plot_violin_box_overall(df_all, "SpeechPercentage", "VS_Speech_overall.png", "Speech activity factor")

# Bandwidth overall
plot_bw_stacked_share(df_all, "VS_Bandwidth_stacked_share.png")

print("Done! Outputs saved in:", OUTDIR)

# R, S 추출 (crowd only)
if "fileName" in crowdsourced_df.columns:
    crowdsourced_df["R"] = crowdsourced_df["fileName"].str.extract(r"_R(\d{2})_").astype(float).astype("Int64")
    crowdsourced_df["S"] = crowdsourced_df["fileName"].str.extract(r"_S(\d{2})_").astype(float).astype("Int64")
else:
    crowdsourced_df["R"] = pd.NA
    crowdsourced_df["S"] = pd.NA

# shopping mall(R08), car(R10) 제외
crowdsourced_df = crowdsourced_df[~crowdsourced_df["R"].isin([8, 10])].copy()

# Crowdsourced R -> MatchedCondition (9개만 남김)
crowd_match = {
    1: "Home_Silent",
    2: "Home_TV",
    3: "Home_OpenWindow",
    4: "Home_OtherNoise",
    5: "Office",
    6: "Street",
    7: "Park",
    9: "SubwayStation",
    11: "CafeRestaurant",
}
crowdsourced_df["MatchedCondition"] = crowdsourced_df["R"].map(crowd_match)

# =========================
# 3.6) Supervised: build MatchedCondition (자동 추정 + 필요시 수동맵)
# =========================
# 네 supervised_df에는 Condition 컬럼이 있다고 가정
# (Condition이 C01..C09 같은 형태면 아래 manual map만 너가 고치면 완벽)

manual_sup_map = {
    # 🔻필요하면 여기만 네 값에 맞춰 수정하면 됨 (예시)
    # "C01": "Home_Silent",
    # "C02": "Home_TV",
    # ...
}

def infer_sup_matched(cond):
    if pd.isna(cond):
        return np.nan
    s = str(cond).lower()

    # 이미 MatchedCondition 이름을 쓰고 있으면 그대로
    if s in [c.lower() for c in crowd_match.values()]:
        # 원래 대소문자 형태로 복원
        for v in crowd_match.values():
            if s == v.lower():
                return v

    # manual map 우선 적용
    if str(cond) in manual_sup_map:
        return manual_sup_map[str(cond)]

    # 키워드 기반 자동 추정(대부분 케이스에서 잘 맞음)
    if "tv" in s:
        return "Home_TV"
    if "open window" in s or "window" in s:
        return "Home_OpenWindow"
    if "open door" in s or "door" in s:
        return "Home_OtherNoise"
    if "office" in s:
        return "Office"
    if "ubahn" in s or "u-bahn" in s or "subway" in s or "metro" in s or "station" in s:
        return "SubwayStation"
    if "mensa" in s or "cafeteria" in s or "cafe" in s or "restaurant" in s:
        return "CafeRestaurant"
    if "campus" in s or "courtyard" in s or "park" in s:
        return "Park"
    if "entrance" in s or "street" in s or "outside" in s:
        return "Street"
    if "closed" in s or "silent" in s or "lab" in s:
        return "Home_Silent"

    return np.nan

if "Condition" in supervised_df.columns:
    supervised_df["MatchedCondition"] = supervised_df["Condition"].apply(infer_sup_matched)
else:
    supervised_df["MatchedCondition"] = np.nan

# 매칭 안 된 행은 제외(조건별 비교에서 섞이면 해석 망가짐)
supervised_df = supervised_df[supervised_df["MatchedCondition"].notna()].copy()
crowdsourced_df = crowdsourced_df[crowdsourced_df["MatchedCondition"].notna()].copy()

MATCHED_ORDER = [
    "Home_Silent", "Home_TV", "Home_OpenWindow", "Home_OtherNoise",
    "Office", "Street", "Park", "SubwayStation", "CafeRestaurant"
]

# =========================
# 3.7) DeviceGroup 만들기 (둘 다 공통)
# =========================
def device_group(x):
    if pd.isna(x):
        return "Other"
    s = str(x).lower()

    # 헤드셋/이어폰/외장마이크
    if ("with a headset or external mic" in s) or ("AirPods" in s) or ("Headset" in s):
        return "Headset"

    # 스마트폰
    if ("Smartphone, built-in mic" in s) or ("Smartphone (iPhone 11 Pro)" in s):
        return "Smartphone"

    if ("Laptop, built-in mic" in s) or ("Laptop (Lenovo ThinkPad X1 Carbon)" in s):
        return "Laptop"

    return "Other"

supervised_df["DeviceGroup"] = supervised_df["Device"].apply(device_group) if "Device" in supervised_df.columns else "Other"
crowdsourced_df["DeviceGroup"] = crowdsourced_df["Device"].apply(device_group) if "Device" in crowdsourced_df.columns else "Other"

DEVICE_ORDER = ["Smartphone", "Laptop", "Headset"]

# =========================
# 4) Add Dataset column & merge (for plotting convenience)  [<= 네 기존 코드 유지]
# =========================
supervised_df["Dataset"] = "Supervised"
crowdsourced_df["Dataset"] = "Crowdsourced"
df_all = pd.concat([supervised_df, crowdsourced_df], ignore_index=True)

DATASET_ORDER = ["Supervised", "Crowdsourced"]

DATASET_LABEL = {
    "Supervised": "Simulation CS",
    "Crowdsourced": "General CS",
}

DS_TICKLABELS = [DATASET_LABEL.get(ds, ds) for ds in DATASET_ORDER]


def ds_labels(order=DATASET_ORDER):
    return [DATASET_LABEL.get(ds, ds) for ds in order]

COND_GROUP_COL = "Condition Nr."
COND_GROUPS = list(range(1, len(MATCHED_ORDER) + 1))  # [1..9]
COND_LABELS = pretty_xticklabels(COND_GROUP_COL, COND_GROUPS)

COND_ROT = rotation_for(COND_GROUP_COL)       # 30도
COND_FIGSIZE = figsize_for(COND_GROUP_COL)    # (10.5, 4.5)
# 필요하면 "C01" 형식으로:
# COND_LABELS = [f"C{i:02d}" for i in range(1, len(MATCHED_ORDER) + 1)]

# ==========================================================
# NEW PLOTS
# ==========================================================
def tune_fonts_for_cat_plots(ax, xlabel_size=18, ylabel_size=24, legend_size=16, legend_title_size=16):
    ax.xaxis.label.set_size(xlabel_size)
    ax.yaxis.label.set_size(ylabel_size)

    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(legend_size)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(legend_title_size)


def plot_grouped_box_by_category(df, category_col, category_order, metric, out_png, ylabel,
                                 rotation=25, xtick_labels=None, figsize=(10.5, 4.4)):
    fig, ax = plt.subplots(figsize=(22, 7), dpi=150)

    positions = []
    data = []
    xticks = []
    x = 1

    for cat in category_order:
        for ds in DATASET_ORDER:
            vals = pd.to_numeric(
                df.loc[(df[category_col] == cat) & (df["Dataset"] == ds), metric],
                errors="coerce"
            ).dropna().values
            if len(vals) == 0:
                vals = np.array([np.nan])

            data.append(vals)

            positions.append(x)
            x += 1

        xticks.append((positions[-2] + positions[-1]) / 2)
        x += 1  # gap

    # ✅ patch_artist=True -> 박스 색칠 가능
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        showfliers=False,
        patch_artist=True
    )

    # ✅ 색: 왼쪽(Supervised), 오른쪽(Crowdsourced)
    sup_color = "#4C78A8"
    crd_color = "#72B7B2"

    for i, box in enumerate(bp["boxes"]):
        if i % 2 == 0:      # left box of each pair
            box.set_facecolor(sup_color)
        else:               # right box of each pair
            box.set_facecolor(crd_color)
        box.set_alpha(0.55)

    # (선택) 중앙값 선도 조금 진하게
    for med in bp["medians"]:
        med.set_linewidth(1.6)

    # x축 라벨
    ax.set_xticks(xticks)
    labels = xtick_labels if xtick_labels is not None else category_order
    ax.set_xticklabels(labels)  # ✅ 여기서는 회전/정렬 주지 않기

    ax.tick_params(axis="x", pad=6, labelsize=14)
    fig.subplots_adjust(bottom=0.33 if rotation else 0.18)

    finalize(ax, ylabel=ylabel, rotation=rotation)

    # ✅ 범례 추가 (이게 제일 중요)
    legend_handles = [
        Patch(facecolor=sup_color, alpha=0.55, label=f"{DATASET_LABEL['Supervised']} (left)"),
        Patch(facecolor=crd_color, alpha=0.55, label=f"{DATASET_LABEL['Crowdsourced']} (right)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=True)
    fig.subplots_adjust(right=0.60)

    if category_col in ["MatchedCondition", "DeviceGroup"]:
        tune_fonts_for_cat_plots(ax, xlabel_size=18, ylabel_size=24, legend_size=16, legend_title_size=16)
    # 기존 설명 텍스트는 유지해도 되고, 이제는 없어도 됨(선택)
    #ax.text(0.01, 1.02, "Within each category: left=Supervised, right=Crowdsourced",
    #        transform=ax.transAxes, fontsize=12)

    save(fig, os.path.join(OUTDIR, out_png))

BW_ORDER = ["NB", "WB", "SWB", "FB", "EMPTY"]
BW_PLOT  = [b for b in BW_ORDER if b != "EMPTY"]

def plot_bw_stacked_share_by_category(df, category_col, category_order, out_png,
                                      xtick_labels=None, figsize=(11.5, 4.8), rotation=0):
    fig, ax = plt.subplots(figsize=figsize)
    # x 위치: 카테고리마다 2개 막대
    base_x = np.arange(len(category_order))
    w = 0.26
    x_sup = base_x - w/2
    x_crowd = base_x + w/2

    def share_table(sub):
        # 1) 카테고리 x 대역폭 count 테이블
        tab = pd.crosstab(sub[category_col], sub["Bandwidth"]).reindex(index=category_order, fill_value=0)

        # 2) BW_ORDER 컬럼이 항상 존재하도록 보정
        for c in BW_ORDER:
            if c not in tab.columns:
                tab[c] = 0
        tab = tab[BW_ORDER]

        # 3) EMPTY% (전체 대비) 계산: 투명성 보고용
        total = tab.sum(axis=1).replace(0, np.nan)
        empty_pct = (tab["EMPTY"] / total * 100.0).fillna(0.0)

        # 4) plot용: EMPTY 제외 후 다시 100%로 재정규화
        tab_valid = tab[BW_PLOT]  # NB/WB/SWB/FB만
        denom = tab_valid.sum(axis=1).replace(0, np.nan)
        share_valid = (tab_valid.div(denom, axis=0) * 100.0).fillna(0.0)

        return share_valid, empty_pct

    share_sup, empty_sup = share_table(df[df["Dataset"] == "Supervised"])
    share_crowd, empty_crowd = share_table(df[df["Dataset"] == "Crowdsourced"])

    empty_out = pd.DataFrame({
        "EMPTY_%_Supervised": empty_sup,
        "EMPTY_%_Crowdsourced": empty_crowd
    }, index=category_order)
    empty_out.index.name = category_col
    empty_out.to_csv(os.path.join(OUTDIR, f"BW_EMPTY_share_by_{category_col}.csv"))

    # 누적 막대
    bottom_sup = np.zeros(len(category_order))
    bottom_crowd = np.zeros(len(category_order))

    color_map = {
        "NB": "#0072B2",  # blue
        "WB": "#E69F00",  # orange
        "SWB": "#009E73",  # green
        "FB": "#CC79A7"  # purple/pink
    }

    for bw in BW_PLOT:
        c = color_map.get(bw, "#999999")

        # left bar = Supervised
        ax.bar(
            x_sup, share_sup[bw].values, width=w, bottom=bottom_sup,
            color=c, edgecolor="white", linewidth=0.6, alpha=0.90,
            label=bw
        )
        # right bar = Crowdsourced (같은 색, 투명도만 낮게)
        ax.bar(
            x_crowd, share_crowd[bw].values, width=w, bottom=bottom_crowd,
            color=c, edgecolor="white", linewidth=0.6, alpha=0.90,
            label="_nolegend_"
        )

        bottom_sup += share_sup[bw].values
        bottom_crowd += share_crowd[bw].values

    ax.set_xticks(base_x)
    labels = xtick_labels if xtick_labels is not None else category_order
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    fig.subplots_adjust(bottom=0.25 if rotation else 0.18)

    ax.set_ylabel("Share", fontsize=YLABEL_FONTSIZE, labelpad=YLABEL_PAD)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.text(
        0.01, 1.02,
        f"For each category: left bar={DATASET_LABEL['Supervised']}, right bar={DATASET_LABEL['Crowdsourced']}",
        transform=ax.transAxes, fontsize=14
    )

    ax.legend(title="Bandwidth", bbox_to_anchor=(1.02, 1), loc="upper left")

    if category_col in ["MatchedCondition", "DeviceGroup"]:
        tune_fonts_for_cat_plots(ax, xlabel_size=18, ylabel_size=24, legend_size=16, legend_title_size=16)


    save(fig, os.path.join(OUTDIR, out_png))


# =========================
# (선택) 요약 통계도 category별로 export
# =========================
def export_summary_by_category(df, category_col, category_order, out_csv):
    rows = []
    for cat in category_order:
        for ds in DATASET_ORDER:
            sub = df[(df[category_col] == cat) & (df["Dataset"] == ds)]
            for m, _ in metrics:
                if m in sub.columns:
                    s = summarize_metric(sub, m)
                    s["Category"] = cat
                    s["CategoryCol"] = category_col
                    s["Dataset"] = ds
                    s["Metric"] = m
                    rows.append(s)
    out = pd.DataFrame(rows)
    out = out[["CategoryCol","Category","Dataset","Metric","N","Mean","SD","Median","Q1","Q3","IQR"]]
    out.to_csv(os.path.join(OUTDIR, out_csv), index=False)


# =========================
# 11) Make plots (NEW: MatchedCondition / DeviceGroup)
# =========================
# Condition별
plot_grouped_box_by_category(
    df_cond, "MatchedCondition", MATCHED_ORDER, "ASL",
    "VS_ASL_box_by_MatchedCondition.png", "ASL",
    rotation=COND_ROT, xtick_labels=COND_LABELS
)

plot_grouped_box_by_category(
    df_cond, "MatchedCondition", MATCHED_ORDER, "Duration",
    "VS_Duration_box_by_MatchedCondition.png", "Duration",
    rotation=COND_ROT, xtick_labels=COND_LABELS
)

plot_grouped_box_by_category(
    df_cond, "MatchedCondition", MATCHED_ORDER, "SpeechPercentage",
    "VS_Speech_box_by_MatchedCondition.png", "Speech activity factor",
    rotation=COND_ROT, xtick_labels=COND_LABELS
)


plot_bw_stacked_share_by_category(
    df_cond, "MatchedCondition", MATCHED_ORDER,
    "VS_Bandwidth_share_by_MatchedCondition.png",
    xtick_labels=COND_LABELS, figsize=COND_FIGSIZE, rotation=COND_ROT
)


# DeviceGroup별
plot_grouped_box_by_category(df_dev, "DeviceGroup", DEVICE_ORDER, "ASL",
                            "VS_ASL_box_by_DeviceGroup.png", "ASL", rotation=0)
plot_grouped_box_by_category(df_dev, "DeviceGroup", DEVICE_ORDER, "SpeechPercentage",
                            "VS_Speech_box_by_DeviceGroup.png", "Speech activity factor", rotation=0)
plot_grouped_box_by_category(df_dev, "DeviceGroup", DEVICE_ORDER, "Duration",
                            "VS_Duration_box_by_DeviceGroup.png", "Duration", rotation=0)
plot_bw_stacked_share_by_category(df_dev, "DeviceGroup", DEVICE_ORDER,
                                 "VS_Bandwidth_share_by_DeviceGroup.png")

# 요약 csv도 뽑고 싶으면(논문에 숫자 넣기용)
export_summary_by_category(df_all, "MatchedCondition", MATCHED_ORDER, "summary_by_MatchedCondition.csv")
export_summary_by_category(df_all, "DeviceGroup", DEVICE_ORDER, "summary_by_DeviceGroup.csv")