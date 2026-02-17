"""
This script generates three sets of figures:

A) Device-type comparison (Supervised)
   - ASL, Speech Percentage, Duration

B) Inter-participant agreement (Supervised)
   - Participant-level distributions for ASL and Speech Percentage

C) Supervised vs Crowdsourced comparison
   - Distribution comparison for ASL, Speech Percentage, Duration
   - Implemented as mirrored horizontal histograms (percentage-based)

"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re

supervised_df = pd.read_csv("analysis_results.csv")

CONDITION_LABEL_MAP = {
    "In the lab doors/windows closed": "In the lab, doors/windows closed",
    "In the lab TV noise": "In the lab, TV noise through loudspeakers",
    "In the lab open windows": "In the lab, open windows",
    "In the lab open door": "In the lab, open door",
    "In the lab office": "In the lab, office with background speech",
    "In front of MAR": "In front of MAR-building",
    "Campus": "Campus",
    "U-Bhf": "U-Bahn station Ernst-Reuter-Platz",
    "Mensa": "Mensa",
}
supervised_df["Condition"] = supervised_df["Condition"].astype(str).str.strip()
supervised_df["Condition"] = supervised_df["Condition"].replace(CONDITION_LABEL_MAP)

CONDITION_ORDER = [
    "In the lab, doors/windows closed",
    "In the lab, TV noise through loudspeakers",
    "In the lab, open windows",
    "In the lab, open door",
    "In the lab, office with background speech",
    "In front of MAR-building",
    "Campus",
    "U-Bahn station Ernst-Reuter-Platz",
    "Mensa",
]

# =========================================
# Condition -> ConditionNr (1..9) 매핑 생성
# =========================================
COND_NR_MAP = {label: i+1 for i, label in enumerate(CONDITION_ORDER)}

# supervised_df에 번호 컬럼 추가
supervised_df["Condition Nr."] = supervised_df["Condition"].map(COND_NR_MAP).astype("Int64")

supervised_df["Condition"] = pd.Categorical(
    supervised_df["Condition"],
    categories=CONDITION_ORDER,
    ordered=True
)
supervised_df = supervised_df.sort_values("Condition")

supervised_df = supervised_df.rename(columns={
    "ActiveSpeechLevel": "ASL",
    "SNR": "SNR",
    "Percentage of Speech": "Speech Percentage",
    "Duration": "Duration",
})


def to_percent_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    vmax = s.dropna().max()

    if pd.isna(vmax):
        return s  # 전부 NaN이면 그대로

    if vmax <= 1.5:
        # 0~1 비율 → 0~100 퍼센트
        return s * 100.0
    elif vmax <= 150:
        # 이미 0~100 퍼센트
        return s
    elif vmax <= 15000:
        # 0~10000처럼 보이면(퍼센트 2번 적용) → 0~100으로 복구
        return s / 100.0
    else:
        # 너무 큰 값이면 일단 그대로 두고(혹시 단위 다름), 필요시 추가 처리
        return s

# =========================
# Device 라벨 통일: Smartphone / Laptop / Headphones / AirPods
# =========================
def normalize_device_4(name: str) -> str:
    s = str(name).strip()

    # AirPods는 헤드폰과 분리하고 싶다면 우선순위로 먼저 처리
    if "AirPods" in s:
        return "AirPods"

    # Headset/Headphones 계열 (Sennheiser 포함)
    if ("Headset" in s) or ("Headphone" in s) or ("Sennheiser" in s) or ("PXC" in s):
        return "Headphones"

    # Laptop
    if "Laptop" in s or "ThinkPad" in s or "Lenovo" in s:
        return "Laptop"

    # Smartphone
    if "Smartphone" in s or "iPhone" in s or "phone" in s.lower():
        return "Smartphone"

    return "Other"

# supervised_df의 Device 값을 표준 4개로 변경
supervised_df["Device"] = supervised_df["Device"].apply(normalize_device_4)

# 보기 순서도 고정 (그래프 순서가 매번 바뀌는 문제 방지)
DEVICE_ORDER_4 = ["Smartphone", "Laptop", "Headphones", "AirPods"]
supervised_df["Device"] = pd.Categorical(supervised_df["Device"], categories=DEVICE_ORDER_4, ordered=True)

sp = supervised_df["Speech Percentage"].dropna()
if len(sp) > 0 and sp.max() <= 1.1:
    supervised_df["Speech Percentage"] = supervised_df["Speech Percentage"] * 100.0



crowdsourced_df = pd.read_csv("batchTotal_report_final_recDevIncl.csv")

# Harmonize column names across datasets to simplify plotting

crowdsourced_df = crowdsourced_df.rename(columns={
    "ActiveSpeechLevelP56": "ASL",
    "speechPercentageP56": "Speech Percentage",
    "fileDurationInSec": "Duration",
    "deviceAndMicrophone": "Device",
})


# =========================
# 1) 공통 스타일 유틸
# =========================
def finalize(ax, ylabel, rotation=45):
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=17)

    # x/y tick 글씨 크기 같이 키우기
    ax.tick_params(axis="both", labelsize=14)

    # x tick 회전 + 정렬 + 글씨 크기
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right", fontsize=15)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
def save(fig, out_png, dpi=300):
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

def group_order(df, group_col):
    if group_col == "Condition":
        present = [c for c in CONDITION_ORDER if c in df[group_col].astype(str).unique()]
        return present

    if group_col == "Condition Nr.":
        nums = pd.to_numeric(df["Condition Nr."], errors="coerce").dropna().astype(int)
        return sorted(nums.unique().tolist())

    if group_col == "Device":
        preferred = ["Smartphone", "Laptop", "Headphones", "AirPods"]
        present = [d for d in preferred if d in df[group_col].astype(str).unique()]
        if present:
            return present

    return list(pd.Series(df[group_col].dropna().unique()).sort_values())

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
    """Make filenames safe (no spaces/dots/slashes)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def pretty_xticklabels(group_col, groups):
    """Convert group values into human-readable tick labels."""
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
    # Condition 라벨은 길어서 30~35도가 보기 좋음
    return 30 if group_col in ["Condition", "Condition Nr."] else 45

def figsize_for(group_col):
    # Condition 라벨이 길면 가로를 좀 키워야 겹침이 줄어듦
    return (10.5, 4.5) if group_col in ["Condition", "Condition Nr."] else (6.8, 4.2)


# =========================
# 2) ASL: Bar + 95% CI
# =========================
def plot_asl_bar_ci(df, group_col, out_png):
    g = df.groupby(group_col)["ASL"]
    means = g.mean()
    sem = g.sem()  # 표준오차
    ci95 = 1.96 * sem

    # 그룹 순서를 보기 좋게 정렬
    idx = means.index.tolist()
    x = np.arange(len(idx))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x, means.values, yerr=ci95.values, capsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(idx)
    finalize(ax, ylabel="ASL", rotation=45)
    save(fig, out_png)


def plot_asl_violin_box(df, group_col, out_png):
    groups = group_order(df, group_col)
    data = [df.loc[df[group_col] == g, "ASL"].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=figsize_for(group_col))

    # Violin: 분포 모양 + 중앙값/평균/극값 표시
    ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    # 얇은 boxplot 오버레이(논문에서 흔한 스타일)
    ax.boxplot(data, widths=0.15, showfliers=False)

    ax.set_xticks(range(1, len(groups) + 1))
    labels = pretty_xticklabels(group_col, groups)
    ax.set_xticklabels(labels)
    finalize(ax, ylabel="ASL", rotation=rotation_for(group_col))
    save(fig, out_png)
# =========================
# 3) Duration: Violin + Box (내부에 중앙값)
# =========================
def plot_duration_violin_box(df, group_col, out_png):
    groups = group_order(df, group_col)
    data = [df.loc[df[group_col] == g, "Duration"].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=figsize_for(group_col))
    ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    # 박스플롯을 얇게 덧입히면 논문에서 많이 쓰는 "violin+box" 느낌이 남
    ax.boxplot(data, widths=0.15, showfliers=False)

    ax.set_xticks(range(1, len(groups)+1))
    labels = pretty_xticklabels(group_col, groups)
    ax.set_xticklabels(labels)
    finalize(ax, ylabel="Duration", rotation=rotation_for(group_col))
    save(fig, out_png)

# =========================
# 4) Speech%: Boxplot
# =========================
def plot_saf_violin_box(df, group_col, out_png):
    groups = group_order(df, group_col)

    fig, ax = plt.subplots(figsize=figsize_for(group_col))

    data = []
    for g in groups:
        vals = df.loc[df[group_col] == g, "Speech Percentage"]
        vals = to_percent_0_100(vals).dropna().values
        data.append(vals)

    ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    ax.boxplot(data, widths=0.15, showfliers=False)

    ax.set_xticks(range(1, len(groups) + 1))
    labels = pretty_xticklabels(group_col, groups)
    ax.set_xticklabels(labels)
    finalize(ax, ylabel="speech activity factor", rotation=rotation_for(group_col))
    # ✅ 0~100으로 고정
    ax.set_ylim(0, 100)

    save(fig, out_png)

# =========================
# 5) SNR: Violin + Points (jitter)
# =========================
def plot_snr_violin_box_with_points(df, group_col, out_png, ylim=(25, 105)):
    groups = group_order(df, group_col)
    data = [df.loc[df[group_col] == g, "SNR"].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=figsize_for(group_col))

    # Figure 1/2처럼: violin은 분포만, box가 요약 담당
    ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    ax.boxplot(data, widths=0.15, showfliers=False)

    # 점(개별 녹음) 추가: 회색 + 작게 + 투명하게
    #rng = np.random.default_rng(0)
    #for i, arr in enumerate(data, start=1):
    #    if len(arr) == 0:
    #        continue
    #    jitter = (rng.random(len(arr)) - 0.5) * 0.15
    #    ax.scatter(
    #       np.full(len(arr), i) + jitter,
    #       arr,
    #       s=10,
    #        alpha=0.25,
    #        color="0.2",
    #        zorder=3
    #    )

    ax.set_xticks(range(1, len(groups) + 1))
    labels = pretty_xticklabels(group_col, groups)
    ax.set_xticklabels(labels)
    finalize(ax, ylabel="SNR", rotation=rotation_for(group_col))

    ax.set_ylim(*ylim)
    save(fig, out_png)
# =========================
# 6) 자동 생성: Device / Condition
# =========================
for group_col in ["Device", "Condition Nr."]:
    base = safe_name(group_col)  # "Condition_Nr" 같은 안전한 파일명

    plot_asl_violin_box(supervised_df, group_col, f"{base}_ASL_violin_box.png")
    plot_duration_violin_box(supervised_df, group_col, f"{base}_Duration_violin_box.png")
    plot_saf_violin_box(supervised_df, group_col, f"{base}_SAF_violin_box.png")

    # ✅ FIX: SNR도 group_col을 그대로 사용해야 함
    plot_snr_violin_box_with_points(supervised_df, group_col, f"{base}_SNR_violin_box.png", ylim=(25, 105))

    # (선택) outlier(100s) 때문에 중앙 분포가 안 보이면 확대 버전도 저장
    # plot_snr_violin_box_with_points(supervised_df, group_col, f"{base}_SNR_violin_box_zoom.png", ylim=(25, 60))
# =========================
# 7) mean csv기반
# =========================
# 1) CSV 불러오기
device_mean = pd.read_csv("device_statistics_mean.csv")
condition_mean = pd.read_csv("condition_statistics_mean.csv")

# 2) Speech 값이 0~1이면 %로 바꾸기
def ensure_speech_percent(df):
    col = "Percentage of Speech"
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().max() <= 1.5:
            df[col] = s * 100.0
    return df

device_mean = ensure_speech_percent(device_mean)
DEVICE_ORDER = [
    "Laptop (Lenovo ThinkPad X1 Carbon)",
    "Smartphone (iPhone 11 Pro)",
    "Headset (Sennheiser PXC 550)",
    "AirPods",
]
device_mean["Device"] = device_mean["Device"].astype(str).str.strip()

device_mean["Device"] = pd.Categorical(
    device_mean["Device"],
    categories=DEVICE_ORDER,
    ordered=True
)

device_mean = device_mean.sort_values("Device")


condition_mean = ensure_speech_percent(condition_mean)

CONDITION_LABEL_MAP = {
    "In the lab doors/windows closed": "In the lab, doors/windows closed",
    "In the lab TV noise": "In the lab, TV noise through loudspeakers",
    "In the lab open windows": "In the lab, open windows",
    "In the lab open door": "In the lab, open door",
    "In the lab office": "In the lab, office with background speech",
    "In front of MAR": "In front of MAR-building",
    "Campus": "Campus",
    "U-Bhf": "U-Bahn station Ernst-Reuter-Platz",
    "Mensa": "Mensa",
}
condition_mean["Condition"] = condition_mean["Condition"].astype(str).str.strip()
condition_mean["Condition"] = condition_mean["Condition"].replace(CONDITION_LABEL_MAP)

CONDITION_ORDER = [
    "In the lab, doors/windows closed",
    "In the lab, TV noise through loudspeakers",
    "In the lab, open windows",
    "In the lab, open door",
    "In the lab, office with background speech",
    "In front of MAR-building",
    "Campus",
    "U-Bahn station Ernst-Reuter-Platz",
    "Mensa",
]

condition_mean["Condition"] = pd.Categorical(
    condition_mean["Condition"],
    categories=CONDITION_ORDER,
    ordered=True
)
condition_mean = condition_mean.sort_values("Condition")

# 4) 그리고 싶은 지표들
metrics = [
    ("ASL", "ASL"),
    ("SNR", "SNR"),
    ("Percentage of Speech", "Percentage of Speech"),
    ("Duration", "Duration"),
]

# 5) 막대그래프 함수
def bar_plot(df, category_col, value_col, ylabel, title, filename, horizontal=False):
    cats = df[category_col].astype(str).tolist()
    vals = pd.to_numeric(df[value_col], errors="coerce").tolist()

    # ✅ 조건(환경) 그래프면 라벨이 많고 길어서 더 크게 + 그림도 넓게
    is_condition = category_col.lower().startswith("condition")
    fig_w = 14 if is_condition else 10  # 조건 그래프는 가로를 키움
    fig_h = 6
    plt.figure(figsize=(fig_w, fig_h))

    # ✅ (옵션) 조건이 숫자(1~9)라면 C1~C9로 짧게 바꿔서 더 크게 보이게
    if is_condition:
        short = []
        for c in cats:
            try:
                n = int(c)
                short.append(COND_SHORT_NR.get(n, f"C{n}"))
            except:
                short.append(c)
        cats = short

    if horizontal:
        plt.barh(cats, vals, height=0.6)
        plt.xlabel(ylabel, fontsize=14)
        plt.ylabel("")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    else:
        plt.bar(cats, vals, width=0.35)
        plt.ylabel(ylabel, fontsize=14)

        # ✅ 폰트/회전: 조건 그래프는 30도 + 11pt, 디바이스는 12pt
        rot = 30 if is_condition else 0
        fs = 12 if is_condition else 12
        plt.xticks(rotation=rot, ha="right", fontsize=fs)
        plt.yticks(fontsize=12)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# 6) Device별 평균 그래프 저장
for col, ylabel in metrics:
    if col in device_mean.columns:
        bar_plot(
            device_mean,
            category_col="Device",
            value_col=col,
            ylabel=ylabel,
            title=f"Supervised crowdsourcing simulation dataset (mean): {ylabel} by device type",
            filename=f"fig_3_1_device_mean_{col.replace(' ', '_')}.png"
        )

# 7) Condition별 평균 그래프 저장
for col, ylabel in metrics:
    if col in condition_mean.columns:
        bar_plot(
            condition_mean,
            category_col="Condition",
            value_col=col,
            ylabel=ylabel,
            title=f"Supervised crowdsourcing simulation dataset (mean): {ylabel} by environment condition",
            filename=f"fig_3_1_condition_mean_{col.replace(' ', '_')}.png"
        )



print("Done! Saved to:")

# ==========================================
# B) Plot: Participant-level comparison
# ==========================================
part_metrics = [
    ("ASL", "ASL"),
    ("Speech Percentage", "Speech Percentage(%)"),
]

for col, title in part_metrics:
    plt.figure(figsize=(12, 6))

    supervised_df["ParticipantID"] = supervised_df["File_Name"].str.extract(r'(P\d+)')
    supervised_df.boxplot(column=col, by="ParticipantID")

    plt.title(f"Participant comparison: {title}")

    plt.savefig(f"graph_participant_{col}.png")
    plt.close()

    print("saved_graph_participant")

# ==========================================
# C) Plot: Supervised vs Crowdsourced comparison
# ==========================================
"""
compare_list = [
    ("ASL", "ASL", "ASL"),
    ("SpeechPercentage", "SpeechPercentage", "SpeechPercentage(%)"),
    ("Duration", "Duration", "Duration"),
]

for col1, col2, title in compare_list:
    plt.figure(figsize=(6, 6))

    data1 = supervised_df[col1].dropna()
    data2 = crowdsourced_df[col2].dropna()

    plt.boxplot([data1, data2], tick_labels=["Supervised", "CS"])

    plt.title(f"Supervised vs Crowd: {title}")

    plt.savefig(f"VSgraph_compare_{col1}_{col2}.png")
    plt.close()
    print("saved_graph_compare")
"""

compare_list = [
    ("ASL", "ASL", "ASL"),
    ("Speech Percentage", "Speech Percentage", "Speech Percentage(%)"),
    ("Duration", "Duration", "Duration"),
]

# Number of bins for the histogram (adjustable)
BINS = 20

for col1, col2, title in compare_list:
    # Load values and remove missing entries
    data_sup = supervised_df[col1].dropna()
    data_crowd = crowdsourced_df[col2].dropna()

    # Use shared bin edges so both datasets are comparable on the same scale
    min_val = min(data_sup.min(), data_crowd.min())
    max_val = max(data_sup.max(), data_crowd.max())
    bins = np.linspace(min_val, max_val, BINS)

    # Convert counts to percentages
    sup_hist, _ = np.histogram(data_sup, bins=bins)
    crowd_hist, _ = np.histogram(data_crowd, bins=bins)

    sup_hist = sup_hist / sup_hist.sum() * 100
    crowd_hist = crowd_hist / crowd_hist.sum() * 100

    # Bin centers for y-axis placement
    y = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(10, 8))

    # Mirrored horizontal bars
    plt.barh(y, sup_hist, color="steelblue", alpha=0.7, label="Supervised")
    plt.barh(y, -crowd_hist, color="lightgreen", alpha=0.7, label="Crowd")

    # Symmetric x-limits around 0 for visual balance
    max_ratio = max(sup_hist.max(), crowd_hist.max())
    plt.xlim(-max_ratio * 1.2, max_ratio * 1.2)

    plt.xlabel("Percentage (%)")
    plt.ylabel(title)
    plt.title(f"{title}: Distribution Comparison (Supervised vs Crowd)")

    # Zero line in the center (0%)
    plt.axvline(0, color="black", linewidth=1)

    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"VSgraph_mirrored_{col1}_{col2}.png", dpi=300)
    plt.close()

    print(f"saved mirrored graph for {title}")