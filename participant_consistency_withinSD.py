import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
IN_CSV = "analysis_results.csv"
OUT_DIR = "."  # 필요하면 "figs" 같은 폴더명으로 바꿔도 됨

DEVICE_ORDER_4 = ["Smartphone", "Laptop", "Headphones", "AirPods"]

METRICS = [
    ("ASL", "ASL"),
    ("SNR", "SNR"),
    ("Speech Percentage", "Speech activity factor"),
    ("Duration", "Duration"),
]

# =========================
# Helpers: device / participant / condition
# =========================
def normalize_device_4(name: str) -> str:
    s = str(name).strip()
    if "AirPods" in s:
        return "AirPods"
    if ("Headset" in s) or ("Headphone" in s) or ("Sennheiser" in s) or ("PXC" in s):
        return "Headphones"
    if ("Laptop" in s) or ("ThinkPad" in s) or ("Lenovo" in s):
        return "Laptop"
    if ("Smartphone" in s) or ("iPhone" in s) or ("phone" in s.lower()):
        return "Smartphone"
    return "Other"

def extract_participant_id(row) -> str:
    # 1) Participant 컬럼이 있으면 우선 사용
    if "Participant" in row and pd.notna(row["Participant"]):
        return str(row["Participant"]).strip()
    # 2) 없으면 File_Name에서 Pxx 추출
    if "File_Name" in row and pd.notna(row["File_Name"]):
        m = re.search(r"(P\d{1,3})", str(row["File_Name"]))
        if m:
            return m.group(1)
    return "Unknown"

def extract_condition_nr(row) -> int | None:
    # 이미 ConditionNr 있으면 사용
    if "ConditionNr" in row and pd.notna(row["ConditionNr"]):
        try:
            return int(row["ConditionNr"])
        except:
            pass
    # 없으면 File_Name에서 _C(\d+)_ 패턴 추출
    if "File_Name" in row and pd.notna(row["File_Name"]):
        m = re.search(r"_C(\d+)_", str(row["File_Name"]))
        if m:
            return int(m.group(1))
    return None

# =========================
# Plot: Figure X (spaghetti device-wise)
# =========================
def plot_participant_spaghetti_device(df, metric_col, ylabel, out_png):
    # 참가자×디바이스 평균
    pivot = (df.groupby(["ParticipantID", "DeviceStd"])[metric_col]
               .mean()
               .unstack("DeviceStd"))

    # 디바이스 순서 고정
    pivot = pivot.reindex(columns=DEVICE_ORDER_4)

    # 참가자 중 결측 너무 많은 경우 대비(전부 NaN인 행 제거)
    pivot = pivot.dropna(how="all")

    x = np.arange(len(DEVICE_ORDER_4))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # 참가자별 얇은 선 (연하게)
    for pid, row in pivot.iterrows():
        y = row.values.astype(float)
        if not np.isfinite(y).all():
            continue
        ax.plot(x, y, linewidth=1.0, alpha=0.25, color="0.6")  # 연한 회색

    # 전체 평균 굵게
    mean_line = pivot.mean(axis=0).values.astype(float)
    ax.plot(x, mean_line, linewidth=3.0, marker="o", color="0.1")  # 진한 회색(거의 검정)

    ax.set_xticks(x)
    ax.set_xticklabels(DEVICE_ORDER_4, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="y", labelsize=12)  # y축 눈금 글씨
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

# =========================
# Plot: Figure Y (within-participant SD across 9 conditions)
# =========================
def plot_within_participant_sd(df, metric_col, ylabel, out_png):
    # 참가자×조건 평균(디바이스는 조건 내에서 평균으로 “접어” 버림)
    tmp = (df.groupby(["ParticipantID", "ConditionNr"])[metric_col]
             .mean()
             .reset_index())

    # 참가자별 조건 간 SD
    sd_by_participant = tmp.groupby("ParticipantID")[metric_col].std(ddof=1)
    sd_by_participant = sd_by_participant.dropna()

    vals = sd_by_participant.values.astype(float)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))

    # 바이올린 + 박스(요약)
    ax.violinplot([vals], showmeans=False, showmedians=True, showextrema=True)
    ax.boxplot([vals], widths=0.2, showfliers=False)

    # 개별 점도 같이(참가자 수가 적으면 가독성 좋아짐)
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(vals)) - 0.5) * 0.08
    ax.scatter(np.ones(len(vals)) + jitter, vals, s=14, alpha=0.55, color="0.2")

    ax.set_xticks([1])
    ax.set_xticklabels(["Participants"], fontsize=12)
    ax.set_ylabel(f"The standard deviation of {ylabel}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

# =========================
# Load + harmonize
# =========================
df = pd.read_csv(IN_CSV)

# 컬럼명 통일(네 파일마다 약간씩 달라질 수 있어서 안전하게)
rename_map = {}
if "ActiveSpeechLevel" in df.columns: rename_map["ActiveSpeechLevel"] = "ASL"
if "Percentage of Speech" in df.columns: rename_map["Percentage of Speech"] = "Speech Percentage"
if "SpeechPercentage" in df.columns: rename_map["SpeechPercentage"] = "Speech Percentage"
# Duration은 보통 이미 Duration
df = df.rename(columns=rename_map)

# Speech%가 0~1이면 0~100으로
if "Speech Percentage" in df.columns:
    sp = pd.to_numeric(df["Speech Percentage"], errors="coerce")
    if sp.dropna().max() <= 1.1:
        df["Speech Percentage"] = sp * 100.0

# Device 표준화
if "Device" not in df.columns:
    raise KeyError("analysis_results.csv에 'Device' 컬럼이 없어. 컬럼명을 확인해줘.")
df["DeviceStd"] = df["Device"].apply(normalize_device_4)
df = df[df["DeviceStd"].isin(DEVICE_ORDER_4)].copy()
df["DeviceStd"] = pd.Categorical(df["DeviceStd"], categories=DEVICE_ORDER_4, ordered=True)

# ParticipantID 생성
df["ParticipantID"] = df.apply(extract_participant_id, axis=1)
df = df[df["ParticipantID"] != "Unknown"].copy()

# ConditionNr 생성(없으면 File_Name에서 추출)
df["ConditionNr"] = df.apply(extract_condition_nr, axis=1)
df["ConditionNr"] = pd.to_numeric(df["ConditionNr"], errors="coerce")
df = df.dropna(subset=["ConditionNr"]).copy()
df["ConditionNr"] = df["ConditionNr"].astype(int)

INDOOR_CONDITIONS = [1, 2, 3, 4, 5]
# =========================
# Generate Figure X & Y for each metric
# =========================
for metric_col, ylabel in METRICS:
    if metric_col not in df.columns:
        print(f"[skip] '{metric_col}' column not found")
        continue

    out_x = f"{OUT_DIR}/FigX_spaghetti_device_{metric_col.replace(' ', '_')}.png"
    out_y = f"{OUT_DIR}/FigY_withinSD_{metric_col.replace(' ', '_')}.png"

    plot_participant_spaghetti_device(df, metric_col, ylabel, out_x)
    plot_within_participant_sd(df, metric_col, ylabel, out_y)

print("Done.")

#=========================
# Bandwidth -> numeric rank (for participant-level plots)
# =========================
if "Bandwidth" not in df.columns:
    raise KeyError("analysis_results.csv에 'Bandwidth' 컬럼이 없어. bandwidth 분석을 못해.")

# 표기 흔들림 정리
bw = df["Bandwidth"].astype(str).str.strip().str.upper()

# 비어있거나 이상한 값 처리
bw = bw.replace({"NAN": "EMPTY", "NONE": "EMPTY", "": "EMPTY", " ": "EMPTY"})
bw = bw.where(bw.isin(["EMPTY", "NB", "WB", "SWB", "FB"]), other="EMPTY")

df["BandwidthStd"] = bw

BW_RANK = {"EMPTY": 0, "NB": 1, "WB": 2, "SWB": 3, "FB": 4}
df["BandwidthRank"] = df["BandwidthStd"].map(BW_RANK).astype(float)

# =========================
# (optional) make spaghetti plot y-axis show category names
# =========================
def plot_participant_spaghetti_device_with_bw_ticks(df, metric_col, ylabel, out_png):
    df = df[df["ConditionNr"].isin(INDOOR_CONDITIONS)].copy()
    pivot = (df.groupby(["ParticipantID", "DeviceStd"])[metric_col]
               .mean()
               .unstack("DeviceStd")
               .reindex(columns=DEVICE_ORDER_4))

    pivot = pivot.dropna(how="all")
    x = np.arange(len(DEVICE_ORDER_4))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # participant lines (thin/light)
    for pid, row in pivot.iterrows():
        y = row.values.astype(float)
        ax.plot(x, y, linewidth=1.0, alpha=0.25, color="0.6")

    # overall mean (thick)
    mean_line = pivot.mean(axis=0).values.astype(float)
    ax.plot(x, mean_line, linewidth=3.0, marker="o", color="0.1")

    ax.set_xticks(x)
    ax.set_xticklabels(DEVICE_ORDER_4, fontsize=15)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=15)

    # ✅ bandwidth rank이면 y축을 범주 라벨로 보여주기
    if metric_col == "BandwidthRank":
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["empty", "NB", "WB", "SWB", "FB"])

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

def plot_within_participant_sd_bw(df, metric_col, ylabel, out_png):
    # 참가자×조건 평균(디바이스는 조건 내 평균으로 접음)
    tmp = (df.groupby(["ParticipantID", "ConditionNr"])[metric_col]
             .mean()
             .reset_index())

    sd_by_participant = tmp.groupby("ParticipantID")[metric_col].std(ddof=1).dropna()
    vals = sd_by_participant.values.astype(float)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.violinplot([vals], showmeans=False, showmedians=True, showextrema=True)
    ax.boxplot([vals], widths=0.2, showfliers=False)

    rng = np.random.default_rng(0)
    jitter = (rng.random(len(vals)) - 0.5) * 0.08
    ax.scatter(np.ones(len(vals)) + jitter, vals, s=14, alpha=0.55, color="0.2")

    ax.set_xticks([1])
    ax.set_xticklabels(["Participants"], fontsize=12)
    ax.set_ylabel(f"The standard deviation of {ylabel}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

# =========================
# Create Bandwidth Figure X & Y
# =========================
out_x_bw = f"{OUT_DIR}/FigX_spaghetti_device_BandwidthRank.png"
out_y_bw = f"{OUT_DIR}/FigY_withinSD_BandwidthRank.png"

plot_participant_spaghetti_device_with_bw_ticks(
    df, "BandwidthRank", "Bandwidth", out_x_bw
)

plot_within_participant_sd_bw(
    df, "BandwidthRank", "Bandwidth", out_y_bw
)

print("Bandwidth participant-level figures done.")
