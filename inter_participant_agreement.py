import re
import pandas as pd

# ====== main_copy.py에서 가져온 헬퍼들 (그대로) ======
DEVICE_ORDER_4 = ["Smartphone", "Laptop", "Headphones", "AirPods"]

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
    if "Participant" in row and pd.notna(row["Participant"]):
        return str(row["Participant"]).strip()
    if "File_Name" in row and pd.notna(row["File_Name"]):
        m = re.search(r"(P\d{1,3})", str(row["File_Name"]))
        if m:
            return m.group(1)
    return "Unknown"

def extract_condition_nr(row):
    if "ConditionNr" in row and pd.notna(row["ConditionNr"]):
        try:
            return int(row["ConditionNr"])
        except:
            pass
    if "File_Name" in row and pd.notna(row["File_Name"]):
        m = re.search(r"_C(\d+)_", str(row["File_Name"]))
        if m:
            return int(m.group(1))
    return None

# ====== Load ======
df = pd.read_csv("analysis_results.csv")

# 컬럼명 통일(안전장치)
rename_map = {}
if "ActiveSpeechLevel" in df.columns: rename_map["ActiveSpeechLevel"] = "ASL"
if "SpeechPercentage" in df.columns: rename_map["SpeechPercentage"] = "Speech Percentage"
df = df.rename(columns=rename_map)

# Speech%가 0~1이면 0~100으로
if "Speech Percentage" in df.columns:
    sp = pd.to_numeric(df["Speech Percentage"], errors="coerce")
    if sp.dropna().max() <= 1.1:
        df["Speech Percentage"] = sp * 100.0

# DeviceStd
df["DeviceStd"] = df["Device"].apply(normalize_device_4)
df = df[df["DeviceStd"].isin(DEVICE_ORDER_4)].copy()

# ParticipantID / ConditionNr
df["ParticipantID"] = df.apply(extract_participant_id, axis=1)
df = df[df["ParticipantID"] != "Unknown"].copy()

df["ConditionNr"] = df.apply(extract_condition_nr, axis=1)
df = df.dropna(subset=["ConditionNr"]).copy()
df["ConditionNr"] = df["ConditionNr"].astype(int)

# BandwidthStd + Rank
bw = df["Bandwidth"].astype(str).str.strip().str.upper()
bw = bw.replace({"NAN": "EMPTY", "NONE": "EMPTY", "": "EMPTY", " ": "EMPTY"})
bw = bw.where(bw.isin(["EMPTY", "NB", "WB", "SWB", "FB"]), other="EMPTY")
bw = bw.where(bw.isin(["EMPTY", "NB", "WB", "SWB", "FB"]), other="EMPTY")
df["Bandwidth"] = bw
#df["BandwidthRank"] = df["Bandwidth"].map({"EMPTY":0,"NB":1,"WB":2,"SWB":3,"FB":4}).astype(float)

# --- Speech% 컬럼명 자동 통일 ---
speech_candidates = ["Speech Percentage", "Percentage of Speech", "SpeechPercentage", "speechPercentage"]
speech_found = next((c for c in speech_candidates if c in df.columns), None)

if speech_found is None:
    raise KeyError(f"Speech percentage column not found. Available columns: {list(df.columns)}")

# 최종 표준 이름으로 통일
if speech_found != "Speech Percentage":
    df = df.rename(columns={speech_found: "Speech Percentage"})

# ====== ✅ 참가자별 그래프 “원재료” long CSV 저장 ======
out_cols = [
    "File_Name",
    "ParticipantID",
    "Device",
    "ConditionNr",
    "ASL",
    "SNR",
    "Speech Percentage",
    "Duration",
    "Bandwidth",
]
df[out_cols].to_csv("participant_acoustic_long.csv", index=False)
print("saved: participant_acoustic_long.csv")

# =========================
# (추가) Device / Condition 별 그룹화 결과 CSV 저장 (mean-only)
# =========================
METRICS = ["ASL", "SNR", "Speech Percentage", "Duration"]

missing_metrics = [m for m in METRICS if m not in df.columns]
if missing_metrics:
    raise KeyError(f"Missing metric columns: {missing_metrics}. Available: {list(df.columns)}")

# 1) Device별 mean
mean_by_device = (
    df.groupby("DeviceStd")[METRICS]
    .mean()
    .reset_index()
)
mean_by_device.to_csv("mean_by_device.csv", index=False)
print("saved: mean_by_device.csv")

# 2) Condition별 mean
mean_by_condition = (
    df.groupby("ConditionNr")[METRICS]
    .mean()
    .reset_index()
    .sort_values("ConditionNr")
)
mean_by_condition.to_csv("mean_by_condition.csv", index=False)
print("saved: mean_by_condition.csv")

# 3) Condition × Device별 mean
#mean_by_condition_device = (
#    df.groupby(["ConditionNr", "DeviceStd"])[METRICS]
#    .mean()
#    .reset_index()
#    .sort_values(["ConditionNr", "DeviceStd"])
#)
#mean_by_condition_device.to_csv("mean_by_condition_device.csv", index=False)
#print("saved: mean_by_condition_device.csv")


# =========================
# (추가) Bandwidth는 범주형이라 mean이 안 됨 → 분포(share) 테이블로 저장
# =========================
BW_ORDER = ["EMPTY", "NB", "WB", "SWB", "FB"]

# A) Device별 Bandwidth 분포(행 합=1)
bw_share_by_device = pd.crosstab(df["DeviceStd"], df["Bandwidth"], normalize="index")
bw_share_by_device = bw_share_by_device.reindex(columns=BW_ORDER, fill_value=0).reset_index()
bw_share_by_device.to_csv("bandwidth_share_by_device.csv", index=False)
print("saved: bandwidth_share_by_device.csv")

# B) Condition별 Bandwidth 분포(행 합=1)
bw_share_by_condition = pd.crosstab(df["ConditionNr"], df["Bandwidth"], normalize="index")
bw_share_by_condition = bw_share_by_condition.reindex(columns=BW_ORDER, fill_value=0).reset_index()
bw_share_by_condition = bw_share_by_condition.sort_values("ConditionNr")
bw_share_by_condition.to_csv("bandwidth_share_by_condition.csv", index=False)
print("saved: bandwidth_share_by_condition.csv")

# C) Condition × Device별 Bandwidth 분포(행 합=1)
#tmp = (
#    df.groupby(["ConditionNr", "DeviceStd"])["Bandwidth"]
#    .value_counts(normalize=True)
#    .rename("share")
#    .reset_index()
#)
#bw_share_by_condition_device = (
#    tmp.pivot_table(
#        index=["ConditionNr", "DeviceStd"],
#        columns="Bandwidth",
#        values="share",
#        fill_value=0
#    )
#    .reindex(columns=BW_ORDER, fill_value=0)
#    .reset_index()
#    .sort_values(["ConditionNr", "DeviceStd"])
#)
#bw_share_by_condition_device.to_csv("bandwidth_share_by_condition_device.csv", index=False)
#print("saved: bandwidth_share_by_condition_device.csv")