import re
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from scipy import stats
except ImportError as e:
    raise ImportError("This section requires scipy. Install with: pip install scipy") from e

# =========================
# Config (입력/출력)
# =========================
SUP_CSV = "analysis_results.csv"
CS_CSV  = "batchTotal_report_final_recDevIncl.csv"

OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Helpers: condition / device / speech% / bandwidth
# =========================
def sup_condition_nr(file_name: str):
    m = re.search(r"_C(\d+)_", str(file_name))
    return int(m.group(1)) if m else None

def cs_condition_nr(file_name: str):
    # crowdsourced는 _R##_ 패턴
    m = re.search(r"_R(\d+)_", str(file_name))
    return int(m.group(1)) if m else None

def normalize_sup_device_collapsed(name: str) -> str:
    s = str(name).strip()
    # crowd와 맞추기 위해 AirPods + Headset 계열을 "Headset"으로 묶음
    if ("AirPods" in s) or ("Headset" in s) or ("Sennheiser" in s) or ("PXC" in s):
        return "Headset"
    if "Laptop" in s:
        return "Laptop"
    if "Smartphone" in s:
        return "Smartphone"
    return "Other"

def normalize_cs_device(name: str) -> str:
    s = str(name)
    if ("Laptop with a headset or external mic" in s) or ("Smartphone with a headset or external mic" in s):
        return "Headset"
    if "Laptop, built-in mic" in s:
        return "Laptop"
    if "Smartphone, built-in mic" in s:
        return "Smartphone"
    return "Other"

def to_percent_0_100(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    # 0~1 비율이면 0~100으로
    if x.dropna().max() <= 1.1:
        return x * 100.0
    return x

def normalize_bandwidth(series: pd.Series) -> pd.Series:
    bw = series.astype(str).str.strip().str.upper()
    bw = bw.replace({"NAN": "EMPTY", "NONE": "EMPTY", "": "EMPTY", " ": "EMPTY"})
    allowed = ["EMPTY", "NB", "WB", "SWB", "FB"]
    bw = bw.where(bw.isin(allowed), other="EMPTY")
    return bw

# =========================
# Load
# =========================
sup = pd.read_csv(SUP_CSV)
cs  = pd.read_csv(CS_CSV)

# =========================
# Harmonize metrics (mean-only, Loudness 제외)
# =========================
# supervised 컬럼명 후보 정리
# Speech%는 네 프로젝트에서 "Percentage of Speech"가 많음
sup_rename = {}
if "ActiveSpeechLevel" in sup.columns:  # 혹시 이런 이름이면
    sup_rename["ActiveSpeechLevel"] = "ASL"
if "Percentage of Speech" in sup.columns:
    sup_rename["Percentage of Speech"] = "Speech Percentage"
if "SpeechPercentage" in sup.columns:
    sup_rename["SpeechPercentage"] = "Speech Percentage"
sup = sup.rename(columns=sup_rename)

required_sup = {"ASL", "Speech Percentage", "Duration", "File_Name", "Device", "Bandwidth"}
missing_sup = required_sup - set(sup.columns)
if missing_sup:
    raise KeyError(f"Supervised missing columns: {sorted(missing_sup)}")

required_cs = {"ActiveSpeechLevelP56", "speechPercentageP56", "fileDurationInSec",
               "fileName", "deviceAndMicrophone", "BandwiseBandwidthEstimation"}
missing_cs = required_cs - set(cs.columns)
if missing_cs:
    raise KeyError(f"Crowdsourced missing columns: {sorted(missing_cs)}")

# Speech% 스케일 통일
sup["Speech Percentage"] = to_percent_0_100(sup["Speech Percentage"])
cs["speechPercentageP56"] = to_percent_0_100(cs["speechPercentageP56"])

# Stratification 변수 통일
sup["DeviceComp"] = sup["Device"].apply(normalize_sup_device_collapsed)
cs["DeviceComp"]  = cs["deviceAndMicrophone"].apply(normalize_cs_device)

sup["ConditionNr"] = sup["File_Name"].apply(sup_condition_nr)
cs["ConditionNr"]  = cs["fileName"].apply(cs_condition_nr)

# =========================
# ConditionNr alignment (ConditionNr 없이, ConditionNr 자체를 1~9로 맞춤)
#   - CS: R08, R10 제외
#   - CS: R09->8, R11->9 로 재라벨 (ConditionNr 컬럼에 덮어쓰기)
# =========================
R_TO_CONDNR = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 9:8, 11:9}

# supervised: 1~9만 남김
sup["ConditionNr"] = pd.to_numeric(sup["ConditionNr"], errors="coerce")
sup = sup.dropna(subset=["ConditionNr"]).copy()
sup["ConditionNr"] = sup["ConditionNr"].astype(int)
sup = sup[sup["ConditionNr"].between(1, 9)].copy()

# crowdsourced: R raw 보관(선택) -> 나중에 디버깅할 때 편함
cs["R_raw"] = pd.to_numeric(cs["ConditionNr"], errors="coerce")

cs = cs.dropna(subset=["R_raw"]).copy()
cs["R_raw"] = cs["R_raw"].astype(int)

# R08, R10 제외
cs = cs[~cs["R_raw"].isin([8, 10])].copy()

# ConditionNr를 '정렬된 1~9'로 덮어쓰기
cs["ConditionNr"] = cs["R_raw"].map(R_TO_CONDNR)
cs = cs.dropna(subset=["ConditionNr"]).copy()
cs["ConditionNr"] = cs["ConditionNr"].astype(int)

# (강추) 확인 로그
print("SUP ConditionNr:", sorted(sup["ConditionNr"].unique().tolist()))
print("CS  R_raw:", sorted(cs["R_raw"].unique().tolist()))
print("CS  ConditionNr(aligned):", sorted(cs["ConditionNr"].unique().tolist()))

# Bandwidth 라벨 통일
sup["BandwidthStd"] = normalize_bandwidth(sup["Bandwidth"])
cs["BandwidthStd"]  = normalize_bandwidth(cs["BandwiseBandwidthEstimation"])

# 비교 대상 device만 남김
sup = sup[sup["DeviceComp"].isin(["Smartphone", "Laptop", "Headset"])].copy()
cs  = cs[cs["DeviceComp"].isin(["Smartphone", "Laptop", "Headset"])].copy()

# =========================
# Utility: compare mean tables (tidy output)
# =========================
# =========================
# Utility: compare mean±SD tables (tidy output)
# =========================
METRIC_MAP = [
    ("ASL", "ASL", "ActiveSpeechLevelP56"),
    ("Speech activity factor (%)", "Speech Percentage", "speechPercentageP56"),
    ("Recording duration (s)", "Duration", "fileDurationInSec"),
]

def compare_mean_sd(group_cols, out_name: str):
    rows = []

    for label, sup_col, cs_col in METRIC_MAP:

        if len(group_cols) == 0:
            sup_stats = pd.DataFrame({
                "Supervised_mean": [pd.to_numeric(sup[sup_col], errors="coerce").mean()],
                "Supervised_sd":   [pd.to_numeric(sup[sup_col], errors="coerce").std(ddof=1)],
            })
            cs_stats = pd.DataFrame({
                "Crowdsourced_mean": [pd.to_numeric(cs[cs_col], errors="coerce").mean()],
                "Crowdsourced_sd":   [pd.to_numeric(cs[cs_col], errors="coerce").std(ddof=1)],
            })
            m = pd.concat([sup_stats, cs_stats], axis=1)

        else:
            sup_g = (
                sup.groupby(group_cols)[sup_col]
                   .agg(["mean", "std"])
                   .reset_index()
                   .rename(columns={"mean": "Supervised_mean", "std": "Supervised_sd"})
            )
            cs_g = (
                cs.groupby(group_cols)[cs_col]
                  .agg(["mean", "std"])
                  .reset_index()
                  .rename(columns={"mean": "Crowdsourced_mean", "std": "Crowdsourced_sd"})
            )
            m = pd.merge(sup_g, cs_g, on=group_cols, how="outer")

        m["Metric"] = label
        m["DiffMean(CS-SUP)"] = m["Crowdsourced_mean"] - m["Supervised_mean"]
        rows.append(m)

    out = pd.concat(rows, ignore_index=True)

    # column order
    if len(group_cols) == 0:
        out_cols = [
            "Metric",
            "Supervised_mean", "Supervised_sd",
            "Crowdsourced_mean", "Crowdsourced_sd",
            "DiffMean(CS-SUP)"
        ]
    else:
        out_cols = list(group_cols) + [
            "Metric",
            "Supervised_mean", "Supervised_sd",
            "Crowdsourced_mean", "Crowdsourced_sd",
            "DiffMean(CS-SUP)"
        ]

    out = out[out_cols]

    out_path = OUT_DIR / out_name
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"saved: {out_path}")

# =========================
# NEW: median / quartiles / IQR (+ mean±SD) tables
# =========================
def _summarize(x: pd.Series) -> dict:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if len(v) == 0:
        return {"N": 0, "Mean": np.nan, "SD": np.nan, "Median": np.nan, "Q1": np.nan, "Q3": np.nan, "IQR": np.nan}
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    return {
        "N": int(len(v)),
        "Mean": float(v.mean()),
        "SD": float(v.std(ddof=1)),
        "Median": float(v.median()),
        "Q1": float(q1),
        "Q3": float(q3),
        "IQR": float(q3 - q1),
    }

def compare_median_iqr(group_cols, out_name: str):
    rows = []

    for label, sup_col, cs_col in METRIC_MAP:

        # ---- supervised summary ----
        if len(group_cols) == 0:
            sup_s = pd.DataFrame([_summarize(sup[sup_col])])
        else:
            sup_s = (sup.groupby(group_cols)[sup_col]
                     .apply(_summarize)
                     .unstack()
                     .reset_index())
        sup_s = sup_s.rename(columns={c: f"Supervised_{c}" for c in ["N","Mean","SD","Median","Q1","Q3","IQR"]})

        # ---- crowdsourced summary ----
        if len(group_cols) == 0:
            cs_s = pd.DataFrame([_summarize(cs[cs_col])])
        else:
            cs_s = (cs.groupby(group_cols)[cs_col]
                    .apply(_summarize)
                    .unstack()
                    .reset_index())
        cs_s = cs_s.rename(columns={c: f"Crowdsourced_{c}" for c in ["N","Mean","SD","Median","Q1","Q3","IQR"]})

        # ---- merge ----
        if len(group_cols) == 0:
            m = pd.concat([sup_s, cs_s], axis=1)
        else:
            m = pd.merge(sup_s, cs_s, on=group_cols, how="outer")

        m["Metric"] = label
        m["DiffMedian(CS-SUP)"] = m["Crowdsourced_Median"] - m["Supervised_Median"]
        m["DiffMean(CS-SUP)"] = m["Crowdsourced_Mean"] - m["Supervised_Mean"]
        rows.append(m)

    out = pd.concat(rows, ignore_index=True)

    if len(group_cols) == 0:
        out_cols = [
            "Metric",
            "Supervised_N","Supervised_Median","Supervised_Q1","Supervised_Q3","Supervised_IQR","Supervised_Mean","Supervised_SD",
            "Crowdsourced_N","Crowdsourced_Median","Crowdsourced_Q1","Crowdsourced_Q3","Crowdsourced_IQR","Crowdsourced_Mean","Crowdsourced_SD",
            "DiffMedian(CS-SUP)","DiffMean(CS-SUP)"
        ]
    else:
        out_cols = list(group_cols) + [
            "Metric",
            "Supervised_N","Supervised_Median","Supervised_Q1","Supervised_Q3","Supervised_IQR","Supervised_Mean","Supervised_SD",
            "Crowdsourced_N","Crowdsourced_Median","Crowdsourced_Q1","Crowdsourced_Q3","Crowdsourced_IQR","Crowdsourced_Mean","Crowdsourced_SD",
            "DiffMedian(CS-SUP)","DiffMean(CS-SUP)"
        ]

    out = out[out_cols]
    out_path = OUT_DIR / out_name
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"saved: {out_path}")


# 1) Overall (THIS is what your professor asked for)
compare_mean_sd(group_cols=[], out_name="sup_vs_cs_overall_mean_sd.csv")
# =========================
# 2) By device comparison
# =========================
compare_mean_sd(group_cols=["DeviceComp"], out_name="sup_vs_cs_by_device_mean_sd.csv")

compare_median_iqr(group_cols=[], out_name="sup_vs_cs_overall_median_iqr.csv")
compare_median_iqr(group_cols=["DeviceComp"], out_name="sup_vs_cs_by_device_median_iqr.csv")
# =========================
# 3) By condition comparison
# =========================
# condition이 추출 안 된 행 제거
sup_c = sup.dropna(subset=["ConditionNr"]).copy()
cs_c  = cs.dropna(subset=["ConditionNr"]).copy()

sup_backup, cs_backup = sup, cs
sup, cs = sup_c, cs_c

compare_mean_sd(group_cols=["ConditionNr"], out_name="sup_vs_cs_by_condition_mean_sd.csv")
compare_median_iqr(group_cols=["ConditionNr"], out_name="sup_vs_cs_by_condition_median_iqr.csv")  # ✅ 이 줄 추가

sup, cs = sup_backup, cs_backup
# =========================
# 4) Bandwidth share comparisons (percent)
# =========================
BW_ORDER = ["EMPTY", "NB", "WB", "SWB", "FB"]

def bandwidth_share(df, group_cols):
    # group_cols별 bandwidth 분포(%)를 wide로
    if len(group_cols) == 0:
        pct = df["BandwidthStd"].value_counts(normalize=True) * 100
        out = pd.DataFrame({"Bandwidth": BW_ORDER})
        out["Percent"] = [float(pct.get(b, 0.0)) for b in BW_ORDER]
        return out
    tmp = (
        df.groupby(group_cols)["BandwidthStd"]
          .value_counts(normalize=True)
          .rename("share")
          .reset_index()
    )
    wide = tmp.pivot_table(index=group_cols, columns="BandwidthStd", values="share", fill_value=0) * 100
    wide = wide.reindex(columns=BW_ORDER, fill_value=0).reset_index()
    return wide

# 4-1) overall bandwidth share
sup_bw_over = bandwidth_share(sup, [])
cs_bw_over  = bandwidth_share(cs, [])
bw_over = pd.merge(
    sup_bw_over.rename(columns={"Percent": "Supervised(%)"}),
    cs_bw_over.rename(columns={"Percent": "Crowdsourced(%)"}),
    on="Bandwidth", how="outer"
).fillna(0)
bw_over["Diff(CS-SUP)"] = bw_over["Crowdsourced(%)"] - bw_over["Supervised(%)"]
bw_over.to_csv(OUT_DIR / "sup_vs_cs_bandwidth_overall.csv", index=False, float_format="%.2f")
print(f"saved: {OUT_DIR / 'sup_vs_cs_bandwidth_overall.csv'}")

# 4-2) by device bandwidth share
sup_bw_dev = bandwidth_share(sup, ["DeviceComp"])
cs_bw_dev  = bandwidth_share(cs, ["DeviceComp"])

# wide 형태라서 merge 후 각 컬럼 diff 계산
bw_dev = pd.merge(sup_bw_dev, cs_bw_dev, on=["DeviceComp"], how="outer", suffixes=("_SUP(%)", "_CS(%)")).fillna(0)

# Diff 컬럼 추가
for b in BW_ORDER:
    bw_dev[f"{b}_Diff(CS-SUP)"] = bw_dev[f"{b}_CS(%)"] - bw_dev[f"{b}_SUP(%)"]

bw_dev.to_csv(OUT_DIR / "sup_vs_cs_bandwidth_by_device.csv", index=False, float_format="%.2f")
print(f"saved: {OUT_DIR / 'sup_vs_cs_bandwidth_by_device.csv'}")

# 4-3) by condition bandwidth share
sup_c = sup.dropna(subset=["ConditionNr"]).copy()
cs_c  = cs.dropna(subset=["ConditionNr"]).copy()

sup_bw_cond = bandwidth_share(sup_c, ["ConditionNr"])
cs_bw_cond  = bandwidth_share(cs_c, ["ConditionNr"])   # <- 중복 호출 1번만!

bw_cond = pd.merge(
    sup_bw_cond, cs_bw_cond,
    on=["ConditionNr"], how="outer",
    suffixes=("_SUP(%)", "_CS(%)")
).fillna(0)

for b in BW_ORDER:
    bw_cond[f"{b}_Diff(CS-SUP)"] = bw_cond[f"{b}_CS(%)"] - bw_cond[f"{b}_SUP(%)"]

bw_cond = bw_cond.sort_values("ConditionNr")
bw_cond.to_csv(OUT_DIR / "sup_vs_cs_bandwidth_by_condition.csv", index=False, float_format="%.2f")
print(f"saved: {OUT_DIR / 'sup_vs_cs_bandwidth_by_condition.csv'}")


def sup_speaker_id(file_name: str):
    """Extract speaker id from supervised File_Name like P01_..."""
    base = Path(str(file_name)).name
    m = re.search(r"^P(\d+)", base)
    if not m:
        m = re.search(r"(?:^|[_-])P(\d+)(?:[_-]|$)", base)
    if not m:
        return None
    return f"P{int(m.group(1)):02d}"


def infer_cs_speaker_id(df: pd.DataFrame):
    """
    Prefer an explicit speaker/worker id column if present.
    Otherwise try to parse from fileName.
    Fallback: fileName itself (least ideal).
    """
    candidates = [
        "speakerId", "speaker_id", "speaker", "spk", "spkId",
        "workerId", "worker_id", "userId", "user_id",
        "participantId", "participant_id", "subjectId", "subject_id"
    ]
    for col in candidates:
        if col in df.columns:
            return df[col].astype(str)

    # try parsing from fileName
    s = df["fileName"].astype(str)
    # patterns: ...P12... or ...spk12... or ...speaker12...
    parsed = s.str.extract(r"(?:^|[_-])(P\d{1,3})(?:[_-]|$)", expand=False)
    if parsed.notna().any():
        return parsed.fillna(s)

    parsed2 = s.str.extract(r"(?:spk|speaker)(\d{1,4})", flags=re.IGNORECASE, expand=False)
    if parsed2.notna().any():
        return ("spk" + parsed2).fillna(s)

    return s


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment (FWER)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)

    # step-down: adj_i = max_{j<=i} ( (n-j)*p_j )
    running_max = 0.0
    for i, p in enumerate(ranked):
        factor = n - i
        val = factor * p
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)

    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def cliffs_delta(x, y):
    """Cliff's delta effect size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # O(n*m) but fine for typical speaker counts
    gt = sum(xi > yj for xi in x for yj in y)
    lt = sum(xi < yj for xi in x for yj in y)
    return (gt - lt) / (len(x) * len(y))


def iqr(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan
    return np.percentile(a, 75) - np.percentile(a, 25)


# --- create SpeakerID ---
sup["SpeakerID"] = sup["File_Name"].apply(sup_speaker_id)
cs["SpeakerID"] = infer_cs_speaker_id(cs)

# drop rows where we truly cannot identify a speaker in supervised
sup = sup.dropna(subset=["SpeakerID"]).copy()
cs = cs.dropna(subset=["SpeakerID"]).copy()

# --- unify metric columns into one schema ---
sup_u = sup[["SpeakerID", "ConditionNr", "DeviceComp", "ASL", "Speech Percentage", "Duration"]].copy()
sup_u = sup_u.rename(columns={"Speech Percentage": "SpeechPct"})
sup_u["Dataset"] = "SUP"

cs_u = cs[["SpeakerID", "ConditionNr", "DeviceComp",
           "ActiveSpeechLevelP56", "speechPercentageP56", "fileDurationInSec"]].copy()
cs_u = cs_u.rename(columns={
    "ActiveSpeechLevelP56": "ASL",
    "speechPercentageP56": "SpeechPct",
    "fileDurationInSec": "Duration"
})
cs_u["Dataset"] = "CS"

all_u = pd.concat([sup_u, cs_u], ignore_index=True)

# --- speaker-level aggregation (median recommended) ---
speaker_lvl = (
    all_u.groupby(["Dataset", "SpeakerID", "ConditionNr", "DeviceComp"], as_index=False)
         .agg({"ASL": "median", "SpeechPct": "median", "Duration": "median"})
)

# --- speaker-level descriptive summary by (condition, device) ---
metrics = ["ASL", "SpeechPct", "Duration"]
summary_rows = []
for metric in metrics:
    g = (speaker_lvl.groupby(["Dataset", "ConditionNr", "DeviceComp"])[metric]
                   .agg(median="median", q25=lambda x: np.percentile(x, 25),
                        q75=lambda x: np.percentile(x, 75), n="count")
                   .reset_index())
    g["IQR"] = g["q75"] - g["q25"]
    g["Metric"] = metric
    summary_rows.append(g)

summary = pd.concat(summary_rows, ignore_index=True)

# pivot to have SUP vs CS side-by-side
summary_w = summary.pivot_table(
    index=["ConditionNr", "DeviceComp", "Metric"],
    columns="Dataset",
    values=["n", "median", "IQR"],
    aggfunc="first"
).reset_index()

# flatten columns
summary_w.columns = [
    "_".join([c for c in col if c]) if isinstance(col, tuple) else col
    for col in summary_w.columns
]

summary_w = summary_w.rename(columns={
    "n_SUP": "n_sup", "n_CS": "n_cs",
    "median_SUP": "median_sup", "median_CS": "median_cs",
    "IQR_SUP": "iqr_sup", "IQR_CS": "iqr_cs"
})

summary_w.to_csv(OUT_DIR / "sup_vs_cs_speakerlevel_summary_by_condition_device.csv",
                 index=False, float_format="%.4f")
print(f"saved: {OUT_DIR / 'sup_vs_cs_speakerlevel_summary_by_condition_device.csv'}")


# --- statistical tests by (condition, device, metric) ---
test_rows = []
for metric in metrics:
    for cond in sorted(speaker_lvl["ConditionNr"].unique()):
        for dev in ["Smartphone", "Laptop", "Headset"]:
            sub = speaker_lvl[(speaker_lvl["ConditionNr"] == cond) & (speaker_lvl["DeviceComp"] == dev)]
            x = sub[sub["Dataset"] == "SUP"][metric].dropna().values
            y = sub[sub["Dataset"] == "CS"][metric].dropna().values

            # minimum n (you can tune this)
            if len(x) < 3 or len(y) < 3:
                continue

            # central tendency: Brunner–Munzel (robust), fallback to MWU
            try:
                bm = stats.brunnermunzel(x, y, alternative="two-sided")
                ct_test = "Brunner-Munzel"
                ct_stat, ct_p = float(bm.statistic), float(bm.pvalue)
            except Exception:
                mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
                ct_test = "Mann-Whitney U"
                ct_stat, ct_p = float(mwu.statistic), float(mwu.pvalue)

            # dispersion: Brown–Forsythe (Levene centered at median)
            lev = stats.levene(x, y, center="median")
            var_stat, var_p = float(lev.statistic), float(lev.pvalue)

            test_rows.append({
                "Metric": metric,
                "ConditionNr": cond,
                "DeviceComp": dev,
                "n_sup": len(x),
                "n_cs": len(y),
                "median_sup": float(np.median(x)),
                "median_cs": float(np.median(y)),
                "iqr_sup": float(iqr(x)),
                "iqr_cs": float(iqr(y)),
                "CentralTest": ct_test,
                "CentralStat": ct_stat,
                "p_central": ct_p,
                "VarStat": var_stat,
                "p_variance": var_p,
                "CliffsDelta": float(cliffs_delta(x, y)),
            })

tests = pd.DataFrame(test_rows)

# Holm correction separately for central and variance p-values
if len(tests) > 0:
    tests["p_central_holm"] = holm_adjust(tests["p_central"].values)
    tests["p_variance_holm"] = holm_adjust(tests["p_variance"].values)

tests.to_csv(OUT_DIR / "sup_vs_cs_speakerlevel_tests_by_condition_device.csv",
             index=False, float_format="%.6f")
print(f"saved: {OUT_DIR / 'sup_vs_cs_speakerlevel_tests_by_condition_device.csv'}")

print("Done.")