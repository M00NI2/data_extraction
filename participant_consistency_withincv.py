import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def pretty_xticklabels(group_col, groups):
    """Convert group values into human-readable tick labels."""
    # ✅ 너 코드에서 쓰는 컬럼명들까지 커버
    if group_col in ["Condition Nr.", "ConditionNr", "Condition"]:
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
    return 30 if group_col in ["Condition", "Condition Nr.", "ConditionNr"] else 45

def figsize_for(group_col):
    # Condition 라벨이 길면 가로를 좀 키워야 겹침이 줄어듦
    return (10.5, 4.5) if group_col in ["Condition", "Condition Nr.", "ConditionNr"] else (6.8, 4.2)


# =========================
# Config
# =========================
IN_CSV = "analysis_results.csv"
OUT_DIR = "."


DEVICE_ORDER_4 = ["Smartphone", "Laptop", "Headphones", "AirPods"]

# ✅ laptop이 outdoor(C6–C9)에는 없으므로,
#    4-device 비교(스파게티/변동성)는 indoor 조건만 쓰는 게 가장 깔끔함
USE_INDOOR_ONLY_FOR_4DEVICE = True
INDOOR_CONDITIONS = {1, 2, 3, 4, 5}

# ✅ 5문장 평균을 명시적으로 만들기 위한 최소 문장 개수
MIN_SENTENCES_PER_CELL = 5  # 완벽 버전(권장). 데이터 누락 많으면 4로 낮춰도 됨.

# ✅ 교수님 요구 반영: participant 섹션에서 duration은 보통 제거 권장
INCLUDE_DURATION = False

# ✅ 지표 목록 (Bandwidth 제외)
METRICS = [
    ("ASL", "ASL"),
    ("SNR", "SNR"),
    ("Speech Activity Factor", "Speech activity factor"),
]
if INCLUDE_DURATION:
    METRICS.append(("Duration", "Recording duration (s)"))

# =========================
# Helpers: device / participant / condition / sentence
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
    if "Participant" in row and pd.notna(row["Participant"]):
        return str(row["Participant"]).strip()
    if "File_Name" in row and pd.notna(row["File_Name"]):
        m = re.search(r"(P\d{1,3})", str(row["File_Name"]))
        if m:
            return m.group(1)
    return "Unknown"

def extract_condition_nr(row) -> int | None:
    if "ConditionNr" in row and pd.notna(row["ConditionNr"]):
        try:
            return int(row["ConditionNr"])
        except Exception:
            pass
    if "File_Name" in row and pd.notna(row["File_Name"]):
        m = re.search(r"_C(\d+)_", str(row["File_Name"]))
        if m:
            return int(m.group(1))
    return None

def extract_sentence_nr(row) -> int | None:
    """
    파일명 패턴 예: Pxx_G_Cyy_D_ Szz.wav  (혹은 _Szz_)
    """
    if "SentenceNr" in row and pd.notna(row["SentenceNr"]):
        try:
            return int(row["SentenceNr"])
        except Exception:
            pass

    if "File_Name" in row and pd.notna(row["File_Name"]):
        fn = str(row["File_Name"])
        m = re.search(r"_S(\d+)_", fn) or re.search(r"_S(\d+)\b", fn) or re.search(r"S(\d+)", fn)
        if m:
            return int(m.group(1))
    return None

# =========================
# Core: sentence-first aggregation
# =========================
def build_sentence_aggregated(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """
    1) ParticipantID, ConditionNr, DeviceStd, SentenceNr를 만든다
    2) (P, C, D, S) 단위로 먼저 평균(중복 방지/문장 균등화의 출발점)
    3) 그 다음 (P, C, D) 단위로 5문장 평균을 만들어 "한 셀=한 값"을 만든다
    """
    df = df.copy()

    # ParticipantID
    df["ParticipantID"] = df.apply(extract_participant_id, axis=1)
    df = df[df["ParticipantID"] != "Unknown"].copy()

    # ConditionNr
    df["ConditionNr"] = df.apply(extract_condition_nr, axis=1)
    df["ConditionNr"] = pd.to_numeric(df["ConditionNr"], errors="coerce")
    df = df.dropna(subset=["ConditionNr"]).copy()
    df["ConditionNr"] = df["ConditionNr"].astype(int)

    # SentenceNr
    df["SentenceNr"] = df.apply(extract_sentence_nr, axis=1)
    df["SentenceNr"] = pd.to_numeric(df["SentenceNr"], errors="coerce")
    df = df.dropna(subset=["SentenceNr"]).copy()
    df["SentenceNr"] = df["SentenceNr"].astype(int)

    # ✅ step A: (P,C,D,S)로 먼저 접기
    g1_keys = ["ParticipantID", "ConditionNr", "DeviceStd", "SentenceNr"]
    df_g1 = (
        df.groupby(g1_keys, as_index=False)[metric_cols]
          .mean()
    )

    # ✅ step B: (P,C,D)에서 "문장 5개 평균" 만들기
    # 문장 개수 체크
    counts = df_g1.groupby(["ParticipantID", "ConditionNr", "DeviceStd"])["SentenceNr"].nunique()
    ok_index = counts[counts >= MIN_SENTENCES_PER_CELL].index

    df_g1 = df_g1.set_index(["ParticipantID", "ConditionNr", "DeviceStd"])
    df_g1 = df_g1.loc[df_g1.index.isin(ok_index)].reset_index()

    df_cell = (
        df_g1.groupby(["ParticipantID", "ConditionNr", "DeviceStd"], as_index=False)[metric_cols]
             .mean()
    )
    return df_cell

# =========================
# Plot: device spaghetti (participant trajectories across devices)
# =========================
def plot_spaghetti_by_device(cell_df: pd.DataFrame, metric_col: str, ylabel: str, out_png: str):
    # (P,D)로 평균을 내되, 어떤 범위를 평균내는지 명시적으로 고정
    tmp = cell_df.copy()

    if USE_INDOOR_ONLY_FOR_4DEVICE:
        tmp = tmp[tmp["ConditionNr"].isin(INDOOR_CONDITIONS)].copy()

    # participant × device (across selected conditions)
    pivot = (
        tmp.groupby(["ParticipantID", "DeviceStd"])[metric_col]
           .mean()
           .unstack("DeviceStd")
           .reindex(columns=DEVICE_ORDER_4)
    )

    pivot = pivot.dropna(how="all")

    # ✅ 스파게티 선 끊김 방지: complete-case를 기본으로 하되,
    #    너무 엄격해서 0명 되면 자동 완화
    complete = pivot.dropna(subset=DEVICE_ORDER_4)
    if len(complete) >= 3:
        pivot_plot = complete
    else:
        pivot_plot = pivot  # fallback (데이터 누락이 많으면 어쩔 수 없이 일부 선이 끊길 수 있음)

    x = np.arange(len(DEVICE_ORDER_4))
    fig, ax = plt.subplots(figsize=(9.2, 5.6))  # 라벨 가독성 개선

    for _, row in pivot_plot.iterrows():
        ax.plot(x, row.values.astype(float), linewidth=1.0, alpha=0.25, color="0.6")

    mean_line = pivot_plot.mean(axis=0).values.astype(float)
    ax.plot(x, mean_line, linewidth=3.0, marker="o", color="0.1")

    ax.set_xticks(x)
    ax.set_xticklabels(DEVICE_ORDER_4, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

def plot_delta_vs_baseline(cell_df: pd.DataFrame, metric_col: str, ylabel: str, out_png: str,
                           baseline: str = "Smartphone"):
    """
    참가자별 device effect를 'baseline 대비 차이(Δ)'로 시각화.
    - participant마다 baseline이 있을 때만 Δ 계산됨.
    - 각 device에 값이 없는 참가자는 해당 device Δ에서 자동으로 제외됨(끊김 없음).
    """

    tmp = cell_df.copy()

    # 4-device 비교를 indoor-only로 맞추고 싶으면 그대로 유지
    if USE_INDOOR_ONLY_FOR_4DEVICE:
        tmp = tmp[tmp["ConditionNr"].isin(INDOOR_CONDITIONS)].copy()

    # participant × device 값 (선택 조건들에 대해 평균)
    pivot = (
        tmp.groupby(["ParticipantID", "DeviceStd"])[metric_col]
           .mean()
           .unstack("DeviceStd")
           .reindex(columns=DEVICE_ORDER_4)
    )

    # baseline이 없는 참가자는 Δ를 정의할 수 없으니 제외
    pivot = pivot.dropna(subset=[baseline])

    # Δ 계산 (각 device - baseline)
    deltas = {}
    for dev in DEVICE_ORDER_4:
        if dev == baseline:
            continue
        deltas[dev] = (pivot[dev] - pivot[baseline]).dropna()

    # 그릴 데이터 준비
    data = []
    labels = []
    for dev in ["Laptop", "Headphones", "AirPods"]:
        if dev in deltas and len(deltas[dev]) > 0:
            data.append(deltas[dev].astype(float).values)
            labels.append(f"{dev} - {baseline}")

    if not data:
        print(f"[skip] Delta plot for {metric_col}: no data after filtering")
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    # 분포 + 요약(중앙값) 같이
    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    ax.boxplot(data, widths=0.18, showfliers=False)

    # 0 기준선(Δ=0)
    ax.axhline(0, linewidth=1.2, linestyle="--", color="0.3", alpha=0.8)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel(f"Δ {ylabel} (participant-level)", fontsize=16)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

# =========================
# Plot: within-participant CV across conditions (per device)
# =========================
def plot_within_participant_cv(cell_df: pd.DataFrame, metric_col: str, ylabel: str, out_png: str):
    """
    참가자별로 '조건 간 변동성'을 봄.
    교수님 요구대로 SD 대신 CV = SD / |Mean| 사용.
    여기서는 device별로 따로 CV를 계산해서 비교(섞어서 평균내지 않음).
    """
    tmp = cell_df.copy()

    data = []
    labels = []

    for dev in DEVICE_ORDER_4:
        dev_df = tmp[tmp["DeviceStd"] == dev].copy()

        # Laptop만 indoor-only
        if dev == "Laptop":
            dev_df = dev_df[dev_df["ConditionNr"].isin(INDOOR_CONDITIONS)].copy()

        # participant별 CV 계산: 환경 축으로 mean/sd
        grp = dev_df.groupby("ParticipantID")[metric_col]

        mean_p = grp.mean()
        sd_p = grp.std(ddof=1)

        # 최소 2개 환경 이상 있는 participant만(표준편차 정의 위해)
        n_env = dev_df.groupby("ParticipantID")["ConditionNr"].nunique()
        ok = n_env[n_env >= 2].index

        cv = (sd_p.loc[ok] / mean_p.loc[ok].abs()).replace([np.inf, -np.inf], np.nan).dropna()

        if len(cv) > 0:
            data.append(cv.astype(float).values)
            labels.append(dev)

    if not data:
        print(f"[skip] CV plot for {metric_col}: no data after filtering")
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    ax.boxplot(data, widths=0.18, showfliers=False)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = (rng.random(len(vals)) - 0.5) * 0.10
        ax.scatter(np.ones(len(vals)) * i + jitter, vals, s=14, alpha=0.55, color="0.2")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=15)

    # ✅ “conditions” 대신 “environments”로 표현 통일
    ax.set_ylabel(ylabel, fontsize=16)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

def plot_environment_participant_distribution(cell_df, metric_col, ylabel, out_png,
                                              env_col="ConditionNr"):
    """
    x축=Environment(C1–C9), y축=metric.
    participant–environment 값(필요시 device 평균 포함)의 분포를 시각화.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    tmp = cell_df.copy()

    # participant–environment 값 만들기 (device가 여러 개면 평균으로 접힘)
    pe_df = (
        tmp.groupby(["ParticipantID", env_col])[metric_col]
           .mean()
           .reset_index()
    )

    envs = sorted(pe_df[env_col].unique())
    data = [pe_df.loc[pe_df[env_col] == e, metric_col].dropna().values for e in envs]

    # ✅ figsize/rotation/pretty labels 적용
    fig, ax = plt.subplots(figsize=figsize_for(env_col))

    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    ax.boxplot(data, widths=0.18, showfliers=False)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = (rng.random(len(vals)) - 0.5) * 0.12
        ax.scatter(np.ones(len(vals))*i + jitter, vals, s=14, alpha=0.6, color="0.2")

    ax.set_xticks(range(1, len(envs) + 1))
    ax.set_xticklabels(pretty_xticklabels(env_col, envs),
                       rotation=rotation_for(env_col),
                       ha="right",
                       fontsize=12)

    ax.set_xlabel("", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

def plot_duration_by_device_avg_texts(df, out_png,
                                      duration_col="Duration",
                                      text_col="TextID",
                                      pid_col="ParticipantID",
                                      dev_col="DeviceStd"):
    """
    Recording duration:
    - average across texts (per participant × device)
    - do NOT average across devices
    Result: one value per participant × device (mean duration across 5 texts)
    Plot: x=device, y=duration
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) participant × device × text 단위로 먼저 평균(혹시 반복 측정이 있으면 정리)
    d1 = (
        df.groupby([pid_col, dev_col, text_col])[duration_col]
          .mean()
          .reset_index()
    )

    # 2) participant × device 단위로 texts 평균 (핵심!)
    d2 = (
        d1.groupby([pid_col, dev_col])[duration_col]
          .mean()
          .reset_index()
          .rename(columns={duration_col: "Duration_mean_texts"})
    )

    # device 순서 고정
    device_order = ["Smartphone", "Laptop", "Headphones", "AirPods"]
    present = [d for d in device_order if d in set(d2[dev_col])]

    data = [d2.loc[d2[dev_col] == d, "Duration_mean_texts"].values for d in present]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    ax.boxplot(data, widths=0.18, showfliers=False)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = (rng.random(len(vals)) - 0.5) * 0.10
        ax.scatter(np.ones(len(vals))*i + jitter, vals, s=14, alpha=0.6, color="0.2")

    ax.set_xticks(range(1, len(present)+1))
    ax.set_xticklabels(present, fontsize=15)
    ax.set_xlabel("Device", fontsize=16)
    ax.set_ylabel("Mean duration across texts (s)", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")




def plot_duration_by_text_across_participants(df, out_png,
                                              duration_col="Duration",
                                              text_col="TextID",
                                              pid_col="ParticipantID",
                                              agg="median"):
    """
    문장 기준만:
    - x축: 문장(TextID)
    - y축: Duration
    - 한 점: participant × text (그 참가자의 해당 문장 duration을 하나로 요약)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    tmp = df.copy()

    # check columns
    for c in [pid_col, text_col, duration_col]:
        if c not in tmp.columns:
            raise KeyError(f"Missing column: {c}")

    tmp[duration_col] = pd.to_numeric(tmp[duration_col], errors="coerce")
    tmp = tmp.dropna(subset=[pid_col, text_col, duration_col]).copy()

    # participant × text -> 1 value
    if agg == "mean":
        d = tmp.groupby([pid_col, text_col], observed=False)[duration_col].mean().reset_index()
    else:
        d = tmp.groupby([pid_col, text_col], observed=False)[duration_col].median().reset_index()

    texts = sorted(d[text_col].unique())
    data_all = [d.loc[d[text_col] == t, duration_col].dropna().values for t in texts]

    if all(len(v) == 0 for v in data_all):
        raise RuntimeError("No duration data after filtering/aggregation.")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    # violin (>=2) + box (>=1) + points
    pos_vio  = [i for i, v in enumerate(data_all, start=1) if len(v) >= 2]
    data_vio = [v for v in data_all if len(v) >= 2]
    pos_box  = [i for i, v in enumerate(data_all, start=1) if len(v) >= 1]
    data_box = [v for v in data_all if len(v) >= 1]

    if data_vio:
        ax.violinplot(data_vio, positions=pos_vio,
                      showmeans=False, showmedians=True, showextrema=True)
    if data_box:
        ax.boxplot(data_box, positions=pos_box, widths=0.18, showfliers=False)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data_all, start=1):
        if len(vals) == 0:
            continue
        jitter = (rng.random(len(vals)) - 0.5) * 0.12
        ax.scatter(np.ones(len(vals))*i + jitter, vals, s=14, alpha=0.6, color="0.2")

    ax.set_xticks(range(1, len(texts)+1))
    ax.set_xticklabels([f"S{t}" for t in texts], fontsize=15)
    ax.set_xlabel("Text", fontsize=15)
    ax.set_ylabel("Duration", fontsize=16)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")


def plot_duration_spaghetti_by_text(df, out_png,
                                    duration_col="Duration",
                                    text_col="TextID",
                                    pid_col="ParticipantID",
                                    agg="median",
                                    normalize=None):
    """
    문장 기준 스파게티 플롯
    - x축: TextID(문장)
    - y축: Duration (또는 정규화된 Duration)
    - 얇은 선: 참가자별 궤적
    - 굵은 선: 전체 중앙값(문장별)

    normalize:
      None         : raw duration
      "ratio"      : duration / participant_mean(duration)
      "centered"   : duration - participant_mean(duration)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    tmp = df.copy()
    for c in [pid_col, text_col, duration_col]:
        if c not in tmp.columns:
            raise KeyError(f"Missing column: {c}")

    tmp[duration_col] = pd.to_numeric(tmp[duration_col], errors="coerce")
    tmp = tmp.dropna(subset=[pid_col, text_col, duration_col]).copy()

    # participant × text -> 1 value (median/mean)
    g = tmp.groupby([pid_col, text_col], observed=False)[duration_col]
    if agg == "mean":
        d = g.mean().reset_index()
    else:
        d = g.median().reset_index()

    # normalize per participant (baseline 제거)
    if normalize is not None:
        pm = d.groupby(pid_col, observed=False)[duration_col].mean()
        d = d.join(pm, on=pid_col, rsuffix="_pmean")
        if normalize == "ratio":
            d[duration_col] = d[duration_col] / d[f"{duration_col}_pmean"]
        elif normalize == "centered":
            d[duration_col] = d[duration_col] - d[f"{duration_col}_pmean"]
        else:
            raise ValueError("normalize must be one of: None, 'ratio', 'centered'")

    # x 순서 고정
    texts = sorted(d[text_col].unique())
    x_map = {t: i for i, t in enumerate(texts, start=1)}

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # 참가자별 선
    for pid, sub in d.groupby(pid_col, observed=False):
        y = [np.nan] * len(texts)
        for _, row in sub.iterrows():
            y[texts.index(row[text_col])] = row[duration_col]
        x = list(range(1, len(texts) + 1))
        ax.plot(x, y, marker="o", linewidth=0.2, alpha=0.35, color="gray")


    # 전체 중앙값(굵은 선)
    med = d.groupby(text_col, observed=False)[duration_col].median().reindex(texts)
    ax.plot(range(1, len(texts) + 1), med.values, marker="o",
            linewidth=3.0, alpha=1.0, color="black")

    ax.set_xticks(range(1, len(texts) + 1))
    ax.set_xticklabels([f"S{t}" for t in texts], fontsize=15)

    ax.set_xlabel("TextID", fontsize=14)
    if normalize == "ratio":
        ax.set_ylabel("Duration", fontsize=16)
    elif normalize == "centered":
        ax.set_ylabel("Duration - participant-mean(Duration) (s)", fontsize=16)
    else:
        ax.set_ylabel("Duration", fontsize=16)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_png}")

# =========================
# Load + harmonize
# =========================
df = pd.read_csv(IN_CSV)

# 컬럼명 통일
rename_map = {}
if "ActiveSpeechLevel" in df.columns:
    rename_map["ActiveSpeechLevel"] = "ASL"

# ✅ Speech Percentage → Speech Activity Factor로 통일
if "Percentage of Speech" in df.columns:
    rename_map["Percentage of Speech"] = "Speech Activity Factor"
if "SpeechPercentage" in df.columns:
    rename_map["SpeechPercentage"] = "Speech Activity Factor"
if "Speech Percentage" in df.columns:
    rename_map["Speech Percentage"] = "Speech Activity Factor"

df = df.rename(columns=rename_map)

# SAF가 0~1이면 %로 변환 (표현은 activity factor로 유지)
if "Speech Activity Factor" in df.columns:
    saf = pd.to_numeric(df["Speech Activity Factor"], errors="coerce")
    if saf.dropna().max() <= 1.1:
        df["Speech Activity Factor"] = saf * 100.0

# Device 표준화
if "Device" not in df.columns:
    raise KeyError("analysis_results.csv에 'Device' 컬럼이 없어. 컬럼명을 확인해줘.")
df["DeviceStd"] = df["Device"].apply(normalize_device_4)
df = df[df["DeviceStd"].isin(DEVICE_ORDER_4)].copy()
df["DeviceStd"] = pd.Categorical(df["DeviceStd"], categories=DEVICE_ORDER_4, ordered=True)

# --- make sure IDs exist in the MAIN df ---
if "ParticipantID" not in df.columns:
    df["ParticipantID"] = df.apply(extract_participant_id, axis=1)
    df = df[df["ParticipantID"] != "Unknown"].copy()

if "TextID" not in df.columns:
    df["TextID"] = df.apply(extract_sentence_nr, axis=1)  # from file name or SentenceNr
    df["TextID"] = pd.to_numeric(df["TextID"], errors="coerce")
    df = df.dropna(subset=["TextID"]).copy()
    df["TextID"] = df["TextID"].astype(int)

# 필요 metric만 numeric으로 정리
metric_cols = [m for m, _ in METRICS if m in df.columns]
if not metric_cols:
    raise RuntimeError("필요한 metric 컬럼을 하나도 못 찾았어. 컬럼명을 확인해줘.")

for col in metric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================
# IDs needed for duration-by-text plot
# =========================
if "ParticipantID" not in df.columns:
    df["ParticipantID"] = df.apply(extract_participant_id, axis=1)
    df = df[df["ParticipantID"] != "Unknown"].copy()

if "TextID" not in df.columns:
    df["TextID"] = df.apply(extract_sentence_nr, axis=1)
    df["TextID"] = pd.to_numeric(df["TextID"], errors="coerce")
    df = df.dropna(subset=["TextID"]).copy()
    df["TextID"] = df["TextID"].astype(int)

# =========================
# Sentence-first aggregation
# =========================
cell_df = build_sentence_aggregated(df, metric_cols)

# =========================
# Generate plots
# =========================
for metric_col, ylabel in METRICS:
    if metric_col not in cell_df.columns:
        print(f"[skip] '{metric_col}' column not found after aggregation")
        continue

    out_sp = f"{OUT_DIR}/Fig_spaghetti_device_{metric_col.replace(' ', '_')}.png"
    out_cv = f"{OUT_DIR}/Fig_withinCV_{metric_col.replace(' ', '_')}.png"
    out_delta = f"{OUT_DIR}/Fig_delta_vsSmartphone_{metric_col.replace(' ', '_')}.png"

    # ✅ (1) 스파게티도 생성
    plot_spaghetti_by_device(cell_df, metric_col, ylabel, out_sp)

    # ✅ (2) Δ plot
    plot_delta_vs_baseline(cell_df, metric_col, ylabel, out_delta, baseline="Smartphone")

    # ✅ (3) CV plot
    plot_within_participant_cv(cell_df, metric_col, ylabel, out_cv)

out_png = f"{OUT_DIR}/Fig_duration_byDevice_meanAcrossTexts.png"
plot_duration_by_device_avg_texts(df, out_png,
                                      duration_col="Duration",
                                      text_col="TextID",
                                      pid_col="ParticipantID",
                                      dev_col="DeviceStd")


for metric_col, ylabel in METRICS:

    out_env = f"{OUT_DIR}/Fig_envDist_{metric_col.replace(' ', '_')}.png"

    plot_environment_participant_distribution(
        cell_df,
        metric_col,
        ylabel,
        out_env
    )
out_dur = f"{OUT_DIR}/Fig_duration_byText_acrossParticipants.png"
plot_duration_by_text_across_participants(
    df, out_dur,
    duration_col="Duration",
    text_col="TextID",
    pid_col="ParticipantID",
    agg="median"
)

out1 = f"{OUT_DIR}/Fig_duration_spaghetti_byText_ratio.png"
plot_duration_spaghetti_by_text(df, out1,
                                duration_col="Duration",
                                text_col="TextID",
                                pid_col="ParticipantID",
                                agg="median",
                                normalize="ratio")

print("Done.")