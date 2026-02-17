import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Input CSV paths
# =========================
DEV_CSV  = "bandwidth_device_supervised_DETAILED.csv"
COND_CSV = "bandwidth_condition_supervised_only.csv"

dev = pd.read_csv(DEV_CSV)
cond = pd.read_csv(COND_CSV)

# =========================
# Common cleanup
# =========================
dev["Device"] = dev["Device"].astype(str).str.strip()
dev["Bandwidth"] = dev["Bandwidth"].astype(str).str.strip()
cond["Bandwidth"] = cond["Bandwidth"].astype(str).str.strip()

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

def normalize_bw(v):
    s = str(v).strip().upper()
    if s in ["EMPTY", "EMPTY ", "NONE", "NA", "N/A", ""]:
        return "EMPTY"
    if s in ["NB", "WB", "SWB", "FB"]:
        return s
    return s

dev["Bandwidth"]  = dev["Bandwidth"].map(normalize_bw)
cond["Bandwidth"] = cond["Bandwidth"].map(normalize_bw)

# Bandwidth order (include "empty" only if it exists)
bw_order_full = ["EMPTY", "NB", "WB", "SWB", "FB"]
present_bw = pd.concat([dev["Bandwidth"], cond["Bandwidth"]], ignore_index=True).dropna().unique().tolist()
present_bw = set([str(x).strip().upper() for x in present_bw])

# keep only categories that actually appear
bw_order = [b for b in bw_order_full if b in present_bw]
if not bw_order:
    bw_order = ["NB", "WB", "SWB", "FB"]

# =========================
# 1) Device plot: 100% stacked bar (Supervised detailed)
# =========================
device_order = ["Smartphone", "Laptop", "Headset (Sennheiser PXC 550)", "AirPods"]

tab_dev = (dev.pivot_table(index="Device", columns="Bandwidth", values="Supervised (%)", fill_value=0)
             .reindex(columns=bw_order, fill_value=0))

empty_dev = None
if "EMPTY" in tab_dev.columns:
    empty_dev = tab_dev["EMPTY"].copy()
    empty_dev.to_csv("BW_EMPTY_share_by_device.csv", header=["EMPTY (%)"])

DROP_EMPTY_DEV = True
if DROP_EMPTY_DEV and "EMPTY" in tab_dev.columns:
    # ✅ 2) 그림/분석용 분포에서는 EMPTY 제외
    tab_dev = tab_dev.drop(columns=["EMPTY"], errors="ignore")

    # ✅ 3) NB/WB/SWB/FB만 남았으니 다시 100%가 되도록 재정규화
    row_sum = tab_dev.sum(axis=1).replace(0, np.nan)
    tab_dev = (tab_dev.div(row_sum, axis=0) * 100).fillna(0)

# (선택) 검증: 각 디바이스에서 합이 100인지 확인
print("Device row sums (should be 100):")
print(tab_dev.sum(axis=1))
# use this order for plotting/legend (EMPTY excluded)
bw_order_dev = [b for b in bw_order if b != "EMPTY"]

# keep a nice order (only devices that exist in the CSV)
present_devices = [d for d in device_order if d in tab_dev.index]
if present_devices:
    tab_dev = tab_dev.reindex(present_devices)
else:
    tab_dev = tab_dev.sort_index()

x = np.arange(len(tab_dev.index))
bottom = np.zeros(len(tab_dev.index))

fig, ax = plt.subplots(figsize=(7.6, 4.2))
for bw in bw_order_dev:
    vals = tab_dev[bw].values
    ax.bar(x, vals, bottom=bottom, label=bw)
    bottom += vals

ax.set_xticks(x)
label_map = {
    "Smartphone": "Smartphone",
    "Laptop": "Laptop",
    "Headset (Sennheiser PXC 550)": "Headphones",
    "AirPods": "AirPods",
}

labels = [label_map.get(d, d) for d in tab_dev.index.astype(str)]
ax.set_xticklabels(labels, rotation=0, fontsize=16)
ax.set_ylabel("Share", fontsize=18)
ax.set_xlabel("")
ax.tick_params(axis="y", labelsize=16)
ax.set_ylim(0, 100)
ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(
    title="Bandwidth",
    bbox_to_anchor=(1.02, 1),   # ✅ 오른쪽 바깥
    loc="upper left",
    borderaxespad=0,
    fontsize=12
)

fig.tight_layout()
fig.savefig("BW_device_stacked_supervised_DETAILED.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved: BW_device_stacked_supervised_DETAILED.png")

# =========================
# 2) Condition plot: heatmap (Supervised only)
# =========================
# Ensure Condition is numeric for sorting
cond["Condition"] = pd.to_numeric(cond["Condition"], errors="coerce")

# (추천) empty는 거의 0이면 빼는 게 훨씬 보기 좋음
DROP_EMPTY = True
bw_order_cond = bw_order.copy()
if DROP_EMPTY and "EMPTY" in bw_order_cond:
    bw_order_cond.remove("EMPTY")

tab_cond = (cond.pivot_table(index="Condition", columns="Bandwidth", values="Supervised (%)", fill_value=0)
              .reindex(columns=bw_order_cond, fill_value=0)
              .sort_index())

# (추천) empty는 거의 0이면 빼는 게 훨씬 보기 좋음
DROP_EMPTY = True

# ✅ 1) pivot을 "전체(bw_order 포함)"로 한번 만들고 EMPTY% 저장
tab_cond_full = (cond.pivot_table(index="Condition", columns="Bandwidth", values="Supervised (%)", fill_value=0)
                   .reindex(columns=bw_order, fill_value=0)
                   .sort_index())

if "EMPTY" in tab_cond_full.columns:
    tab_cond_full["EMPTY"].to_csv("BW_EMPTY_share_by_condition.csv", header=["EMPTY (%)"])

# ✅ 2) 그 다음 그림용은 EMPTY 제외
bw_order_cond = [b for b in bw_order if b != "EMPTY"] if DROP_EMPTY else bw_order.copy()

tab_cond = tab_cond_full.reindex(columns=bw_order_cond, fill_value=0)

# ✅ 3) NB/WB/SWB/FB만 남았으니 다시 100으로 정규화
row_sum = tab_cond.sum(axis=1).replace(0, np.nan)
tab_cond = (tab_cond.div(row_sum, axis=0) * 100).fillna(0)

# (선택) 검증
print("Condition row sums (should be 100):")
print(tab_cond.sum(axis=1).head())


# 혹시 합이 100이 아닌 경우(반올림/누락 등) 보기 좋게 100으로 정규화
row_sum = tab_cond.sum(axis=1).replace(0, np.nan)
tab_cond = (tab_cond.div(row_sum, axis=0) * 100).fillna(0)

y = np.arange(len(tab_cond.index))
left = np.zeros(len(tab_cond.index))

fig, ax = plt.subplots(figsize=(9.2, 5.2))

for bw in bw_order_cond:
    vals = tab_cond[bw].values
    ax.barh(y, vals, left=left, label=bw)
    left += vals

# y축 라벨: Condition 번호
ax.set_yticks(y)
cond_labels = [COND_SHORT_NR.get(int(c), str(int(c))) for c in tab_cond.index]
ax.set_yticklabels(cond_labels, fontsize=16)

ax.set_xlabel("Share", fontsize=14, labelpad=10)
ax.set_ylabel("")
ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(
    title="Bandwidth",
    bbox_to_anchor=(1.02, 1),   # ✅ 오른쪽 바깥
    loc="upper left",
    borderaxespad=0,
    fontsize=11
)

fig.tight_layout()
fig.savefig("BW_condition_stackedbar_supervised_only.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("saved: BW_condition_stackedbar_supervised_only.png")