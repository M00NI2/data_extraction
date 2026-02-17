"""
average.py

1) statistics and mean values across all supervised recordings
2) statistics and mean values per 9 Conditions
4) statistics and mean values per recording device(Earbuds, Headset, Laptop, Smartphone)

The resultings are saved as CSV files.
"""
import pandas as pd

df = pd.read_csv("analysis_results.csv")

#List of metrics to be analysed.
data_list = ["ASL", "Percentage of Speech", "Duration", "SNR"]

# Define rounding rules
rounding_rules = {
    "ASL": 3,
    "Percentage of Speech": 2,
    "Duration": 3,
    "SNR": 2
}

# Rounding helper
def apply_rounding(df_stats):
    for col in df_stats.columns:
        if col in rounding_rules:
            df_stats[col] = df_stats[col].round(rounding_rules[col])
    return df_stats


####################
##### descriptive statistics and means across all recordings
####################
#statistics = df[data_list].describe()
#print(statistics)
statistics_mean = df[data_list].mean()
statistics_mean.to_frame(name="Mean")
#print(df[data_list].mean())
#statistics.to_csv("statistics.csv")
statistics_mean.to_csv("statistics_mean.csv")
print("saved_statistics")


####################
##### statistics and means across all recordings per recording device
####################
#device_statistics = df.groupby("Device")[data_list].describe()
#print(device_statistics)
device_statistics_mean = df.groupby("Device")[data_list].mean()
#print(df.groupby("Device")[data_list].mean())
#device_statistics.to_csv("device_statistics.csv")
device_statistics_mean.to_csv("device_statistics_mean.csv")
print("saved_device")



####################
##### statistics and means across all recordings per condition
####################
#condition_statistics = df.groupby("Condition")[data_list].describe()
#print(condition_statistics)
condition_statistics_mean = df.groupby("Condition")[data_list].mean()
#print(df.groupby("Condition")[data_list].mean())
#print(df.groupby("Condition")[data_list].mean())
#condition_statistics.to_csv("condition_statistics.csv")
condition_statistics_mean.to_csv("condition_statistics_mean.csv")
print("saved_condition")


# =========================================================
# Bandwidth distribution (overall / by Device / by Condition)
# =========================================================
categories = ["empty", "NB", "WB", "SWB", "FB"]

# ---------------------------------------------------------
# (1) Overall bandwidth distribution
# ---------------------------------------------------------
bw_percent = df["Bandwidth"].value_counts(normalize=True) * 100

overall_result = []
for c in categories:
    overall_result.append(bw_percent[c] if c in bw_percent else 0)

bandwidth_overall = pd.DataFrame({
    "Bandwidth": categories,
    "Percentage (%)": overall_result
}).round(2)

bandwidth_overall.to_csv("bandwidth_overall_supervised.csv", index=False)
print("saved_bandwidth_overall_supervised")

# ---------------------------------------------------------
# (2) Device-level bandwidth distribution
# ---------------------------------------------------------
devices = sorted(df["Device"].dropna().unique())

result_bandwidth = []
result_device = []
result_percent = []

for d in devices:
    single = df[df["Device"] == d]
    bw_dev_percent = single["Bandwidth"].value_counts(normalize=True) * 100

    for c in categories:
        v = bw_dev_percent[c] if c in bw_dev_percent else 0
        result_device.append(d)
        result_bandwidth.append(c)
        result_percent.append(v)

bandwidth_device = pd.DataFrame({
    "Device": result_device,
    "Bandwidth": result_bandwidth,
    "Percentage (%)": result_percent
}).round(2)

bandwidth_device.to_csv("bandwidth_device_supervised.csv", index=False)
print("saved_bandwidth_device_supervised")

# ---------------------------------------------------------
# (3) Condition-level bandwidth distribution
# ---------------------------------------------------------
conditions = sorted(df["Condition"].dropna().unique())

result_condition = []
result_condition_bandwidth = []
result_condition_percent = []

for cond in conditions:
    single = df[df["Condition"] == cond]
    bw_cond_percent = single["Bandwidth"].value_counts(normalize=True) * 100

    for c in categories:
        v = bw_cond_percent[c] if c in bw_cond_percent else 0
        result_condition.append(cond)
        result_condition_bandwidth.append(c)
        result_condition_percent.append(v)

bandwidth_condition = pd.DataFrame({
    "Condition": result_condition,
    "Bandwidth": result_condition_bandwidth,
    "Percentage (%)": result_condition_percent
}).round(2)

bandwidth_condition.to_csv("bandwidth_condition_supervised.csv", index=False)
print("saved_bandwidth_condition_supervised")