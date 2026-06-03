import warnings
warnings.filterwarnings("ignore")

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR

try:
    result_path = __file__.replace("VAR.py", "VAR_results/")
except NameError:
    result_path = "./VAR_results/"

os.makedirs(result_path, exist_ok=True)

# Load data
tickers = {
    "Formosa_Plastics": "1301.TW",
    "Nan_Ya": "1303.TW",
    "Formosa_Chemicals": "1326.TW",
    "Formosa_Petrochemical": "6505.TW",

    "Brent_Crude": "BZ=F",
    "Natural_Gas": "NG=F",
    "US_Dollar_Index": "DX-Y.NYB",
    "TAIEX": "^TWII"
}

start_date = "2016-01-01"
end_date = "2025-12-31"

raw = yf.download(
    list(tickers.values()),
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=True,
    progress=False
)

price = raw["Close"].rename(columns={v: k for k, v in tickers.items()})
price = price.ffill().dropna()

price.to_csv(
    result_path + "raw_price_2016_2025.csv",
    encoding="utf-8-sig"
)

# Log return
return_data = np.log(price / price.shift(1)).dropna()

return_data.to_csv(
    result_path + "return_data_2016_2025.csv",
    encoding="utf-8-sig"
)

# ADF / KPSS test
def adf_test(series):
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1]


def kpss_test(series):
    result = kpss(series.dropna(), regression="c", nlags="auto")
    return result[1]


def stationarity_table(df):
    rows = []

    for col in df.columns:
        adf_p = adf_test(df[col])
        kpss_p = kpss_test(df[col])

        rows.append({
            "Variable": col,
            "ADF_p_value": adf_p,
            "ADF_Stationary": adf_p < 0.05,
            "KPSS_p_value": kpss_p,
            "KPSS_Stationary": kpss_p > 0.05,
            "Both_Stationary": (adf_p < 0.05) and (kpss_p > 0.05)
        })

    return pd.DataFrame(rows)


stationarity_result = stationarity_table(return_data)

stationarity_result.to_csv(
    result_path + "stationarity_test_return_2016_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

# Apply first difference
if stationarity_result["Both_Stationary"].all():
    var_data = return_data.copy()
    data_used = "Log Return"
else:
    var_data = return_data.diff().dropna()
    data_used = "Differenced Log Return"

    stationarity_diff_result = stationarity_table(var_data)

    stationarity_diff_result.to_csv(
        result_path + "stationarity_test_after_diff_2016_2025.csv",
        index=False,
        encoding="utf-8-sig"
    )

var_data.to_csv(
    result_path + "var_data_used_2016_2025.csv",
    encoding="utf-8-sig"
)

# VAR Lag Selection
model = VAR(var_data)

lag_selection = model.select_order(maxlags=20)

aic_values = lag_selection.ics["aic"]
bic_values = lag_selection.ics["bic"]
hqic_values = lag_selection.ics["hqic"]
fpe_values = lag_selection.ics["fpe"]

lag_table = pd.DataFrame({
    "Lag": range(len(aic_values)),
    "AIC": aic_values,
    "BIC": bic_values,
    "HQIC": hqic_values,
    "FPE": fpe_values
})

lag_table["Best_AIC"] = lag_table["AIC"] == lag_table["AIC"].min()
lag_table["Best_BIC"] = lag_table["BIC"] == lag_table["BIC"].min()
lag_table["Best_HQIC"] = lag_table["HQIC"] == lag_table["HQIC"].min()
lag_table["Best_FPE"] = lag_table["FPE"] == lag_table["FPE"].min()

lag_table.to_csv(
    result_path + "var_lag_selection_table_2016_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

selected_lag = lag_selection.selected_orders["bic"]

if selected_lag is None or selected_lag < 1:
    selected_lag = 1

# Fit VAR Model and Stability Test
var_model = model.fit(selected_lag)

stable = var_model.is_stable()

roots = var_model.roots

stability_table = pd.DataFrame({
    "Root": roots,
    "Modulus": np.abs(roots)
})

stability_table.to_csv(
    result_path + "VAR_stability_test.csv",
    index=False,
    encoding="utf-8-sig"
)

# Bidirectional Granger Causality Test
def run_granger_tests(df, maxlag):
    results = []
    variables = df.columns.tolist()

    for y in variables:
        for x in variables:
            if x == y:
                continue

            test_data = df[[y, x]].dropna()

            try:
                test_result = grangercausalitytests(
                    test_data,
                    maxlag=maxlag,
                    verbose=False
                )

                p_values = [
                    test_result[i][0]["ssr_ftest"][1]
                    for i in range(1, maxlag + 1)
                ]

                min_p = min(p_values)
                best_lag = p_values.index(min_p) + 1

                results.append({
                    "Direction": f"{x} -> {y}",
                    "X_causing": x,
                    "Y_caused": y,
                    "Best_Lag": best_lag,
                    "Min_p_value": min_p,
                    "Significant_5pct": min_p < 0.05
                })

            except Exception as e:
                results.append({
                    "Direction": f"{x} -> {y}",
                    "X_causing": x,
                    "Y_caused": y,
                    "Best_Lag": None,
                    "Min_p_value": None,
                    "Significant_5pct": False,
                    "Error": str(e)
                })

    return pd.DataFrame(results)


granger_result = run_granger_tests(var_data, selected_lag)
granger_result = granger_result.sort_values("Min_p_value")

granger_result.to_csv(
    result_path + "bidirectional_granger_result_2016_2025_BIC.csv",
    index=False,
    encoding="utf-8-sig"
)

# Bidirectional summary
pair_results = []

macro_factors = [
    "Brent_Crude",
    "Natural_Gas",
    "US_Dollar_Index",
    "TAIEX"
]

formosa_companies = [
    "Formosa_Plastics",
    "Nan_Ya",
    "Formosa_Chemicals",
    "Formosa_Petrochemical"
]

for factor in macro_factors:
    for company in formosa_companies:

        factor_to_company = granger_result[
            (granger_result["X_causing"] == factor) &
            (granger_result["Y_caused"] == company)
        ]

        company_to_factor = granger_result[
            (granger_result["X_causing"] == company) &
            (granger_result["Y_caused"] == factor)
        ]

        if len(factor_to_company) > 0 and len(company_to_factor) > 0:

            f2c_p = factor_to_company.iloc[0]["Min_p_value"]
            c2f_p = company_to_factor.iloc[0]["Min_p_value"]

            pair_results.append({
                "Pair": f"{factor} <-> {company}",
                "Factor_to_Company_p": f2c_p,
                "Factor_to_Company_Significant": f2c_p < 0.05,
                "Company_to_Factor_p": c2f_p,
                "Company_to_Factor_Significant": c2f_p < 0.05,
                "Relationship": (
                    "Bidirectional"
                    if (f2c_p < 0.05 and c2f_p < 0.05)
                    else "Factor -> Company"
                    if (f2c_p < 0.05 and c2f_p >= 0.05)
                    else "Company -> Factor"
                    if (f2c_p >= 0.05 and c2f_p < 0.05)
                    else "No significant causality"
                )
            })

bidirectional_summary = pd.DataFrame(pair_results)

bidirectional_summary.to_csv(
    result_path + "bidirectional_summary_2016_2025_BIC.csv",
    index=False,
    encoding="utf-8-sig"
)

# IRF
irf_periods = 20
irf = var_model.irf(irf_periods)

irf_targets = [
    "Brent_Crude",
    "Natural_Gas",
    "US_Dollar_Index",
    "TAIEX",
    "Formosa_Plastics",
    "Formosa_Chemicals",
    "Formosa_Petrochemical"
]

irf_colors = {
    "Brent_Crude": "#1f77b4",
    "Natural_Gas": "#17becf",
    "US_Dollar_Index": "#d62728",
    "TAIEX": "#9467bd",
    "Formosa_Plastics": "#ff7f0e",
    "Formosa_Chemicals": "#2ca02c",
    "Formosa_Petrochemical": "#8c564b"
}

for shock in irf_targets:
    fig = irf.plot(
        impulse=shock,
        response="Nan_Ya",
        signif=0.05
    )

    fig.set_size_inches(10, 6)

    ax = fig.axes[0]

    ax.set_title(
        f"{shock.replace('_', ' ')} Shock on Nan Ya Stock Return",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Forecast Horizon (Trading Days)",
        fontsize=12
    )

    ax.set_ylabel(
        "Response of Nan Ya Return",
        fontsize=12
    )

    ax.axhline(
        y=0,
        color="black",
        linewidth=1
    )

    ax.grid(
        alpha=0.3,
        linestyle="--"
    )

    if len(ax.lines) > 0:
        ax.lines[0].set_color(irf_colors[shock])
        ax.lines[0].set_linewidth(2.8)

    for line in ax.lines[1:]:
        line.set_color("gray")
        line.set_alpha(0.6)
        line.set_linewidth(1.5)
        line.set_linestyle("--")

    plt.tight_layout()

    fig.savefig(
        result_path + f"IRF_{shock}_to_Nan_Ya.png",
        bbox_inches="tight",
        dpi=300
    )

    plt.close(fig)

#FEVD
fevd_periods = 20
fevd = var_model.fevd(fevd_periods)

nan_ya_index = var_data.columns.get_loc("Nan_Ya")

nan_ya_fevd = pd.DataFrame(
    fevd.decomp[nan_ya_index],
    columns=var_data.columns
)

nan_ya_fevd.insert(
    0,
    "Period",
    range(1, fevd_periods + 1)
)

# Nan Ya FEVD Plot
plot_cols = [
    "Nan_Ya",
    "Formosa_Plastics",
    "Formosa_Chemicals",
    "Formosa_Petrochemical",
    "Brent_Crude",
    "US_Dollar_Index",
    "Natural_Gas",
    "TAIEX"
]

plot_colors = {
    "Nan_Ya": "#ff7f0e",                # 橘
    "Formosa_Plastics": "#1f77b4",      # 藍
    "Formosa_Chemicals": "#2ca02c",     # 綠
    "Formosa_Petrochemical": "#8c564b", # 棕
    "Brent_Crude": "#9467bd",           # 紫
    "US_Dollar_Index": "#d62728",       # 紅
    "Natural_Gas": "#17becf",           # 青
    "TAIEX": "#7f7f7f"                  # 灰
}

fig, ax = plt.subplots(figsize=(12, 6))

bottom = np.zeros(len(nan_ya_fevd))

for col in plot_cols:

    if col not in nan_ya_fevd.columns:
        continue

    ax.bar(
        nan_ya_fevd["Period"],
        nan_ya_fevd[col] * 100,
        bottom=bottom * 100,
        label=col.replace("_", " "),
        color=plot_colors[col],
        width=0.8
    )

    bottom += nan_ya_fevd[col].values

ax.set_title(
    "Forecast Error Variance Decomposition of Nan Ya",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Forecast Horizon (Trading Days)",
    fontsize=12
)

ax.set_ylabel(
    "Contribution (%)",
    fontsize=12
)

ax.set_ylim(0, 100)

ax.legend(
    title="Factors",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

fig.savefig(
    result_path + "Nan_Ya_FEVD_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)