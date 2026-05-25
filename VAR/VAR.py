import warnings
warnings.filterwarnings("ignore")

import os
import yfinance as yf
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR

result_path = __file__.replace("VAR.py", "VAR_results/")
os.makedirs(result_path, exist_ok=True)

# load data from Yahoo Finance
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

# 轉成漲跌率 log return
return_data = np.log(price / price.shift(1)).dropna()

return_data.to_csv(
    result_path + "return_data_2016_2025.csv",
    encoding="utf-8-sig"
)

# ADF / KPSS test for stationarity
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

print("\n=== Stationarity Test: Log Return ===")
print(stationarity_result)

# if not stationary, apply first difference and test again
if stationarity_result["Both_Stationary"].all():
    var_data = return_data.copy()
    data_used = "Log Return"

else:
    print("\nSome variables are still non-stationary. Applying first difference...")

    var_data = return_data.diff().dropna()
    data_used = "Differenced Log Return"

    stationarity_diff_result = stationarity_table(var_data)

    stationarity_diff_result.to_csv(
        result_path + "stationarity_test_after_diff_2016_2025.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== Stationarity Test After Differencing ===")
    print(stationarity_diff_result)

var_data.to_csv(
    result_path + "var_data_used_2016_2025.csv",
    encoding="utf-8-sig"
)

print(f"\nData used for VAR: {data_used}")

# VAR Lag Selection
model = VAR(var_data)

lag_selection = model.select_order(maxlags=20)

print("\n=== VAR Lag Selection Summary ===")
print(lag_selection.summary())

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

print("\n=== Lag Selection Table ===")
print(lag_table)

lag_selection = model.select_order(maxlags=10)
selected_lag = lag_selection.selected_orders["bic"]

if selected_lag is None or selected_lag < 1:
    selected_lag = 1

print(f"\nSelected Lag by BIC: {selected_lag}")

# Fit VAR Model
var_model = model.fit(selected_lag)

with open(result_path + "var_model_summary.txt", "w", encoding="utf-8") as f:
    f.write(str(var_model.summary()))

print("\n=== VAR Model Summary ===")
print(var_model.summary())

# bidirectional Granger Causality Test
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

print("\n=== Bidirectional Granger Causality Result ===")
print(granger_result)

# 外部因子 -> 台塑四寶
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

macro_to_formosa = granger_result[
    (granger_result["X_causing"].isin(macro_factors)) &
    (granger_result["Y_caused"].isin(formosa_companies))
].sort_values("Min_p_value")

macro_to_formosa.to_csv(
    result_path + "macro_to_formosa_2016_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

# 台塑四寶 -> 外部因子
formosa_to_macro = granger_result[
    (granger_result["X_causing"].isin(formosa_companies)) &
    (granger_result["Y_caused"].isin(macro_factors))
].sort_values("Min_p_value")

formosa_to_macro.to_csv(
    result_path + "formosa_to_macro_2016_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

# 雙向關係整理表
pair_results = []

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

print("\n=== Bidirectional Summary ===")
print(bidirectional_summary)

print("\n=== Analysis Completed ===")
print(f"Results saved to: {result_path}")