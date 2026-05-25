import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = __file__.replace(
    "Financial_statements.py",
    "南亞_all.csv"
)

result_path = __file__.replace(
    "Financial_statements.py",
    "result/"
)

os.makedirs(result_path, exist_ok=True)

chart_path = f"{result_path}financial_charts/"
os.makedirs(chart_path, exist_ok=True)

#load data
df = pd.read_csv(csv_path, encoding="cp950")

df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

df["公司"] = df["證券代碼"].astype(str).str.split(" ", n=1).str[1]
df["年份"] = pd.to_numeric(df["年月"], errors="coerce") // 100

df = df.sort_values(["公司", "年份"])

text_columns = [
    "證券代碼",
    "公司",
    "幣別",
    "合併(Y/N)",
    "市場別"
]

df = df.replace(
    ["--", "N/A", "", "nan", "NaN"],
    np.nan
)

for col in df.columns:
    if col not in text_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

def safe_divide(a, b):
    return np.where(
        (b != 0) & (~pd.isna(b)),
        a / b,
        np.nan
    )

#avg_item
df["Avg_Total_Assets"] = (
    df.groupby("公司")["資產總額"].shift(1) + df["資產總額"]
) / 2

df["Avg_Equity"] = (
    df.groupby("公司")["股東權益總額"].shift(1) + df["股東權益總額"]
) / 2

df["Avg_Inventory"] = (
    df.groupby("公司")["  存貨"].shift(1) + df["  存貨"]
) / 2

df["Avg_AR"] = (
    df.groupby("公司")["  應收帳款及票據"].shift(1) + df["  應收帳款及票據"]
) / 2

df["Avg_Total_Assets"] = df["Avg_Total_Assets"].fillna(df["資產總額"])
df["Avg_Equity"] = df["Avg_Equity"].fillna(df["股東權益總額"])
df["Avg_Inventory"] = df["Avg_Inventory"].fillna(df["  存貨"])
df["Avg_AR"] = df["Avg_AR"].fillna(df["  應收帳款及票據"])

#profitability
df["Gross_Margin"] = safe_divide(df["營業毛利"], df["營業收入淨額"]) * 100
df["Operating_Margin"] = safe_divide(df["營業利益"], df["營業收入淨額"]) * 100
df["Net_Margin"] = safe_divide(df["歸屬母公司淨利（損）"], df["營業收入淨額"]) * 100
df["ROA"] = safe_divide(df["歸屬母公司淨利（損）"], df["Avg_Total_Assets"]) * 100
df["ROE"] = safe_divide(df["歸屬母公司淨利（損）"], df["Avg_Equity"]) * 100
df["EPS"] = pd.to_numeric(df["每股盈餘"], errors="coerce")

#solvency
df["Current_Ratio"] = safe_divide(df["流動資產"], df["流動負債"])
df["Quick_Ratio"] = safe_divide(df["流動資產"] - df["  存貨"], df["流動負債"])
df["Debt_Ratio"] = safe_divide(df["負債總額"], df["資產總額"]) * 100
df["Equity_Ratio"] = safe_divide(df["股東權益總額"], df["資產總額"]) * 100
df["Interest_Coverage"] = safe_divide(df["稅前息前淨利"], df["財務成本"])

# operating efficiency
df["Inventory_Turnover"] = safe_divide(df["營業成本"], df["Avg_Inventory"])
df["AR_Turnover"] = safe_divide(df["營業收入淨額"], df["Avg_AR"])
df["Asset_Turnover"] = safe_divide(df["營業收入淨額"], df["Avg_Total_Assets"])

# Cash flow
df["OCF_Ratio"] = safe_divide(df["來自營運之現金流量"], df["流動負債"])

df["Free_Cash_Flow"] = (
    df["來自營運之現金流量"]
    + df["  購置不動產廠房設備（含預付）－CFI"]
)

# Growth
df["Revenue_Growth"] = (
    df.groupby("公司")["營業收入淨額"]
    .pct_change()
) * 100

df["EPS_Growth"] = (
    df.groupby("公司")["EPS"]
    .pct_change()
) * 100

# Market valuation
df["Market_Cap"] = pd.to_numeric(df["季底普通股市值"], errors="coerce")
df["PE"] = pd.to_numeric(df["當季季底P/E"], errors="coerce")
df["PB"] = pd.to_numeric(df["當季季底P/B"], errors="coerce")
df["Dividend_Yield"] = pd.to_numeric(df["股利殖利率"], errors="coerce")
df["BVPS"] = pd.to_numeric(df["每股淨值(B)"], errors="coerce")

# Altman Z-score
df["X1"] = safe_divide(
    df["流動資產"] - df["流動負債"],
    df["資產總額"]
)

df["X2"] = safe_divide(
    df["  保留盈餘"],
    df["資產總額"]
)

df["X3"] = safe_divide(
    df["稅前息前淨利"],
    df["資產總額"]
)

df["X4"] = safe_divide(
    df["Market_Cap"],
    df["負債總額"]
)

df["X5"] = safe_divide(
    df["營業收入淨額"],
    df["資產總額"]
)

df["Altman_Zscore"] = (
    1.2 * df["X1"]
    + 1.4 * df["X2"]
    + 3.3 * df["X3"]
    + 0.6 * df["X4"]
    + 1.0 * df["X5"]
)

# Industry average
numeric_cols = (
    df.select_dtypes(include=[np.number])
    .columns
    .drop("年份")
)

industry_avg = (
    df.groupby("年份")[numeric_cols]
    .mean()
    .reset_index()
)

industry_avg["公司"] = "Industry_Average"

df_with_industry = pd.concat(
    [df, industry_avg],
    ignore_index=True
)

# Export CSV
df.to_csv(
    f"{result_path}financial_ratio_result.csv",
    index=False,
    encoding="utf-8-sig"
)

df_with_industry.to_csv(
    f"{result_path}financial_ratio_with_industry.csv",
    index=False,
    encoding="utf-8-sig"
)

# Ratio-only dataframe
ratio_columns = [
    "公司",
    "年份",

    # Profitability
    "Gross_Margin",
    "Operating_Margin",
    "Net_Margin",
    "ROA",
    "ROE",
    "EPS",

    # Solvency
    "Current_Ratio",
    "Quick_Ratio",
    "Debt_Ratio",
    "Equity_Ratio",
    "Interest_Coverage",

    # Efficiency
    "Inventory_Turnover",
    "AR_Turnover",
    "Asset_Turnover",

    # Cash flow
    "OCF_Ratio",
    "Free_Cash_Flow",

    # Growth
    "Revenue_Growth",
    "EPS_Growth",

    # Valuation
    "Market_Cap",
    "PE",
    "PB",
    "Dividend_Yield",
    "BVPS",

    # Altman
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "Altman_Zscore"
]

ratio_df = df[ratio_columns].copy()

ratio_df.to_csv(
    f"{result_path}financial_ratio_only.csv",
    index=False,
    encoding="utf-8-sig"
)

# Plot settings
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

metrics = [
    "Gross_Margin",
    "Operating_Margin",
    "Net_Margin",
    "ROA",
    "ROE",
    "EPS",
    "Debt_Ratio",
    "Current_Ratio",
    "Quick_Ratio",
    "OCF_Ratio",
    "Free_Cash_Flow",
    "Revenue_Growth",
    "EPS_Growth",
    "PE",
    "PB",
    "Dividend_Yield",
    "Altman_Zscore"
]

compare_metrics = [
    "Gross_Margin",
    "Operating_Margin",
    "Net_Margin",
    "ROE",
    "ROA",
    "EPS",
    "Debt_Ratio",
    "Current_Ratio",
    "Free_Cash_Flow",
    "PE",
    "PB",
    "Altman_Zscore"
]

companies = df["公司"].dropna().unique()

# Company comparison charts
for metric in compare_metrics:

    plt.figure(figsize=(12, 6))
    plotted = False

    for company in companies:

        company_df = df[df["公司"] == company].sort_values("年份")

        if metric not in company_df.columns:
            continue

        if company_df[metric].isna().all():
            continue

        plt.plot(
            company_df["年份"],
            company_df[metric],
            marker="o",
            label=company
        )

        plotted = True

    if plotted:
        plt.title(f"Company Comparison - {metric}")
        plt.xlabel("Year")
        plt.ylabel(metric)

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=2,
            borderaxespad=0,
            fontsize=9
        )

        plt.grid(True)

        plt.savefig(
            f"{chart_path}Compare_{metric}.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close()

# 四寶 vs Industry Average
target_companies = [
    "台塑",
    "南亞",
    "台化",
    "Industry_Average"
]

for metric in compare_metrics:

    plt.figure(figsize=(12, 6))

    plotted = False

    for company in target_companies:

        temp_df = (
            df_with_industry[
                df_with_industry["公司"] == company
            ]
            .sort_values("年份")
        )

        if temp_df.empty:
            continue

        if metric not in temp_df.columns:
            continue

        if temp_df[metric].isna().all():
            continue

        plt.plot(
            temp_df["年份"],
            temp_df[metric],
            marker="o",
            label=company
        )

        plotted = True

    if plotted:

        plt.title(
            f"Formosa Plastics Group vs Industry Average - {metric}"
        )

        plt.xlabel("Year")
        plt.ylabel(metric)

        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=2,
            borderaxespad=0,
            fontsize=9
        )

        plt.grid(True)

        plt.savefig(
            f"{chart_path}FPG_vs_Industry_{metric}.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close()

# Altman Z-score risk charts
for company in companies:

    company_df = df[df["公司"] == company].sort_values("年份")

    if company_df["Altman_Zscore"].isna().all():
        continue

    plt.figure(figsize=(8, 5))

    plt.plot(
        company_df["年份"],
        company_df["Altman_Zscore"],
        marker="o",
        label="Altman Z-score"
    )

    plt.axhline(y=1.8, linestyle="--", label="Distress Zone: 1.8", color="red")
    plt.axhline(y=3.0, linestyle="--", label="Safe Zone: 3.0", color="green")

    plt.title(f"{company} - Altman Z-score")
    plt.xlabel("Year")
    plt.ylabel("Z-score")
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=1,
        borderaxespad=0,
        fontsize=9
    )
    plt.grid(True)

    plt.savefig(
        f"{chart_path}{company}_Altman_Zscore_Risk.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Finished!")
print(f"Results saved to: {result_path}")