import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ==========================
# 1. Load master data
# ==========================
@st.cache_data
def load_master_data(csv_path: str = "/data/bctc_final.csv") -> pd.DataFrame:
    if not Path(csv_path).exists():
        st.warning(f"Dataset not found at {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ==========================
# 2. Gán nhãn Default sát thực tế
# ==========================
def label_default_from_financials(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Xác định Equity âm >= 2 năm
    equity_check = (
        df.groupby("Ticker")["OWNER'S EQUITY(Bn.VND)"]
        .apply(lambda x: (x < 0).sum())
        .reset_index(name="Equity_Negative_Years")
    )
    equity_check["Default_Equity_Flag"] = (equity_check["Equity_Negative_Years"] >= 2).astype(int)

    # Xác định lỗ 3 năm liên tiếp
    df_sorted = df.sort_values(["Ticker", "Year"]).copy()
    df_sorted["is_loss"] = (df_sorted["Net Profit For the Year"] <= 0).astype(int)

    def mark_consecutive_losses(group):
        years = group['Year'].values
        losses = group['is_loss'].values
        streak = [False] * len(group)
        for i in range(len(group) - 2):
            if losses[i] and losses[i+1] and losses[i+2] and \
               years[i+1] == years[i]+1 and years[i+2] == years[i]+2:
                streak[i] = streak[i+1] = streak[i+2] = True
        group['Loss_Streak3'] = streak
        return group

    df_streak = df_sorted.groupby('Ticker', group_keys=False).apply(mark_consecutive_losses)
    loss_flag = (
        df_streak.groupby("Ticker")["Loss_Streak3"]
        .any()
        .reset_index()
        .rename(columns={"Loss_Streak3":"Default_Loss_Flag"})
    )
    loss_flag["Default_Loss_Flag"] = loss_flag["Default_Loss_Flag"].astype(int)

    # Merge
    label_df = equity_check.merge(loss_flag, on="Ticker", how="outer").fillna(0)
    label_df["Default_Label"] = ((label_df["Default_Equity_Flag"] == 1) | (label_df["Default_Loss_Flag"] == 1)).astype(int)

    df = df.merge(label_df[["Ticker", "Default_Label"]], on="Ticker", how="left")
    df["Default"] = df["Default_Label"].astype(int)
    df.drop(columns=["Default_Label"], inplace=True)
    return df


# ==========================
# 3. Tiền xử lý và tạo feature
# ==========================
def preprocess_and_create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ==== Đặt tên chuẩn ====
    df['Total_Assets'] = df['TOTAL ASSETS (Bn. VND)']
    df['Equity'] = df["OWNER'S EQUITY(Bn.VND)"]
    df['Current_Liabilities'] = df['Current liabilities (Bn. VND)']
    df['Long_Term_Liabilities'] = df['Long-term liabilities (Bn. VND)']
    df['Short_Term_Borrowings'] = df['Short-term borrowings (Bn. VND)']
    df['Total_Debt'] = df['Current_Liabilities'] + df['Long_Term_Liabilities'] + df['Short_Term_Borrowings']

    df['Operating_Profit'] = df['Operating Profit/Loss']
    df['Net_Profit'] = df['Net Profit For the Year']
    df['Revenue'] = df['Net Sales']
    df['Gross_Profit'] = df['Gross Profit']
    df['Interest_Expenses'] = df['Interest Expenses']
    df['Cash'] = df['Cash and cash equivalents (Bn. VND)']
    df['Receivables'] = df['Accounts receivable (Bn. VND)']
    df['Inventories'] = df['Net Inventories']
    df['Current_Assets'] = df['CURRENT ASSETS (Bn. VND)']
    df['OCF'] = df['Net cash inflows/outflows from operating activities']

    # ==== Ratios ====
    # Liquidity
    df['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities']
    df['Quick_Ratio'] = (df['Cash'] + df['Receivables']) / df['Current_Liabilities']
    df['Working_Capital_to_Total_Assets'] = (df['Current_Assets'] - df['Current_Liabilities']) / df['Total_Assets']

    # Leverage
    df['Debt_to_Assets'] = df['Total_Debt'] / df['Total_Assets']
    df['Debt_to_Equity'] = df['Total_Debt'] / df['Equity']
    df['Equity_to_Liabilities'] = df['Equity'] / df['Total_Debt']
    df['Long_Term_Debt_to_Assets'] = df['Long_Term_Liabilities'] / df['Total_Assets']

    # Efficiency
    df['Receivables_Turnover'] = df['Revenue'] / df['Receivables']
    df['Inventory_Turnover'] = df['Cost of Sales'] / df['Inventories']
    df['Asset_Turnover'] = df['Revenue'] / df['Total_Assets']

    # Profitability
    df['ROA'] = df['Net_Profit'] / df['Total_Assets']
    df['ROE'] = df['Net_Profit'] / df['Equity']
    df['EBIT_to_Assets'] = df['Operating_Profit'] / df['Total_Assets']
    df['Operating_Income_to_Debt'] = df['Operating_Profit'] / df['Total_Debt']
    df['Net_Profit_Margin'] = df['Net_Profit'] / df['Revenue']
    df['Gross_Margin'] = df['Gross_Profit'] / df['Revenue']

    # Coverage
    df['Interest_Coverage'] = df['Operating_Profit'] / df['Interest_Expenses']

    # EBITDA
    if 'Depreciation and Amortisation' in df.columns:
        df['EBITDA'] = df['Operating_Profit'] + df['Depreciation and Amortisation']
    else:
        df['EBITDA'] = np.nan
    df['EBITDA_to_Interest'] = df['EBITDA'] / df['Interest_Expenses']
    df['Total_Debt_to_EBITDA'] = df['Total_Debt'] / df['EBITDA']

    # ==== Fix Missing Features ====
    interaction_features = [
        ('Current_Ratio', 'Quick_Ratio'),
        ('Current_Ratio', 'Working_Capital_to_Total_Assets'),
        ('Current_Ratio', 'Equity_to_Liabilities'),
        ('Current_Ratio', 'Receivables_Turnover'),
        ('Current_Ratio', 'Asset_Turnover'),
        ('Current_Ratio', 'ROA'),
        ('Current_Ratio', 'ROE'),
        ('Current_Ratio', 'EBIT_to_Assets'),
        ('Current_Ratio', 'Operating_Income_to_Debt'),
        ('Current_Ratio', 'Net_Profit_Margin')
    ]
    for f1, f2 in interaction_features:
        colname = f"{f1}_x_{f2}"
        if f1 in df.columns and f2 in df.columns:
            df[colname] = df[f1] * df[f2]
        else:
            df[colname] = 0  # đảm bảo không báo missing feature nữa

    # Dọn dẹp
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
