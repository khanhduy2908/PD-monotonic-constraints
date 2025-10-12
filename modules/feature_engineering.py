import numpy as np
import pandas as pd

def preprocess_and_create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, standardize column names, derive financial ratios, sentiment features,
    and assign binary Default label (1 = default risk, 0 = non-default) using
    the 3-year consecutive loss rule.
    """
    df = df.copy()

    # ========== 1. Standardize Column Names ==========
    rename_map = {
        'TOTAL ASSETS (Bn. VND)': 'Total_Assets',
        "OWNER'S EQUITY(Bn.VND)": 'Equity',
        'Current liabilities (Bn. VND)': 'Current_Liabilities',
        'Long-term liabilities (Bn. VND)': 'Long_Term_Liabilities',
        'Short-term borrowings (Bn. VND)': 'Short_Term_Borrowings',
        'Operating Profit/Loss': 'Operating_Profit',
        'Net Profit For the Year': 'Net_Profit',
        'Net Sales': 'Revenue',
        'Gross Profit': 'Gross_Profit',
        'Financial Expenses': 'Financial_Expenses',
        'Interest Expenses': 'Interest_Expenses',
        'Cash and cash equivalents (Bn. VND)': 'Cash',
        'Accounts receivable (Bn. VND)': 'Receivables',
        'Net Inventories': 'Inventories',
        'CURRENT ASSETS (Bn. VND)': 'Current_Assets',
        'Net cash inflows/outflows from operating activities': 'OCF',
        'Sentiment Score': 'Sentiment_Score',
        'Positive Ratio': 'Positive_Ratio',
        'Negative Ratio': 'Negative_Ratio',
        'Neutral Ratio': 'Neutral_Ratio',
        'News Volume': 'News_Volume',
        'Sentiment Change': 'Sentiment_Change',
        'News Shock': 'News_Shock',
    }
    df.rename(columns=rename_map, inplace=True)

    # Calculate Total Debt
    df['Total_Debt'] = df['Current_Liabilities'] + df['Long_Term_Liabilities'] + df['Short_Term_Borrowings']

    # ========== 2. Default Label: 3-year consecutive loss rule ==========
    df = df.sort_values(['Ticker', 'Year']).reset_index(drop=True)
    df['is_loss'] = (df['Net_Profit'] <= 0).astype(int)

    def mark_consecutive_losses(group):
        years = group['Year'].values
        losses = group['is_loss'].values
        streak = [False] * len(group)
        for i in range(len(group) - 2):
            if (
                losses[i] and losses[i + 1] and losses[i + 2]
                and years[i + 1] == years[i] + 1
                and years[i + 2] == years[i] + 2
            ):
                streak[i] = streak[i + 1] = streak[i + 2] = True
        group['Loss_Streak3'] = streak
        return group

    df = df.groupby('Ticker', group_keys=False).apply(mark_consecutive_losses)
    default_years = df[df['Loss_Streak3']].groupby('Ticker')['Year'].min().rename("Default_Year")
    df = df.merge(default_years, on='Ticker', how='left')

    df['Years_Before_Default'] = df['Default_Year'] - df['Year']
    df['Will_Default_Within_5Y'] = (
        (df['Years_Before_Default'] >= 0) & (df['Years_Before_Default'] <= 5)
    ).astype(int)
    df['Already_Defaulted'] = np.where(
        df['Default_Year'].notna(),
        (df['Year'] > df['Default_Year']).astype(int),
        0,
    )
    df.drop(columns=['Default'], errors='ignore', inplace=True)
    df['Default'] = df.groupby('Ticker')['Will_Default_Within_5Y'].transform('max')

    # Drop intermediate default vars
    df.drop(columns=['is_loss', 'Loss_Streak3', 'Default_Year', 'Years_Before_Default'], inplace=True)

    # ========== 3. Financial Ratios ==========
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
    df['Net_Debt'] = df['Total_Debt'] - df['Cash']
    df['Net_Debt_to_Equity'] = df['Net_Debt'] / df['Equity']

    # OCF Deficit & Variability
    df['OCF_Negative'] = (df['OCF'] < 0).astype(int)
    df['OCF_Deficit_2of3'] = (
        df.groupby('Ticker')['OCF_Negative']
        .transform(lambda x: x.rolling(3, min_periods=3).sum() >= 2)
        .astype(int)
    )
    df['Revenue_Growth'] = df.groupby('Ticker')['Revenue'].pct_change()
    df['Revenue_CAGR_3Y'] = (
        df.groupby('Ticker')['Revenue_Growth']
        .transform(lambda x: x.rolling(3, min_periods=3).mean())
        .fillna(0)
    )
    df['PAT_Std_3Y'] = (
        df.groupby('Ticker')['Net_Profit']
        .transform(lambda x: x.rolling(3, min_periods=3).std())
        .fillna(0)
    )

    # Low Risk Flag
    df['LowRiskFlag'] = (
        (df['Debt_to_Assets'] < 0.5)
        & (df['ROA'] > 0.1)
        & (df['Current_Ratio'] > 1)
    ).astype(int)

    # Sector Default Rate (optional)
    if 'Sector' in df.columns:
        sector_default_rate_map = (
            df[df['Already_Defaulted'] == 0].groupby('Sector')['Default'].mean()
        )
        df['Sector_Default_Rate'] = df['Sector'].map(sector_default_rate_map).fillna(0)
    else:
        df['Sector_Default_Rate'] = 0

    # ========== 4. Clean up ==========
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop(columns=['OCF_Negative', 'Revenue_Growth'], inplace=True)
    df.fillna(0, inplace=True)

    return df
