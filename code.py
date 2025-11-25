print("Welcome to Field Grain Data Science Project")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

# Load data
df = pd.read_excel('FeedGrains.xls')
print("Columns:", df.columns.tolist())

# Column meanings
info = {
    'SC_Group_Desc': 'Category (e.g., Supply and use, Prices)',
    'SC_Commodity_Desc': 'Commodity (e.g., Barley, Corn, Soybean meal)',
    'SC_Attribute_Desc': 'Attribute (e.g., Yield, Prices, Ending stocks)',
    'SC_Unit_Desc': 'Unit of measurement (e.g., Bushels/acre, Dollars/bushel)',
    'Year_ID': 'Year of data',
    'SC_Frequency_Desc': 'Data frequency (e.g., Annual, Monthly)',
    'Timeperiod_Desc': 'Specific time period (e.g., Commodity Market Year)',
    'Amount': 'Value of the attribute'
}
print("\nColumn meanings:")
for col, desc in info.items():
    print(f"{col}: {desc}")

# Before cleaning
print("\nBefore cleaning:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Clean data
df = df.replace("?", np.nan)
df = df.replace(0, np.nan)
df['High_Yield'] = df['Amount'] > df['Amount'].quantile(0.75)  # Flag high yield years
print("\nAfter replacing '?' and 0 with NaN:")
print(df.isnull().sum())
df_cleaned = df.dropna()
print("\nAfter cleaning:")
print(df_cleaned.info())
print("\nMissing values:")
print(df_cleaned.isnull().sum())
print(f"Rows left: {len(df_cleaned)}")

# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, shapiro, ttest_ind
sns.set(style='whitegrid')

# Graph 1: Data points per commodity
# Get the top 10 commodities by count
top_10_commodities = df_cleaned['SC_Commodity_Desc'].value_counts().nlargest(10).index
df_top_10 = df_cleaned[df_cleaned['SC_Commodity_Desc'].isin(top_10_commodities)]

# Create the plot
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='SC_Commodity_Desc', hue='SC_Commodity_Desc', 
                   data=df_top_10, palette='crest', legend=False)

# Customize the plot
plt.title('Top 10 Commodities by Data Points', fontsize=16, pad=10)
plt.xlabel('Commodity', fontsize=12)
plt.ylabel('Number of Data Points', fontsize=12)

# Add percentage labels
total = len(df_top_10)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 3, 
            f'{height/total:.1%}', ha="center", fontsize=10)

# Add grid and adjust layout
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
print("\nGraph 1: Data distribution across top 10 commodities!")
plt.tight_layout()
plt.show()

# Graph 2: Yield distribution by commodity
plt.figure(figsize=(12, 6))
yield_data = df_cleaned[df_cleaned['SC_Attribute_Desc'] == 'Yield per harvested acre']

# Boxen plot by commodity
sns.boxenplot(x='SC_Commodity_Desc', y='Amount', hue='SC_Commodity_Desc', 
              data=yield_data, palette='muted')

# Customize
plt.title('Yield Distribution by Commodity', fontsize=16, pad=10)
plt.xlabel('Commodity', fontsize=12)
plt.ylabel('Bushels per Acre', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
print("\nGraph 2: Yield distributions across commodities!")
plt.tight_layout()
plt.show()

# Graph 3: Yield by year with step plot and rolling average
plt.figure(figsize=(10, 6))
yield_data = df_cleaned[df_cleaned['SC_Attribute_Desc'] == 'Yield per harvested acre']

# Step plot for yearly averages
sns.lineplot(x='Year_ID', y='Amount', data=yield_data, estimator='mean', 
             drawstyle='steps-mid', marker='o', color='darkblue', linewidth=2, label='Yearly Avg')

# Add 5-year rolling average
rolling_avg = yield_data.groupby('Year_ID')['Amount'].mean().rolling(window=5, center=True).mean()
plt.plot(rolling_avg.index, rolling_avg, color='orange', linestyle='-', linewidth=2, label='5-Year Rolling Avg')

# Customize
plt.title('Average Yield by Year with Trends', fontsize=16, pad=10)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Bushels per Acre', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
print("\nGraph 3: Yearly yield steps with rolling trend!")
plt.tight_layout()
plt.show()


# Graph 5: Median price by commodity
plt.figure(figsize=(12, 6))
price_data = df_cleaned[df_cleaned['SC_Attribute_Desc'] == 'Prices received by farmers']

# Custom function to calculate IQR for error bars
def iqr_error(x):
    q25, q75 = np.percentile(x, [25, 75])
    return [[np.median(x) - q25], [q75 - np.median(x)]]

# Bar plot with median and IQR error bars
sns.barplot(x='SC_Commodity_Desc', y='Amount', hue='SC_Commodity_Desc', 
            data=price_data, estimator=np.median, errorbar=iqr_error, palette='Set2')

# Annotate highest median
medians = price_data.groupby('SC_Commodity_Desc')['Amount'].median()
max_commodity = medians.idxmax()
max_value = medians.max()
plt.annotate(f'Highest: {max_value:.2f}', 
             xy=(list(medians.index).index(max_commodity), max_value), 
             xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10, color='red')

# Customize
plt.title('Median Price by Commodity with IQR', fontsize=16, pad=10)
plt.xlabel('Commodity', fontsize=12)
plt.ylabel('Dollars per Unit', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
print("\nGraph 5: Median price per commodity with IQR!")
plt.tight_layout()
plt.show()

# Graph 6: Yield vs. price with hexbin and regression
plt.figure(figsize=(12, 6))
yield_price = df_cleaned[df_cleaned['SC_Attribute_Desc'].isin(['Yield per harvested acre', 'Prices received by farmers'])]
pivot_data = yield_price.pivot_table(index='Year_ID', columns='SC_Attribute_Desc', values='Amount').dropna()

# Hexbin plot
plt.hexbin(pivot_data['Yield per harvested acre'], pivot_data['Prices received by farmers'], 
           gridsize=20, cmap='Blues', mincnt=1)
cb = plt.colorbar(label='Count')

# Add regression line
m, b = np.polyfit(pivot_data['Yield per harvested acre'], pivot_data['Prices received by farmers'], 1)
plt.plot(pivot_data['Yield per harvested acre'], m * pivot_data['Yield per harvested acre'] + b, 
         color='red', linestyle='--', label='Regression Line')

# Customize
plt.title('Yield vs. Price Density', fontsize=16, pad=10)
plt.xlabel('Yield (Bushels per Acre)', fontsize=12)
plt.ylabel('Price (Dollars per Unit)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
print("\nGraph 6: Yield vs. price density with trend!")
plt.tight_layout()
plt.show()

# Graph 7: Correlations
cols = ['Yield per harvested acre', 'Prices received by farmers', 'Ending stocks', 'Exports, trade year']
plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned[df_cleaned['SC_Attribute_Desc'].isin(cols)].pivot_table(index='Year_ID', columns='SC_Attribute_Desc', values='Amount').corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 10})
plt.title('Metric Correlations', fontsize=16, pad=10)
plt.tight_layout()
print("\荣耀Graph 7: How metrics connect!")
plt.show()

# Graph 8: Metric distributions with outliers
plt.figure(figsize=(12, 6))
check_cols = ['Yield per harvested acre', 'Prices received by farmers', 'Ending stocks']
pivot_data = df_cleaned[df_cleaned['SC_Attribute_Desc'].isin(check_cols)].pivot_table(
    index='Year_ID', columns='SC_Attribute_Desc', values='Amount')

# Calculate outliers
z_scores = pivot_data.apply(zscore)
outlier_years = (z_scores.abs() > 3).any(axis=1)
outlier_flag = df_cleaned['Year_ID'].isin(z_scores.index[outlier_years]).reindex(df_cleaned.index, fill_value=False)

# Strip plot with outlier coloring
sns.stripplot(x='SC_Attribute_Desc', y='Amount', hue=outlier_flag, 
              data=df_cleaned[df_cleaned['SC_Attribute_Desc'].isin(check_cols)], 
              palette={True: 'red', False: 'blue'}, alpha=0.6, size=5, jitter=0.3)

# Add mean points
sns.pointplot(x='SC_Attribute_Desc', y='Amount', 
              data=df_cleaned[df_cleaned['SC_Attribute_Desc'].isin(check_cols)], 
              estimator='mean', color='black', markers='D', linestyles='', scale=1.5, label='Mean')

# Customize
plt.title('Metric Distributions with Outliers Highlighted', fontsize=16, pad=10)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Outlier', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
print("\nGraph 8: Metrics with outliers highlighted!")
plt.tight_layout()
plt.show()

# EDA
print("\nEDA: Digging Deeper!")

# Stats summary
print("\nStats Summary:")
stats = df_cleaned.describe()
print(stats)
print(f"\nHighest yield: {stats['Amount']['max']:.1f} bushels/acre")
print(f"Lowest price: {stats['Amount']['min']:.1f} dollars/unit")

# Correlations
print("\nTop Correlations:")
corr = df_cleaned[df_cleaned['SC_Attribute_Desc'].isin(cols)].pivot_table(index='Year_ID', columns='SC_Attribute_Desc', values='Amount').corr()
high_corrs = corr.unstack().sort_values().drop_duplicates()
print(high_corrs[(high_corrs > 0.5) | (high_corrs < -0.5)][1:6])

# Outliers
print("\nOutliers:")
check_cols = ['Yield per harvested acre', 'Prices received by farmers', 'Ending stocks']
z = df_cleaned[df_cleaned['SC_Attribute_Desc'].isin(check_cols)].pivot_table(index='Year_ID', columns='SC_Attribute_Desc', values='Amount').apply(zscore)
outliers = (z.abs() > 3).any(axis=1)
print(f"Outlier years: {outliers.sum()}")
print(df_cleaned[df_cleaned['Year_ID'].isin(z.index[outliers])][['Year_ID', 'SC_Commodity_Desc', 'SC_Attribute_Desc', 'Amount']].head())

# Stats Introduction
print("\nStats Intro: Easy Number Crunching!")

# 1. Descriptive Stats
print("\nMean and Median Yield:")
yield_data = df_cleaned[df_cleaned['SC_Attribute_Desc'] == 'Yield per harvested acre']['Amount']
print(f"Mean Yield: {yield_data.mean():.1f}")
print(f"Median Yield: {yield_data.median():.1f}")
plt.figure(figsize=(8, 5))
plt.hist(yield_data, bins=20, color='lightblue', rwidth=0.8)
plt.axvline(yield_data.mean(), color='red', linestyle='--', label=f'Mean: {yield_data.mean():.1f}')
plt.title('Yield Spread', fontsize=16, pad=10)
plt.xlabel('Bushels per Acre', fontsize=12)
plt.ylabel('Number of Years', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
print("\nGraph 9: Yield histogram!")
plt.tight_layout()
plt.show()

# 2. Shapiro-Wilk
print("\nShapiro-Wilk Test for Yield:")
stat, p = shapiro(yield_data)
print(f"P-value: {p:.4f}")
print("Normal curve?" if p > 0.05 else "Not normal!")
plt.figure(figsize=(8, 5))
plt.hist(yield_data, bins=20, color='skyblue', rwidth=0.8)
plt.axvline(yield_data.mean(), color='red', linestyle='--', label=f'Mean: {yield_data.mean():.1f}')
plt.title('Yield Normality Check', fontsize=16, pad=10)
plt.xlabel('Bushels per Acre', fontsize=12)
plt.ylabel('Number of Years', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
print("\nGraph 10: Yield distribution!")
plt.tight_layout()
plt.show()

# 3. Normal Distribution
print("\nNormal Distribution:")
m = yield_data.mean()
s = yield_data.std()
norm_data = np.random.normal(m, s, 1000)  # Increased to 1000 for smoother curve
plt.figure(figsize=(8, 5))
plt.hist(norm_data, bins=20, color='green', rwidth=0.8)
plt.axvline(m, color='red', linestyle='--', label=f'Mean: {m:.1f}')
plt.title('Normal Yield Simulation', fontsize=16, pad=10)
plt.xlabel('Bushels per Acre', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
print("\nGraph 11: Simulated normal yield!")
plt.tight_layout()
plt.show()

# 4. T-test
print("\nT-test: Early (2000-2010) vs. Recent (2015-2025) Yield:")
early_yield = df_cleaned[(df_cleaned['SC_Attribute_Desc'] == 'Yield per harvested acre') & 
                        df_cleaned['Year_ID'].between(2000, 2010)]['Amount']
recent_yield = df_cleaned[(df_cleaned['SC_Attribute_Desc'] == 'Yield per harvested acre') & 
                         df_cleaned['Year_ID'].between(2015, 2025)]['Amount']
t_stat, p = ttest_ind(early_yield, recent_yield, equal_var=False)  # Welch's t-test
print(f"P-value: {p:.4f}")
print("Different yields!" if p < 0.05 else "Similar yields!")
plt.figure(figsize=(8, 5))
sns.boxplot(x=pd.cut(df_cleaned['Year_ID'], bins=[1999, 2010, 2025], labels=['2000-2010', '2015-2025']), 
            y='Amount', hue=pd.cut(df_cleaned['Year_ID'], bins=[1999, 2010, 2025], labels=['2000-2010', '2015-2025']), 
            data=df_cleaned[df_cleaned['SC_Attribute_Desc'] == 'Yield per harvested acre'], palette='pastel', legend=False)
plt.title('Yield: 2000-2010 vs. 2015-2025', fontsize=16, pad=10)
plt.xlabel('Period', fontsize=12)
plt.ylabel('Bushels per Acre', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
print("\nGraph 12: Yield for 2000-2010 vs. 2015-2025!")
plt.tight_layout()
plt.show()

print("\nDone! Favorite agricultural stat?")
