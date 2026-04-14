import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Since investing.com did not give full data between 2000-2026 due to a technical problem, I get the data as 2 parts and I concatenated them.
# First gold data (2000-2020)
df_gold1 = pd.read_csv('Gold Futures Historical Data (3).csv')

# Second gold data (2020-2026)
df_gold2 = pd.read_csv('Gold Futures Historical Data (4).csv')

# Concatenated data
df_gold = pd.concat([df_gold1, df_gold2], axis=0)
df_gold['Date'] = pd.to_datetime(df_gold['Date'])

# Sorting the dates
df_gold = df_gold.sort_values(by='Date',ascending=False)

# Dropping duplicating dates
df_gold = df_gold.drop_duplicates()

# As in gold, I also get the oil data as 2 parts
oil1 = pd.read_csv('Brent Oil Futures Historical Data (1).csv', thousands=',', decimal='.')
oil2 = pd.read_csv('Brent Oil Futures Historical Data.csv', thousands=',', decimal='.')

# Concatenating oil data
df_oil = pd.concat([oil1, oil2], axis=0)
df_oil['Date'] = pd.to_datetime(df_oil['Date'])
df_oil = df_oil.sort_values(by='Date', ascending=False).drop_duplicates()

#Reading the GPR data
df_gpr = pd.read_csv('data_gpr_daily_recent.csv')
df_gpr['date'] = pd.to_datetime(df_gpr['date'])

# I filter the dates between 2000-2026. Also, column names were different than gold and oil data so I fixed it.
df_gpr = df_gpr[df_gpr['date'] >= '2000-01-01']
df_gpr = df_gpr.rename(columns={'date': 'Date'})

# I merged the Gold and Oil data with the dates in which they intersect using "how='inner'"". 
df_GoldandOil = pd.merge(df_gold, df_oil, on='Date', how='inner')

# Then merged all of the three data.
df_final = pd.merge(df_GoldandOil, df_gpr, on='Date', how='inner')

#Checking how the data looks
print(df_final.head())

df_final.to_csv('gold_oil_gpr_merged.csv', index=False)

#Making column names more readable.
df_final = df_final.rename(columns={
    'Price_x': 'Gold_Price',
    'Price_y': 'Oil_Price'
})



# Checking the statistical features such as mean, std and quartiles.
print(df_final[['Gold_Price', 'Oil_Price', 'GPRD']].describe())

# Skewness (Asimetri) ve Kurtosis (Basıklık)


# 3. KORELASYON HESAPLAMA
# Pearson: Lineer ilişki (Normal dağılım varsayar)

# 2. Sonsuz (inf) veya NaN değerleri temizleyelim (İlk satır NaN olur)
# Cleaning the comma and percentage signs in numbers and convert them to float.
cols_to_fix = ['Gold_Price', 'Oil_Price', 'GPRD', 'Change %_x', 'Change %_y','GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA7', 'GPRD_MA30']

for col in cols_to_fix:
    if col in df_final.columns:
        # Önce her şeyi string'e çevir (garantiye al)
        # Sonra %, virgül ve boşlukları temizle
        df_final[col] = (df_final[col].astype(str)
                         .str.replace('%', '')
                         .str.replace(',', '')
                         .str.strip())
        
        
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# Dropping 'Nan' parts.
df_final = df_final.dropna(subset=cols_to_fix)

#calculating the pearson correlation between gold and oil prices
pearson_val = df_final["Gold_Price"].corr(df_final["Oil_Price"], method='pearson')

# Spearman: Sıralama ilişkisi (Aykırı değerlere ve normal olmayan dağılıma dayanıklıdır)
spearman_val = df_final["Gold_Price"].corr(df_final["Oil_Price"], method='spearman')

#Checking the correlation values. 
print(f"Pearson Correlation: {pearson_val:.4f}")
print(f"Spearman Correlation: {spearman_val:.4f}")

# Spearman correlation is 0.6012. I costruct hypotesis testing to reflect whether this is a coincedence or there is a meaningful correlation.
# Null Hypothesis H0 = "There is no correlation between Gold and Oil Prices."
# Alternative Hypothesis HA= "There is a considirable correlation between Oil and Gold prices."
#Initially, I set alpha value to 0.05. If p value is less then 0.05, we can reject H0 and conclude that there is a meaningful correlation.

alpha = 0.05
spearman, p_value = stats.spearmanr(df_final['Gold_Price'], df_final['Oil_Price'])

if (p_value < alpha):
    print(f"p value: {p_value:.10e}")
    print("p value is smaller than alpha, Null Hypothesis H0 is rejected.")
    print("There is a statistically meaningful correlation between gold and oil prices.")

# I reach the p value is less then 0.00001 and conclude that there is a strong correlation between gold and oil prices. 

# Same proccess to inspect the correlation between gold prices and GPR index
spearman, p_value = stats.spearmanr(df_final['Gold_Price'], df_final['GPRD'])


if (p_value < alpha):
    print(f"p value: {p_value:.10e}")
    print("p value is smaller than alpha, Null Hypothesis H0 is rejected.")
    print("There is a statistically meaningful correlation between gold prices and GPR index.")


# Same proccess to inspect the correlation between oil prices and GPR index

spearman, p_value = stats.spearmanr(df_final['Oil_Price'], df_final['GPRD'])

if (p_value < alpha):
    print(f"p value: {p_value:.10e}")
    print("p value is smaller than alpha, Null Hypothesis H0 is rejected.")
    print("There is a statistically meaningful correlation between oil prices and GPR index.")


# p value for correlation between such commodities and GPR index resulted in that "Gold and Oil prices meaningfully correlated with GPR index."
