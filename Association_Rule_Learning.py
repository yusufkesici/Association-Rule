############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# Amacımız online retail II veri setine birliktelik analizi uygulayarak kullanıcılara ürün satın alma sürecinde
# ürün önermek

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak


# !pip install mlxtend
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


# Veri Ön İşleme

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.columns

df.isnull().sum()
df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


df.groupby("Country").count()

# Germany müşterileri üzerinden birliktelik kuralları.

df_german = df[df['Country'] == "Germany"]

# Birlikteli Kuralı Öğrenimi Veri yapısı
german_inv_pro_df = pd.pivot_table(df_german, index="Invoice", columns="StockCode", values="Quantity", aggfunc="sum").fillna(0).applymap(
    lambda x: 1 if x > 0 else 0)


# Birliktelik kuralları
# Apriori
frequent_itemsets = apriori(german_inv_pro_df, min_support=0.01, use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("lift", ascending=False).head()

rules.sort_values("lift", ascending=False).head(500)

# ID'leri verilen ürünlerin isimleri nelerdir?

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_german, 21987)
check_id(df_german, 23235)
check_id(df_german, 22747)



 #Kullanıcı 1 ürün id'si: 21987 ==>>  PACK OF 6 SKULL PAPER CUPS
 #Kullanıcı 2 ürün id'si: 23235 ==>>  STORAGE TIN VINTAGE LEAF
 #Kullanıcı 3 ürün id'si: 22747 ==>>  POPPY'S PLAYHOUSE BATHROOM


# Sepetteki kullanıcılar için ürün önerisi


def recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

recommender(rules,21987,3)
recommender(rules,23235,3)
recommender(rules,22747,3)

#  Önerilen ürünlerin isimleri

def check_ids(list):
    for i in list:
            check_id(df_german, i)


# Kullanıcı 1 ürün id'si: 21987 ==>>  PACK OF 6 SKULL PAPER CUPS
check_ids(recommender(rules,21987,3))  #  ==>> ['POSTAGE']['SET/10 BLUE POLKADOT PARTY CANDLES']['SPACEBOY BIRTHDAY CARD']

# Kullanıcı 2 ürün id'si: 23235 ==>>  STORAGE TIN VINTAGE LEAF
check_ids(recommender(rules,23235,3))  #  ==>> ['POSTAGE']['BLUE POLKADOT PLATE ']['SPACEBOY BIRTHDAY CARD']

# Kullanıcı 3 ürün id'si: 22747 ==>>  POPPY'S PLAYHOUSE BATHROOM
check_ids(recommender(rules,22747,3))  #  ==>> ['POSTAGE']['PINK 3 PIECE POLKADOT CUTLERY SET']['RED RETROSPOT MINI CASES']


