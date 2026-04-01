import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel("C:\\Users\86155\Desktop\PythonProject\data\\raw\input.xlsx")

# ---------- 数据预处理 ----------
# 将售价和原价转换为数字（去除逗号等），并处理缺失值
df['actual_price'] = df['actual_price'].astype(str).str.replace(',', '').astype(float)
df['selling_price'] = df['selling_price'].astype(str).str.replace(',', '').astype(float)

# 填充缺失的售价和原价（使用中位数）
median_actual = df['actual_price'].median()
median_selling = df['selling_price'].median()
df['actual_price'].fillna(median_actual, inplace=True)
df['selling_price'].fillna(median_selling, inplace=True)

# 计算折扣率，避免分母为零
df['discount_rate'] = (df['selling_price'] / df['actual_price'].replace(0, np.nan)).fillna(0).clip(0, 0.99)

# 评分缺失值填充为 3.5
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce').fillna(3.5)

# 子类别缺失值填充为 'Unknown'
df['sub_category'] = df['sub_category'].fillna('Unknown').astype(str)

# ---------- 定义品类因子（基于关键词匹配） ----------
# 销量因子映射
sales_factor_map = [
    (['innerwear', 'swimwear', 'underwear', 'socks', 'vest', 'boxer'], 1.6),
    (['accessory', 'tie', 'cufflink', 'scarf', 'muffler', 'bags', 'wallets', 'belts'], 1.3),
    (['topwear', 't-shirt', 'shirt', 'sweater', 'sweatshirt'], 1.2),
    (['bottomwear', 'track pant', 'trouser', 'jeans', 'short'], 1.2),
    (['footwear', 'shoe', 'sneaker', 'loafer'], 1.2),
    (['sleepwear', 'pajama', 'nightwear'], 1.2),
    (['winter wear', 'jacket', 'coat', 'blazer'], 1.0),
    (['ethnic', 'kurta', 'pyjama', 'pathani', 'ethnic set'], 1.0),
    (['tracksuit'], 1.0),
    (['fabric'], 0.6),
    (['raincoat'], 0.8),
]

def get_sales_factor(sub_cat):
    sub_lower = sub_cat.lower()
    for keywords, factor in sales_factor_map:
        if any(kw in sub_lower for kw in keywords):
            return factor
    return 1.0

# 复购率因子映射
repeat_factor_map = [
    (['innerwear', 'swimwear', 'underwear', 'socks', 'vest', 'boxer'], 1.5),
    (['accessory', 'tie', 'cufflink', 'scarf', 'muffler', 'bags', 'wallets', 'belts'], 0.8),
    (['topwear', 't-shirt', 'shirt', 'sweater', 'sweatshirt'], 1.2),
    (['bottomwear', 'track pant', 'trouser', 'jeans', 'short'], 1.0),
    (['footwear', 'shoe', 'sneaker', 'loafer'], 0.9),
    (['sleepwear', 'pajama', 'nightwear'], 1.2),
    (['winter wear', 'jacket', 'coat', 'blazer'], 0.8),
    (['ethnic', 'kurta', 'pyjama', 'pathani', 'ethnic set'], 0.9),
    (['tracksuit'], 0.8),
    (['fabric'], 0.5),
    (['raincoat'], 0.7),
]

def get_repeat_factor(sub_cat):
    sub_lower = sub_cat.lower()
    for keywords, factor in repeat_factor_map:
        if any(kw in sub_lower for kw in keywords):
            return factor
    return 1.0

# ---------- 生成销量 ----------
np.random.seed(42)
df['sales'] = 0
for idx, row in df.iterrows():
    price = row['selling_price']
    discount = row['discount_rate']
    rating = row['average_rating']
    sub_cat = row['sub_category']

    # 防止 price 为零或负值导致除零
    price = max(1, price)
    price_factor = max(0.2, min(1.5, 3000 / (price + 100)))
    discount_factor = 1 + discount * 0.8
    rating_factor = 0.6 + rating * 0.2
    cat_factor = get_sales_factor(sub_cat)
    random_factor = np.random.uniform(0.7, 1.3)

    sales = 1000 * price_factor * discount_factor * rating_factor * cat_factor * random_factor
    if np.isnan(sales):
        sales = 5  # 如果仍然是 NaN，则设为最小值
    df.at[idx, 'sales'] = max(5, int(sales))

# ---------- 生成复购率 ----------
df['repeat_rate'] = 0.0
for idx, row in df.iterrows():
    price = row['selling_price']
    rating = row['average_rating']
    sub_cat = row['sub_category']

    price = max(1, price)
    base_repeat = 0.03
    price_factor = max(0.5, min(1.2, 1500 / (price + 500)))
    rating_factor = 0.9 + rating * 0.02
    cat_factor = get_repeat_factor(sub_cat)
    random_factor = np.random.uniform(0.8, 1.2)

    repeat = base_repeat * cat_factor * rating_factor * price_factor * random_factor
    if np.isnan(repeat):
        repeat = 0.03
    df.at[idx, 'repeat_rate'] = round(max(0.01, min(0.25, repeat)), 4)

# ---------- 保存结果 ----------
df.to_csv("C:\\Users\86155\Desktop\PythonProject\data\\raw\\new_input.csv", index=False)
print("处理完成，结果已保存为 new_input.csv")
