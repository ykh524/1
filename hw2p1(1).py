# hw2
# Kaihan Yang

import pandas as pd, numpy as np

df = pd.read_csv("adult.csv")  # 读取成人数据集 CSV 文件

# 对数值型属性做差分隐私直方图
# series: 待统计的 Pandas Series
# bins: 直方图区间边界
# eps: 隐私预算 ε
# 返回: (带噪计数, 区间边界)
def dp_histogram(series, bins, eps):
    counts, edges = numpy.histogram(series, bins=bins)  # 计算原始直方图计数
    # global sensitivity = 1 （插入/删除邻居模型）
    # Laplace 噪声尺度 b = Δ/ε = 1/ε
    noisy = counts + numpy.random.laplace(scale=1/eps, size=len(counts))
    return noisy, edges

# 对类别型属性做差分隐私计数
# series: 待统计的 Pandas Series
# eps: 隐私预算 ε
# 返回: (带噪计数, 类别序列)
def categorical_dp_hist(series, eps):
    levels = series.unique()  # 获取所有类别水平
    counts = series.value_counts().reindex(levels, fill_value=0).values  # 原始每类计数
    # Laplace scale b = 1/ε
    noisy = counts + numpy.random.laplace(scale=1/eps, size=len(counts))
    return noisy, levels

# 总隐私预算 ε
EPS_TOTAL = 1.0
# 年龄直方图的区间边界，从 17 到 91 年，10 个等宽区间
bins_age = numpy.linspace(17, 91, 11)

# Scenario (i): 每个属性单独使用 ε=1
eps_age_i = EPS_TOTAL
eps_wc_i = EPS_TOTAL
eps_ed_i = EPS_TOTAL

# 生成带噪结果
age_noisy_i, _   = dp_histogram(df["age"], bins_age, eps_age_i)
wc_noisy_i, lab_w = categorical_dp_hist(df["workclass"], eps_wc_i)
ed_noisy_i, lab_e = categorical_dp_hist(df["education"], eps_ed_i)

print("Scenario (i) — 每个属性 ε=1, Laplace scale b=1")
print("Noisy age histogram:    ", age_noisy_i.astype(int))
print("Noisy workclass counts: ", dict(zip(lab_w, wc_noisy_i.astype(int))))
print("Noisy education counts: ", dict(zip(lab_e, ed_noisy_i.astype(int))))
print()

# Scenario (ii): 属性不相关时，仍可并行使用 ε=1
# 不拆分隐私预算，保持与 Scenario (i) 一致
eps_each = EPS_TOTAL  # 与 Scenario (i) 相同，不需拆分budget
age_noisy_ii, _ = dp_histogram(df["age"], bins_age, eps_each)
wc_noisy_ii, _ = categorical_dp_hist(df["workclass"], eps_each)
ed_noisy_ii, _ = categorical_dp_hist(df["education"], eps_each)

print("Scenario (ii) — 属性不相关，仍每个属性 ε=1, Laplace scale b=1")
print("Noisy age histogram:    ", age_noisy_ii.astype(int))
print("Noisy workclass counts: ", dict(zip(lab_w, wc_noisy_ii.astype(int))))
print("Noisy education counts: ", dict(zip(lab_e, ed_noisy_ii.astype(int))))

# Explain/justify your choices # I set ε = 1 and δ = 10⁻⁵ for Gaussian noise because this is the standard moderate privacy budget used in many demos; δ < 1/n. # For numeric queries the global sensitivity equals (max − min)/n, for histogram counts it is 1, so the Laplace scale is b = 1/ε. # What is a meaningful comparison? # keep ε identical and δ identical. # run the code ≥ 1 000 times and record the distribution of errors if you want an empirical plot.
