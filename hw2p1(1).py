# hw2
# Kaihan Yang

import pandas as pd, numpy as np

df = pd.read_csv("adult.csv")  # Read adult dataset CSV file

# Differential privacy histogram for numerical attributes
# series: Pandas Series to be analyzed
# bins: Directly set histogram bin edges
# eps: Privacy budget ε
# Returns: (noisy counts, bin edges)
def dp_histogram(series, bins, eps):
    counts, edges = numpy.histogram(series, bins=bins)  # Compute original histogram counts
    # global sensitivity = 1 （insert/delete neighboring model）
    # Laplace noise scale b = Δ/ε = 1/ε
    noisy = counts + numpy.random.laplace(scale=1/eps, size=len(counts))
    return noisy, edges

# Differential privacy histogram for categorical attributes
# series: Pandas Series to be analyzed
# eps: Privacy budget ε
# Returns: (noisy counts, category list)
def categorical_dp_hist(series, eps):
    levels = series.unique()  # Get all category levels
    counts = series.value_counts().reindex(levels, fill_value=0).values  # Original category counts
    # Laplace scale b = 1/ε
    noisy = counts + numpy.random.laplace(scale=1/eps, size=len(counts))
    return noisy, levels

# Total privacy budget ε
EPS_TOTAL = 1.0
# Age histogram bin edges, from 17 to 91 years, 11 equally spaced intervals
bins_age = numpy.linspace(17, 91, 11)

# Scenario (i): Each attribute uses ε = 1 independently
eps_age_i = EPS_TOTAL
eps_wc_i = EPS_TOTAL
eps_ed_i = EPS_TOTAL

# Generate noisy results
age_noisy_i, _   = dp_histogram(df["age"], bins_age, eps_age_i)
wc_noisy_i, lab_w = categorical_dp_hist(df["workclass"], eps_wc_i)
ed_noisy_i, lab_e = categorical_dp_hist(df["education"], eps_ed_i)

print("Scenario (i) — 每个属性 ε=1, Laplace scale b=1")
print("Noisy age histogram:    ", age_noisy_i.astype(int))
print("Noisy workclass counts: ", dict(zip(lab_w, wc_noisy_i.astype(int))))
print("Noisy education counts: ", dict(zip(lab_e, ed_noisy_i.astype(int))))
print()

# Scenario (ii): Attributes are independent, still use ε=1 in parallel
# Same total privacy budget, same effect as Scenario (i)
eps_each = EPS_TOTAL  # Same as Scenario (i), no need to split budget
age_noisy_ii, _ = dp_histogram(df["age"], bins_age, eps_each)
wc_noisy_ii, _ = categorical_dp_hist(df["workclass"], eps_each)
ed_noisy_ii, _ = categorical_dp_hist(df["education"], eps_each)

print("Scenario (ii) – attributes independent, still ε=1, Laplace scale b=1")
print("Noisy age histogram:    ", age_noisy_ii.astype(int))
print("Noisy workclass counts: ", dict(zip(lab_w, wc_noisy_ii.astype(int))))
print("Noisy education counts: ", dict(zip(lab_e, ed_noisy_ii.astype(int))))

# Explain/justify your choices # I set ε = 1 and δ = 10⁻⁵ for Gaussian noise because this is the standard moderate privacy budget used in many demos; δ < 1/n. # For numeric queries the global sensitivity equals (max − min)/n, for histogram counts it is 1, so the Laplace scale is b = 1/ε. # What is a meaningful comparison? # keep ε identical and δ identical. # run the code ≥ 1 000 times and record the distribution of errors if you want an empirical plot.
