import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, normaltest

# Load the CSV file into a pandas DataFrame
file_path = "data/preprocessed_data_for_dist.csv"  # Update this path if necessary
df = pd.read_csv(file_path)

# Remove non-numerical columns (if any)
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns

# Set up the plotting grid
num_cols = len(numerical_columns)
fig, axes = plt.subplots(nrows=(num_cols // 3) + 1, ncols=3, figsize=(15, 5 * ((num_cols // 3) + 1)))
axes = axes.flatten()

# Plot histograms and check normality for each numerical column
for i, col in enumerate(numerical_columns):
    sns.histplot(df[col], kde=True, ax=axes[i], color="blue", bins=30)
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

    shapiro_test = shapiro(df[col])
    ks_test = kstest(df[col], 'norm', args=(df[col].mean(), df[col].std()))
    dagostino_test = normaltest(df[col])

    # Display test results in the plot title
    axes[i].text(
        0.95, 0.95,
        f"Shapiro-Wilk p={shapiro_test.pvalue:.3f}\n"
        f"KS p={ks_test.pvalue:.3f}\n"
        f"D'Agostino p={dagostino_test.pvalue:.3f}",
        transform=axes[i].transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

# Remove empty subplots (if any)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
