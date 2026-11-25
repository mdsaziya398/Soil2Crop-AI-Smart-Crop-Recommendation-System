
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Dataset/Crop_recommendation.csv")

# Set visualization style
sns.set_style("whitegrid")

# 1. Crop Distribution
plt.figure(figsize=(14, 6))

sns.countplot(
    data=df,
    x="label",
    order=df["label"].value_counts().index
)

plt.title("Crop Distribution")
plt.xlabel("Crop Name")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graph.png")


# 2. Feature Distribution
features = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]

df[features].hist(
    figsize=(14, 10),
    bins=20
)

plt.suptitle("Distribution of Agricultural Features")
plt.tight_layout()
plt.savefig("graph.png")


# 3. Correlation Heatmap
plt.figure(figsize=(10, 7))

correlation = df[features].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Agricultural Features")
plt.tight_layout()
plt.savefig("graph.png")


# 4. Boxplots
plt.figure(figsize=(14, 7))

sns.boxplot(data=df[features])

plt.title("Boxplot of Soil and Weather Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graph.png")

print("EDA completed successfully!")
