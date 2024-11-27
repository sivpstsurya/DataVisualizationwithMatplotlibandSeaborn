import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data = pd.read_csv(url, names=column_names)

# Display the first few rows of the DataFrame
print("Data Preview:")
print(data.head())

# Set the style for seaborn
sns.set(style="whitegrid")

# 1. Bar Chart: Average Sepal Length by Species
plt.figure(figsize=(10, 6))
sns.barplot(x="species", y="sepal_length", data=data, estimator=lambda x: sum(x) / len(x), palette="viridis")
plt.title("Average Sepal Length by Species")
plt.ylabel("Average Sepal Length (cm)")
plt.xlabel("Species")
plt.show()

# 2. Scatter Plot: Sepal Length vs Sepal Width
plt.figure(figsize=(10, 6))
sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", style="species", data=data, palette="deep", s=100)
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title='Species')
plt.show()

# 3. Pair Plot: Pairwise relationships in the dataset
sns.pairplot(data, hue="species", palette="deep")
plt.suptitle("Pairwise Relationships in the Iris Dataset", y=1.02)
plt.show()

# 4. Heatmap: Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# 5. Box Plot: Petal Length by Species
plt.figure(figsize=(10, 6))
sns.boxplot(x="species", y="petal_length", data=data, palette="Set2")
plt.title("Petal Length Distribution by Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.show()