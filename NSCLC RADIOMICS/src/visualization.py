import seaborn as sns
import matplotlib.pyplot as plt


def plot_results(results_df, save_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="FS_method", y="F1_score", hue="Classifier")
    plt.title("F1-score per Feature Selection and Classifier")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved -> {save_path}")
