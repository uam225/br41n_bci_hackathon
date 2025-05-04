import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For potentially nicer plots

# --- Data for Plots ---
# (Manually entered based on our final results)

# Average accuracies for overall comparison
pipeline_avg_accuracies = {
    'CSP+LDA (Orig, Best)': 70.17,
    'PCA+TVLDA (Orig, Best)': 63.33,
    'Riemannian LR (Full Band)': 75.21,
    'Riemannian LR (Beta Band)': 89.58 # Our Winner
}

# Pre vs Post accuracies for the winning pipeline (Riemannian Beta LR)
pre_post_data = {
    'Patient': ['P1', 'P1', 'P2', 'P2', 'P3', 'P3'],
    'Stage': ['Pre', 'Post', 'Pre', 'Post', 'Pre', 'Post'],
    'Accuracy': [96.25, 88.75, 80.00, 100.00, 76.25, 96.25]
}
pre_post_df = pd.DataFrame(pre_post_data)

# --- Plot 1: Overall Pipeline Comparison ---

plt.figure(figsize=(10, 6))
pipelines = list(pipeline_avg_accuracies.keys())
avg_accuracies = list(pipeline_avg_accuracies.values())

bars = plt.bar(pipelines, avg_accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.ylabel('Average Accuracy (%)')
plt.title('Average BCI Decoding Accuracy Across Pipeline Configurations')
plt.ylim(0, 100)
plt.xticks(rotation=15, ha='right') # Rotate labels slightly if long

# Add accuracy values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom', ha='center') # Adjust position as needed

plt.tight_layout()
plt.savefig('plot_pipeline_comparison_avg.png')
print("Saved plot_pipeline_comparison_avg.png")
plt.close()

# --- Plot 2: Pre vs. Post Therapy Effect (Winning Pipeline) ---

plt.figure(figsize=(8, 6))
sns.barplot(x='Patient', y='Accuracy', hue='Stage', data=pre_post_df, palette='viridis')

plt.ylabel('Accuracy (%)')
plt.title('Pre vs. Post Therapy Accuracy (Riemannian Beta LR Pipeline)')
plt.ylim(0, 105) # Extend ylim slightly above 100
plt.legend(title='Therapy Stage', loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add accuracy values on top of bars (more complex for grouped bars)
# Optional: Add text labels manually if needed, or skip for clarity
# for container in plt.gca().containers:
#     plt.gca().bar_label(container, fmt='%.2f%%')

plt.tight_layout()
plt.savefig('plot_pre_post_comparison_best_pipeline.png')
print("Saved plot_pre_post_comparison_best_pipeline.png")
plt.close()

print("\nPlot generation complete.")