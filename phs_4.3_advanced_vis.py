import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import os
import traceback

# --- Configuration ---
detailed_results_file = 'detailed_results_p2_p3.pkl'
patients_to_plot = ['P2', 'P3']
stages_to_plot = ['pre', 'post']
class_names = ['Right MI (-1)', 'Left MI (+1)'] # For confusion matrix labels

# t-SNE parameters (can be tuned)
tsne_perplexity = 25 # Typical values between 5 and 50. Lower for smaller datasets.
tsne_n_iter = 1000   # Number of optimization iterations
tsne_random_state = 42

# --- Load Detailed Results ---
print(f"Loading detailed results from {detailed_results_file}...")
try:
    with open(detailed_results_file, 'rb') as f:
        detailed_results = pickle.load(f)
    print("Detailed results loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{detailed_results_file}' not found. Please run the modified Riemannian script first.")
    exit()
except Exception as e:
    print(f"Error loading detailed results: {e}")
    exit()

# --- Plotting Loop ---
print("\n--- Generating Advanced Visualizations ---")

for patient in patients_to_plot:
    if patient not in detailed_results:
        print(f"Warning: No detailed results found for {patient}. Skipping.")
        continue
    for stage in stages_to_plot:
        if stage not in detailed_results[patient]:
            print(f"Warning: No detailed results found for {patient} - {stage}. Skipping.")
            continue

        print(f"\nProcessing plots for {patient} - {stage}...")
        try:
            data = detailed_results[patient][stage]
            true_labels = data['test_labels']
            pred_labels = data['predictions']
            features = data['test_features'] # Tangent space vectors

            # --- 1. Confusion Matrix ---
            print("  Generating Confusion Matrix...")
            cm = confusion_matrix(true_labels, pred_labels, labels=[-1, 1]) # Specify labels order

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d') # 'd' for integer format
            ax.set_title(f'{patient} - {stage} Confusion Matrix\n(Riemannian Beta LR)')
            plt.tight_layout()
            cm_filename = f'confusion_matrix_{patient}_{stage}.png'
            plt.savefig(cm_filename)
            print(f"    Saved plot: {cm_filename}")
            plt.close(fig)

            # --- 2. t-SNE Feature Visualization ---
            print("  Generating t-SNE plot...")
            # Check if number of samples is sufficient for perplexity
            n_samples = features.shape[0]
            current_perplexity = min(tsne_perplexity, n_samples - 1) # Perplexity must be less than n_samples
            if current_perplexity <= 0:
                 print(f"    Skipping t-SNE: Not enough samples ({n_samples})")
                 continue

            tsne = TSNE(n_components=2,
                        perplexity=current_perplexity,
                        n_iter=tsne_n_iter,
                        random_state=tsne_random_state,
                        init='pca', # PCA initialization is often more stable
                        learning_rate='auto')

            features_2d = tsne.fit_transform(features)

            # Create scatter plot
            plt.figure(figsize=(8, 7))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
            plt.title(f'{patient} - {stage} t-SNE Visualization of Features\n(Riemannian Beta LR - Colors=True Label)')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')

            # Create legend
            handles, _ = scatter.legend_elements()
            legend_labels = [f'{name}' for name in class_names] # Map -1, 1 to names
            # Adjust legend labels based on unique values present if necessary
            unique_labels = np.unique(true_labels)
            if len(handles) == len(unique_labels):
                 legend_map = {1: 'Left MI (+1)', -1: 'Right MI (-1)'}
                 legend_labels_ordered = [legend_map[lbl] for lbl in sorted(unique_labels)]
                 plt.legend(handles, legend_labels_ordered, title="True Class")
            else:
                 # Fallback if legend elements don't match unique labels perfectly
                 plt.legend(handles, [f'Class {int(l)}' for l in sorted(unique_labels)], title="True Class")


            plt.grid(True, linestyle='--', alpha=0.5)
            tsne_filename = f'tsne_features_{patient}_{stage}.png'
            plt.savefig(tsne_filename)
            print(f"    Saved plot: {tsne_filename}")
            plt.close()

        except Exception as e:
            print(f"  Error generating plots for {patient} {stage}: {e}")
            traceback.print_exc()


print("\n--- Advanced Visualization Complete ---")