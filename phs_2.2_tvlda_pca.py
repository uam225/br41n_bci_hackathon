import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import traceback
import pandas as pd # For storing and displaying results

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
# Define the time window within the epoch for TVLDA classification (in seconds, relative to epoch start)
# Paper used 1.5s to 8s post-trigger. Our epochs start 1s post-trigger.
# So, relative to epoch start, this is 0.5s to 7.0s.
classification_window_start_sec = 2.0
classification_window_end_sec = 6.0
# Number of PCA components to keep
n_pca_components = 8 # Let's start with 4, can be tuned
data_key_to_use = 'epochs' # Use whitened data for this pipeline

# --- Load Processed Data ---
print(f"Loading processed data from {input_data_file}...")
try:
    with open(input_data_file, 'rb') as f:
        all_data = pickle.load(f)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Data file '{input_data_file}' not found. Please run the Phase 1 script first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# --- Pipeline Function ---
def run_tvlda_pca_pipeline(patient_id, stage, data_key, all_data, fs,
                           class_win_start_sample, class_win_end_sample, n_components):
    """
    Runs the PCA + TVLDA pipeline for a given patient/stage using specified data key.

    Args:
        patient_id (str): Patient ID ('P1', 'P2', 'P3').
        stage (str): Stage ('pre', 'post').
        data_key (str): Key for epoch data ('epochs' or 'epochs_whitened').
        all_data (dict): The main data dictionary.
        fs (float): Sampling rate.
        class_win_start_sample (int): Start sample index for classification window.
        class_win_end_sample (int): End sample index for classification window.
        n_components (int): Number of PCA components.

    Returns:
        float: Classification accuracy (%), or None if an error occurs.
    """
    print(f"\n--- Running PCA+TVLDA for {patient_id} - {stage} ({data_key}) ---")
    try:
        # Get data, ensuring it exists and is not None
        train_data = all_data.get(patient_id, {}).get(stage, {}).get('training')
        test_data = all_data.get(patient_id, {}).get(stage, {}).get('test')

        if not train_data or not test_data:
             print(f"  Error: Missing training or testing data for {patient_id} {stage}.")
             return None
        # Check if the specific data_key (e.g., 'epochs_whitened') exists
        if data_key not in train_data or data_key not in test_data:
             print(f"  Error: Data key '{data_key}' not found for {patient_id} {stage}. Whitening might have failed.")
             return None

        train_epochs_list = train_data[data_key]
        train_labels = np.array(train_data['labels'])
        test_epochs_list = test_data[data_key]
        test_labels = np.array(test_data['labels'])

        if not train_epochs_list or not test_epochs_list:
             print(f"  Error: Empty epoch list found for {patient_id} {stage} using key '{data_key}'.")
             return None

        # Convert epoch lists to 3D numpy arrays [n_epochs, n_channels, n_samples]
        # Original shape from list: [n_epochs, n_samples_epoch, n_channels]
        # Transpose needed: [n_epochs, n_channels, n_samples_epoch]
        train_epochs_array = np.array(train_epochs_list).transpose(0, 2, 1)
        test_epochs_array = np.array(test_epochs_list).transpose(0, 2, 1)
        n_epochs_train, n_channels, n_samples_epoch = train_epochs_array.shape
        n_epochs_test = test_epochs_array.shape[0]

        # --- 1. PCA Spatial Filtering ---
        print(f"  Training PCA with {n_components} components...")
        # Reshape training data for PCA fit: [n_epochs * n_samples, n_channels]
        # Need to transpose epochs first: [n_epochs, n_samples, n_channels] -> [n_channels, n_epochs*n_samples] -> transpose
        pca_train_data = train_epochs_array.transpose(1, 0, 2).reshape(n_channels, -1).T
        pca = PCA(n_components=n_components)
        pca.fit(pca_train_data)
        print(f"  PCA Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.3f}")

        # Apply PCA transform to each epoch
        print("  Applying PCA transform...")
        train_epochs_pca = np.zeros((n_epochs_train, n_components, n_samples_epoch))
        for i in range(n_epochs_train):
            # Input to transform: [n_samples, n_channels]
            epoch_original_shape = train_epochs_array[i, :, :].T
            train_epochs_pca[i, :, :] = pca.transform(epoch_original_shape).T # Output: [n_components, n_samples]

        test_epochs_pca = np.zeros((n_epochs_test, n_components, n_samples_epoch))
        for i in range(n_epochs_test):
            epoch_original_shape = test_epochs_array[i, :, :].T
            test_epochs_pca[i, :, :] = pca.transform(epoch_original_shape).T

        print(f"  PCA training data shape: {train_epochs_pca.shape}") # [n_epochs, n_components, n_samples]
        print(f"  PCA testing data shape: {test_epochs_pca.shape}")

        # --- 2. TVLDA Implementation ---
        print("  Running Time-Variant LDA...")
        n_samples_in_window = class_win_end_sample - class_win_start_sample
        # Store LDA scores for each test epoch at each time point in the window
        test_scores_over_time = np.zeros((n_epochs_test, n_samples_in_window))

        # Loop through each time sample in the classification window
        for t_idx, sample_idx in enumerate(range(class_win_start_sample, class_win_end_sample)):
            # Extract features for this time point: [n_epochs, n_components]
            X_train_t = train_epochs_pca[:, :, sample_idx]
            X_test_t = test_epochs_pca[:, :, sample_idx]

            # Train LDA for this time point
            lda_t = LDA()
            try:
                lda_t.fit(X_train_t, train_labels)
                # Get the raw decision function score (distance from hyperplane)
                test_scores_over_time[:, t_idx] = lda_t.decision_function(X_test_t)
            except Exception as lda_e:
                 # Handle potential errors like singular matrices at a specific time point
                 print(f"    Warning: LDA failed at sample index {sample_idx}. Setting scores to 0. Error: {lda_e}")
                 test_scores_over_time[:, t_idx] = 0 # Assign neutral score

        # Sum scores across the time window for each test epoch
        final_tvlda_scores = np.sum(test_scores_over_time, axis=1)

        # Classify based on the sign of the summed score
        # Map score to labels (+1 / -1). Handle score == 0 (assign to one class, e.g., -1)
        predictions = np.sign(final_tvlda_scores)
        predictions[predictions == 0] = -1 # Assign 0 scores to class -1 (arbitrary choice)

        # Calculate accuracy
        accuracy = np.mean(predictions == test_labels) * 100
        print(f"  >>> Accuracy: {accuracy:.2f}% <<<")
        return accuracy

    except Exception as e:
        print(f"  An error occurred during pipeline execution for {patient_id} {stage}: {e}")
        traceback.print_exc()
        return None

# --- Main Execution Loop ---
results = []
patients = ['P1', 'P2', 'P3']
stages = ['pre', 'post']

# Get sampling rate (assuming it's consistent)
try:
    # Try to get fs from a condition likely to exist and have whitened data
    fs = all_data['P1']['pre']['training']['fs']
except KeyError:
    print("Error: Cannot determine sampling rate from loaded data.")
    exit()

# Calculate sample indices for classification window
class_win_start_sample = int(classification_window_start_sec * fs)
class_win_end_sample = int(classification_window_end_sec * fs)
print(f"\nUsing classification window: {class_win_start_sample} to {class_win_end_sample} samples ({classification_window_start_sec}s to {classification_window_end_sec}s within epoch)")
print(f"Using data key: '{data_key_to_use}'")
print(f"Using PCA components: {n_pca_components}")

for patient in patients:
    for stage in stages:
        accuracy = run_tvlda_pca_pipeline(patient, stage, data_key_to_use, all_data, fs,
                                          class_win_start_sample, class_win_end_sample, n_pca_components)
        results.append({
            'Patient': patient,
            'Stage': stage,
            'Pipeline': f'PCA({n_pca_components})+TVLDA',
            'Data': data_key_to_use,
            'Accuracy': accuracy if accuracy is not None else np.nan
        })

print("\n--- All Pipeline Runs Complete ---")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string())

# Optional: Append results to the previous CSV or save separately
# try:
#     # Load previous results if they exist
#     prev_results_df = pd.read_csv('csp_lda_results.csv')
#     combined_results_df = pd.concat([prev_results_df, results_df], ignore_index=True)
# except FileNotFoundError:
#     print("Previous results file not found, saving only current results.")
#     combined_results_df = results_df

# combined_results_df.to_csv('combined_pipeline_results.csv', index=False)
# print("\nCombined results saved to combined_pipeline_results.csv")
# Or save separately:
results_df.to_csv('tvlda_pca_results.csv', index=False)
print("\nTVLDA results saved to tvlda_pca_results.csv")