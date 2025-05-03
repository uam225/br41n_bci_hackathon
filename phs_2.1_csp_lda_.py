import pickle
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import traceback
import pandas as pd # For storing and displaying results

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
feature_window_start_sec = 2.5 # Start 2s into epoch (3s post-trigger)
feature_window_end_sec = 6.0   # End 6s into epoch (7s post-trigger)
n_csp_pairs = 2 # Use 3 pairs (6 filters total)
data_key_to_use = 'epochs' # Specify 'epochs' for original or 'epochs_whitened'

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

# --- CSP Function ---
# (Keep the calculate_csp_filters function as before)
def calculate_csp_filters(epochs_class1, epochs_class2, n_filters):
    """Calculates Common Spatial Pattern (CSP) filters."""
    n_epochs1, n_channels, n_samples1 = epochs_class1.shape
    n_epochs2, _, n_samples2 = epochs_class2.shape
    if n_samples1 != n_samples2: return None
    cov_class1 = np.zeros((n_channels, n_channels))
    for epoch in epochs_class1: cov_class1 += np.cov(epoch) / n_epochs1
    cov_class2 = np.zeros((n_channels, n_channels))
    for epoch in epochs_class2: cov_class2 += np.cov(epoch) / n_epochs2
    try:
        eigenvalues, eigenvectors = eigh(cov_class1, cov_class1 + cov_class2)
    except np.linalg.LinAlgError as e:
        print(f"  Error during eigenvalue decomposition: {e}")
        return None
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    filter_indices = np.concatenate((np.arange(0, n_filters // 2),
                                     np.arange(n_channels - n_filters // 2, n_channels)))
    csp_filters = eigenvectors_sorted[:, filter_indices]
    return csp_filters

# --- Pipeline Function ---
def run_csp_lda_pipeline(patient_id, stage, data_key, all_data, fs, start_sample, end_sample, n_total_filters):
    """
    Runs the CSP+LDA pipeline for a given patient/stage.

    Args:
        patient_id (str): Patient ID ('P1', 'P2', 'P3').
        stage (str): Stage ('pre', 'post').
        data_key (str): Key for epoch data ('epochs' or 'epochs_whitened').
        all_data (dict): The main data dictionary.
        fs (float): Sampling rate.
        start_sample (int): Start sample index for feature window.
        end_sample (int): End sample index for feature window.
        n_total_filters (int): Total number of CSP filters.

    Returns:
        float: Classification accuracy (%), or None if an error occurs.
    """
    print(f"\n--- Running CSP+LDA for {patient_id} - {stage} ({data_key}) ---")
    try:
        # Get data, ensuring it exists and is not None
        train_data = all_data.get(patient_id, {}).get(stage, {}).get('training')
        test_data = all_data.get(patient_id, {}).get(stage, {}).get('test')

        if not train_data or not test_data:
             print(f"  Error: Missing training or testing data for {patient_id} {stage}.")
             return None
        if data_key not in train_data or data_key not in test_data:
             print(f"  Error: Data key '{data_key}' not found for {patient_id} {stage}.")
             return None

        train_epochs_list = train_data[data_key]
        train_labels = np.array(train_data['labels'])
        test_epochs_list = test_data[data_key]
        test_labels = np.array(test_data['labels'])

        if not train_epochs_list or not test_epochs_list:
             print(f"  Error: Empty epoch list found for {patient_id} {stage} using key '{data_key}'.")
             return None

        # Convert epoch lists to 3D numpy arrays [n_epochs, n_channels, n_samples]
        train_epochs_array = np.array(train_epochs_list).transpose(0, 2, 1)
        test_epochs_array = np.array(test_epochs_list).transpose(0, 2, 1)

        # --- CSP Training ---
        train_epochs_windowed = train_epochs_array[:, :, start_sample:end_sample]
        epochs_class0 = train_epochs_windowed[train_labels == 1] # Left MI
        epochs_class1 = train_epochs_windowed[train_labels == -1] # Right MI

        # Check if both classes have data
        if epochs_class0.shape[0] == 0 or epochs_class1.shape[0] == 0:
             print(f"  Error: Not enough data for both classes in training set for {patient_id} {stage}.")
             return None

        csp_matrix = calculate_csp_filters(epochs_class0, epochs_class1, n_total_filters)

        if csp_matrix is None:
            print(f"  CSP calculation failed for {patient_id} {stage}.")
            return None

        # --- Feature Extraction ---
        train_epochs_csp = np.einsum('fc,ecs->efs', csp_matrix.T, train_epochs_windowed)
        test_epochs_windowed = test_epochs_array[:, :, start_sample:end_sample]
        test_epochs_csp = np.einsum('fc,ecs->efs', csp_matrix.T, test_epochs_windowed)
        train_features = np.log(np.var(train_epochs_csp, axis=2))
        test_features = np.log(np.var(test_epochs_csp, axis=2))

        # Check for NaNs or Infs in features (can happen if variance is zero or negative after log)
        if np.any(np.isnan(train_features)) or np.any(np.isinf(train_features)) or \
           np.any(np.isnan(test_features)) or np.any(np.isinf(test_features)):
            print(f"  Warning: NaN or Inf found in features for {patient_id} {stage}. Replacing with 0.")
            train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
            test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)


        # --- LDA Classification ---
        lda = LDA()
        lda.fit(train_features, train_labels)
        predictions = lda.predict(test_features)
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
n_total_filters = n_csp_pairs * 2

# Get sampling rate (assuming it's consistent)
try:
    fs = all_data['P1']['pre']['training']['fs']
except KeyError:
    print("Error: Cannot determine sampling rate from loaded data.")
    exit()

# Calculate sample indices for feature window
start_sample = int(feature_window_start_sec * fs)
end_sample = int(feature_window_end_sec * fs)
print(f"\nUsing feature window: {start_sample} to {end_sample} samples ({feature_window_start_sec}s to {feature_window_end_sec}s within epoch)")
print(f"Using data key: '{data_key_to_use}'")

for patient in patients:
    for stage in stages:
        accuracy = run_csp_lda_pipeline(patient, stage, data_key_to_use, all_data, fs, start_sample, end_sample, n_total_filters)
        results.append({
            'Patient': patient,
            'Stage': stage,
            'Pipeline': 'CSP+LDA',
            'Data': data_key_to_use,
            'Accuracy': accuracy if accuracy is not None else np.nan # Store NaN if failed
        })

print("\n--- All Pipeline Runs Complete ---")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string()) # Use to_string to print full DataFrame

# Save results to CSV
results_df.to_csv('csp_lda_results.csv', index=False)
print("\nResults saved to csp_lda_results.csv")