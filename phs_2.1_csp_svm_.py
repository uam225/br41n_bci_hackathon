import pickle
import numpy as np
from scipy.linalg import eigh
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # Remove LDA import
from sklearn.svm import SVC # Import Support Vector Classification
import os
import traceback
import pandas as pd

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
feature_window_start_sec = 2.0 # Optimized window
feature_window_end_sec = 6.0   # Optimized window
n_csp_pairs = 2 # Optimized number of pairs
data_key_to_use = 'epochs' # Use original data

# --- Load Processed Data ---
# (Keep the loading code as before)
print(f"Loading processed data from {input_data_file}...")
try:
    with open(input_data_file, 'rb') as f:
        all_data = pickle.load(f)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Data file '{input_data_file}' not found.")
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
def run_csp_svm_pipeline(patient_id, stage, data_key, all_data, fs, start_sample, end_sample, n_total_filters):
    """
    Runs the CSP+SVM pipeline for a given patient/stage.
    """
    print(f"\n--- Running CSP+SVM for {patient_id} - {stage} ({data_key}) ---") # Changed name
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

        if np.any(np.isnan(train_features)) or np.any(np.isinf(train_features)) or \
           np.any(np.isnan(test_features)) or np.any(np.isinf(test_features)):
            print(f"  Warning: NaN or Inf found in features for {patient_id} {stage}. Replacing with 0.")
            train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
            test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)

        # --- SVM Classification ---  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CHANGE HERE
        print("  Training and testing SVM classifier...")
        # 7. Train SVM on training features
        #    Using a linear kernel is often a good start, similar to LDA
        #    Regularization parameter C can be tuned, default is 1.0
        #    probability=False can be faster if probabilities aren't needed
        svm = SVC(kernel='linear', C=1.0, probability=False, random_state=42)
        svm.fit(train_features, train_labels) # Use original +1/-1 labels

        # 8. Predict on test features
        predictions = svm.predict(test_features)

        # 9. Calculate accuracy
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
print(f"Using CSP pairs: {n_csp_pairs}")

for patient in patients:
    for stage in stages:
        # Call the updated pipeline function
        accuracy = run_csp_svm_pipeline(patient, stage, data_key_to_use, all_data, fs, start_sample, end_sample, n_total_filters)
        results.append({
            'Patient': patient,
            'Stage': stage,
            'Pipeline': f'CSP({n_total_filters})+SVM', # Updated pipeline name
            'Data': data_key_to_use,
            'Accuracy': accuracy if accuracy is not None else np.nan
        })

print("\n--- All Pipeline Runs Complete ---")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string())

# Optional: Save results
results_df.to_csv('csp_svm_results.csv', index=False)
print("\nCSP+SVM results saved to csp_svm_results.csv")