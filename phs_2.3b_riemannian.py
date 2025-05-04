import pickle
import numpy as np
# Import pyriemann components
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
# Import standard classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline # Useful for chaining steps
from scipy.signal import butter, filtfilt # Import filter functions
import os
import traceback
import pandas as pd

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
# Define the time window for covariance calculation (relative to epoch start)
cov_window_start_sec = 2.0
cov_window_end_sec = 6.0
# Classifier choice
classifier_choice = 'LR'
# Data key: Use original data
data_key_to_use = 'epochs'
# Define frequency bands
mu_band = [8, 12]
beta_band = [13, 30]
filter_order = 5 # Butterworth filter order

# --- Load Processed Data ---
print(f"Loading processed data from {input_data_file}...")
try:
    with open(input_data_file, 'rb') as f:
        all_data = pickle.load(f)
    print("Data loaded successfully.")
except FileNotFoundError: print(f"Error: Data file '{input_data_file}' not found."); exit()
except Exception as e: print(f"Error loading data: {e}"); exit()

# --- Filter Function ---
# (Keep the bandpass_filter function as before)
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a bandpass filter to the data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1 or low >= high: return data # Skip if invalid
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

# --- Pipeline Function ---
def run_riemannian_combined_pipeline(patient_id, stage, data_key, all_data, fs,
                                     cov_start_sample, cov_end_sample, classifier_type,
                                     freq_band1, freq_band2, filter_order_to_use=5):
    """
    Runs the Riemannian Tangent Space pipeline using combined features
    from two frequency bands.
    """
    pipeline_name = f'Riemannian({classifier_type})_Band1[{freq_band1[0]}-{freq_band1[1]}]+Band2[{freq_band2[0]}-{freq_band2[1]}]'

    print(f"\n--- Running {pipeline_name} for {patient_id} - {stage} ({data_key}) ---")
    try:
        # Get data
        train_data = all_data.get(patient_id, {}).get(stage, {}).get('training')
        test_data = all_data.get(patient_id, {}).get(stage, {}).get('test')
        if not train_data or not test_data: return None, pipeline_name
        if data_key not in train_data or data_key not in test_data: return None, pipeline_name
        train_epochs_list = train_data[data_key]
        train_labels = np.array(train_data['labels'])
        test_epochs_list = test_data[data_key]
        test_labels = np.array(test_data['labels'])
        if not train_epochs_list or not test_epochs_list: return None, pipeline_name

        # Convert epoch lists to 3D numpy arrays [n_epochs, n_channels, n_samples]
        train_epochs_array = np.array(train_epochs_list).transpose(0, 2, 1)
        test_epochs_array = np.array(test_epochs_list).transpose(0, 2, 1)

        # --- Filter data into two bands ---
        print(f"  Filtering into Band 1: {freq_band1} Hz and Band 2: {freq_band2} Hz")
        low1, high1 = freq_band1
        train_epochs_band1 = bandpass_filter(train_epochs_array, low1, high1, fs, order=filter_order_to_use)
        test_epochs_band1 = bandpass_filter(test_epochs_array, low1, high1, fs, order=filter_order_to_use)

        low2, high2 = freq_band2
        train_epochs_band2 = bandpass_filter(train_epochs_array, low2, high2, fs, order=filter_order_to_use)
        test_epochs_band2 = bandpass_filter(test_epochs_array, low2, high2, fs, order=filter_order_to_use)

        # --- Select data within the covariance window ---
        train_epochs_win_b1 = train_epochs_band1[:, :, cov_start_sample:cov_end_sample]
        test_epochs_win_b1 = test_epochs_band1[:, :, cov_start_sample:cov_end_sample]
        train_epochs_win_b2 = train_epochs_band2[:, :, cov_start_sample:cov_end_sample]
        test_epochs_win_b2 = test_epochs_band2[:, :, cov_start_sample:cov_end_sample]

        # --- Feature Extraction Pipelines (Covariance + Tangent Space) ---
        print("  Extracting tangent space features for both bands...")
        pipe_feature_extractor = make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann')
        )

        # Fit and transform for Band 1
        pipe_feature_extractor.fit(train_epochs_win_b1, train_labels) # Fit on training data
        train_features_b1 = pipe_feature_extractor.transform(train_epochs_win_b1)
        test_features_b1 = pipe_feature_extractor.transform(test_epochs_win_b1)

        # Fit and transform for Band 2
        pipe_feature_extractor.fit(train_epochs_win_b2, train_labels) # Re-fit on training data
        train_features_b2 = pipe_feature_extractor.transform(train_epochs_win_b2)
        test_features_b2 = pipe_feature_extractor.transform(test_epochs_win_b2)

        # --- Combine Features ---
        train_features_combined = np.hstack((train_features_b1, train_features_b2))
        test_features_combined = np.hstack((test_features_b1, test_features_b2))
        print(f"  Combined feature vector length: {train_features_combined.shape[1]}")

        # --- Classification ---
        print("  Training and Testing Classifier on combined features...")
        if classifier_type == 'LR':
            clf = LogisticRegression(solver='liblinear', random_state=42)
        else: # Add other classifiers here if needed
            print(f"  Error: Unknown classifier type '{classifier_type}'")
            return None, pipeline_name

        clf.fit(train_features_combined, train_labels)
        predictions = clf.predict(test_features_combined)

        # --- Calculate Accuracy ---
        accuracy = np.mean(predictions == test_labels) * 100
        print(f"  >>> Accuracy: {accuracy:.2f}% <<<")
        return accuracy, pipeline_name

    except Exception as e:
        print(f"  An error occurred during pipeline execution for {patient_id} {stage}: {e}")
        traceback.print_exc()
        return None, pipeline_name

# --- Main Execution Loop ---
results = []
patients = ['P1', 'P2', 'P3']
stages = ['pre', 'post']

try: fs = all_data['P1']['pre']['training']['fs']
except KeyError: print("Error: Cannot determine sampling rate."); exit()

cov_start_sample = int(cov_window_start_sec * fs)
cov_end_sample = int(cov_window_end_sec * fs)
print(f"\nUsing covariance window: {cov_start_sample} to {cov_end_sample} samples ({cov_window_start_sec}s to {cov_window_end_sec}s within epoch)")
print(f"Using data key: '{data_key_to_use}'")
print(f"Using classifier: {classifier_choice}")
print(f"Combining features from bands: {mu_band} Hz and {beta_band} Hz")

for patient in patients:
    for stage in stages:
        # Pass both frequency bands to the pipeline function
        accuracy, pipeline_name_used = run_riemannian_combined_pipeline(
            patient, stage, data_key_to_use, all_data, fs,
            cov_start_sample, cov_end_sample, classifier_choice,
            mu_band, beta_band, filter_order_to_use=filter_order
        )
        results.append({
            'Patient': patient,
            'Stage': stage,
            'Pipeline': pipeline_name_used, # Use descriptive name
            'Data': data_key_to_use,
            'Accuracy': accuracy if accuracy is not None else np.nan
        })

print("\n--- All Pipeline Runs Complete ---")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string())

# Optional: Save results
results_df.to_csv(f'riemannian_{classifier_choice}_Mu+Beta_results.csv', index=False)
print(f"\nResults saved to riemannian_{classifier_choice}_Mu+Beta_results.csv")