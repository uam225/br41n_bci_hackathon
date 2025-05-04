import pickle
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from scipy.signal import butter, filtfilt # Import filter functions
import os
import traceback
import pandas as pd

# --- ADDED: Dictionary to store detailed results ---
detailed_results_store = {}

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
cov_window_start_sec = 2.0
cov_window_end_sec = 6.0
classifier_choice = 'LR'
data_key_to_use = 'epochs' # Use original data

# Frequency band configuration - SET TO BETA BAND FOR BEST RESULTS
freq_band = [13, 30]
# freq_band = [8, 12]
# freq_band = None

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
# (Keep the bandpass_filter function as is)
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1 or low >= high: return data
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data


# --- Pipeline Function ---
# --- MODIFIED: Function signature and return values ---
def run_riemannian_pipeline(patient_id, stage, data_key, all_data, fs,
                            cov_start_sample, cov_end_sample, classifier_type,
                            freq_band_to_use=None, filter_order_to_use=5):
    """
    Runs the Riemannian Tangent Space pipeline, optionally filtering first.
    Returns accuracy, pipeline name, and detailed results for P2/P3.
    """
    pipeline_name = f'Riemannian({classifier_type})'
    if freq_band_to_use:
        pipeline_name += f'_{freq_band_to_use[0]}-{freq_band_to_use[1]}Hz'

    print(f"\n--- Running {pipeline_name} for {patient_id} - {stage} ({data_key}) ---")
    try:
        # Get data (keep this part)
        train_data = all_data.get(patient_id, {}).get(stage, {}).get('training')
        test_data = all_data.get(patient_id, {}).get(stage, {}).get('test')
        if not train_data or not test_data: return None, pipeline_name, None # MODIFIED RETURN
        if data_key not in train_data or data_key not in test_data: return None, pipeline_name, None # MODIFIED RETURN
        train_epochs_list = train_data[data_key]
        train_labels = np.array(train_data['labels'])
        test_epochs_list = test_data[data_key]
        test_labels = np.array(test_data['labels'])
        if not train_epochs_list or not test_epochs_list: return None, pipeline_name, None # MODIFIED RETURN

        # Convert epoch lists (keep this part)
        train_epochs_array = np.array(train_epochs_list).transpose(0, 2, 1)
        test_epochs_array = np.array(test_epochs_list).transpose(0, 2, 1)

        # Optional: Bandpass Filtering (keep this part)
        if freq_band_to_use:
            print(f"  Applying bandpass filter: {freq_band_to_use} Hz")
            low, high = freq_band_to_use
            train_epochs_array = bandpass_filter(train_epochs_array, low, high, fs, order=filter_order_to_use)
            test_epochs_array = bandpass_filter(test_epochs_array, low, high, fs, order=filter_order_to_use)

        # Select data within the covariance window (keep this part)
        train_epochs_windowed = train_epochs_array[:, :, cov_start_sample:cov_end_sample]
        test_epochs_windowed = test_epochs_array[:, :, cov_start_sample:cov_end_sample]

        # --- MODIFIED: Feature Extraction and Classification ---
        print("  Extracting tangent space features...")
        # Define the feature extractor pipeline
        pipe_feature_extractor = make_pipeline(
            Covariances(estimator='oas'),
            TangentSpace(metric='riemann')
        )
        # Fit the extractor on TRAINING data and transform BOTH train and test
        pipe_feature_extractor.fit(train_epochs_windowed, train_labels)
        train_features = pipe_feature_extractor.transform(train_epochs_windowed)
        test_features = pipe_feature_extractor.transform(test_epochs_windowed) # Get test features

        print("  Training and Testing Classifier...")
        # Define the classifier
        if classifier_type == 'LR':
            clf = LogisticRegression(solver='liblinear', random_state=42)
        elif classifier_type == 'SVM':
             clf = SVC(kernel='linear', C=1.0, probability=False, random_state=42)
        else:
            print(f"  Error: Unknown classifier type '{classifier_type}'")
            return None, pipeline_name, None # MODIFIED RETURN

        # Train the classifier on TRAINING features
        clf.fit(train_features, train_labels)

        # Predict on TEST features
        predictions = clf.predict(test_features)

        # Calculate Accuracy (keep this part)
        accuracy = np.mean(predictions == test_labels) * 100
        print(f"  >>> Accuracy: {accuracy:.2f}% <<<")

        # --- ADDED: Store detailed results ONLY for P2 and P3 ---
        detailed_output = None
        if patient_id in ['P2', 'P3']:
            detailed_output = {
                'test_labels': test_labels,
                'predictions': predictions,
                'test_features': test_features # Tangent space vectors
            }
        # --- MODIFIED RETURN ---
        return accuracy, pipeline_name, detailed_output

    except Exception as e:
        print(f"  An error occurred during pipeline execution for {patient_id} {stage}: {e}")
        traceback.print_exc()
        # --- MODIFIED RETURN ---
        return None, pipeline_name, None

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
if freq_band: print(f"Using frequency band: {freq_band} Hz") # Ensure freq_band is set correctly above
else: print("Using full band (no filtering)")


for patient in patients:
    # --- ADDED: Initialize patient dict in detailed store ---
    detailed_results_store[patient] = {}
    for stage in stages:
        # --- MODIFIED: Capture detailed_output ---
        accuracy, pipeline_name_used, detailed_output = run_riemannian_pipeline(
            patient, stage, data_key_to_use, all_data, fs,
            cov_start_sample, cov_end_sample, classifier_choice,
            freq_band_to_use=freq_band, filter_order_to_use=filter_order
        )
        results.append({
            'Patient': patient,
            'Stage': stage,
            'Pipeline': pipeline_name_used,
            'Data': data_key_to_use,
            'Accuracy': accuracy if accuracy is not None else np.nan
        })
        # --- ADDED: Store detailed results if generated ---
        if detailed_output is not None:
             detailed_results_store[patient][stage] = detailed_output


print("\n--- All Pipeline Runs Complete ---")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string())

# Optional: Save results (keep this part)
band_suffix = f"{freq_band[0]}-{freq_band[1]}Hz" if freq_band else "fullband"
results_df.to_csv(f'riemannian_{classifier_choice}_{band_suffix}_results.csv', index=False)
print(f"\nResults saved to riemannian_{classifier_choice}_{band_suffix}_results.csv")

# --- ADDED: Save Detailed Results ---
detailed_output_file = 'detailed_results_p2_p3.pkl'
print(f"\nSaving detailed results for P2/P3 to {detailed_output_file}...")
try:
    # Filter the store to only keep P2 and P3
    filtered_store = {p: data for p, data in detailed_results_store.items() if p in ['P2', 'P3']}
    with open(detailed_output_file, 'wb') as f:
        pickle.dump(filtered_store, f, pickle.HIGHEST_PROTOCOL)
    print("Detailed results saved successfully.")
except Exception as e:
    print(f"Error saving detailed results: {e}")