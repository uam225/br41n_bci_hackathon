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

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
cov_window_start_sec = 2.0
cov_window_end_sec = 6.0
classifier_choice = 'LR'
data_key_to_use = 'epochs' # Use original data

# Frequency band configuration (set one pair at a time)
# Option 1: Mu band
#freq_band = [8, 12]
# Option 2: Beta band
# freq_band = [13, 30]
# Option 3: Full band (original - run this if you want to compare easily)
freq_band = None # Set to None to skip filtering

filter_order = 5 # Butterworth filter order

# --- Load Processed Data ---
# (Keep loading code as before)
print(f"Loading processed data from {input_data_file}...")
try:
    with open(input_data_file, 'rb') as f:
        all_data = pickle.load(f)
    print("Data loaded successfully.")
except FileNotFoundError: print(f"Error: Data file '{input_data_file}' not found."); exit()
except Exception as e: print(f"Error loading data: {e}"); exit()

# --- Filter Function ---
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a bandpass filter to the data.

    Args:
        data (np.ndarray): Input data [..., n_samples]. Filter is applied along the last axis.
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (float): Sampling rate.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Check if cutoff frequencies are valid
    if low <= 0 or high >= 1 or low >= high:
        print(f"Warning: Invalid frequency band [{lowcut}, {highcut}] Hz for fs={fs} Hz. Skipping filter.")
        return data
    b, a = butter(order, [low, high], btype='band')
    # Apply filter along the time axis (last axis)
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data


# --- Pipeline Function ---
def run_riemannian_pipeline(patient_id, stage, data_key, all_data, fs,
                            cov_start_sample, cov_end_sample, classifier_type,
                            freq_band_to_use=None, filter_order_to_use=5): # Added filter args
    """
    Runs the Riemannian Tangent Space pipeline, optionally filtering first.
    """
    pipeline_name = f'Riemannian({classifier_type})'
    if freq_band_to_use:
        pipeline_name += f'_{freq_band_to_use[0]}-{freq_band_to_use[1]}Hz'

    print(f"\n--- Running {pipeline_name} for {patient_id} - {stage} ({data_key}) ---")
    try:
        # Get data
        train_data = all_data.get(patient_id, {}).get(stage, {}).get('training')
        test_data = all_data.get(patient_id, {}).get(stage, {}).get('test')
        if not train_data or not test_data: return None
        if data_key not in train_data or data_key not in test_data: return None
        train_epochs_list = train_data[data_key]
        train_labels = np.array(train_data['labels'])
        test_epochs_list = test_data[data_key]
        test_labels = np.array(test_data['labels'])
        if not train_epochs_list or not test_epochs_list: return None

        # Convert epoch lists to 3D numpy arrays [n_epochs, n_channels, n_samples]
        train_epochs_array = np.array(train_epochs_list).transpose(0, 2, 1)
        test_epochs_array = np.array(test_epochs_list).transpose(0, 2, 1)

        # --- Optional: Bandpass Filtering ---
        if freq_band_to_use:
            print(f"  Applying bandpass filter: {freq_band_to_use} Hz")
            low, high = freq_band_to_use
            # Filter along the last axis (time samples)
            train_epochs_array = bandpass_filter(train_epochs_array, low, high, fs, order=filter_order_to_use)
            test_epochs_array = bandpass_filter(test_epochs_array, low, high, fs, order=filter_order_to_use)

        # --- 1. Select data within the covariance window ---
        train_epochs_windowed = train_epochs_array[:, :, cov_start_sample:cov_end_sample]
        test_epochs_windowed = test_epochs_array[:, :, cov_start_sample:cov_end_sample]
        # print(f"  Data shape for covariance: {train_epochs_windowed.shape}") # Optional print

        # --- 2. Build the pyriemann pipeline ---
        if classifier_type == 'LR':
            clf = LogisticRegression(solver='liblinear', random_state=42)
        elif classifier_type == 'SVM':
            clf = SVC(kernel='linear', C=1.0, probability=False, random_state=42)
        else: return None

        pipeline = make_pipeline(Covariances(estimator='oas'), TangentSpace(metric='riemann'), clf)

        # --- 3. Train & 4. Test ---
        print("  Training and Testing Riemannian pipeline...")
        pipeline.fit(train_epochs_windowed, train_labels)
        predictions = pipeline.predict(test_epochs_windowed)

        # --- 5. Calculate Accuracy ---
        accuracy = np.mean(predictions == test_labels) * 100
        print(f"  >>> Accuracy: {accuracy:.2f}% <<<")
        return accuracy, pipeline_name # Return pipeline name for results table

    except Exception as e:
        print(f"  An error occurred during pipeline execution for {patient_id} {stage}: {e}")
        traceback.print_exc()
        return None, pipeline_name # Return None accuracy but keep name

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
if freq_band: print(f"Using frequency band: {freq_band} Hz")
else: print("Using full band (no filtering)")


for patient in patients:
    for stage in stages:
        accuracy, pipeline_name_used = run_riemannian_pipeline(patient, stage, data_key_to_use, all_data, fs,
                                           cov_start_sample, cov_end_sample, classifier_choice,
                                           freq_band_to_use=freq_band, filter_order_to_use=filter_order) # Pass filter args
        results.append({
            'Patient': patient,
            'Stage': stage,
            'Pipeline': pipeline_name_used, # Use returned name
            'Data': data_key_to_use,
            'Accuracy': accuracy if accuracy is not None else np.nan
        })

print("\n--- All Pipeline Runs Complete ---")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string())

# Optional: Save results
band_suffix = f"{freq_band[0]}-{freq_band[1]}Hz" if freq_band else "fullband"
results_df.to_csv(f'riemannian_{classifier_choice}_{band_suffix}_results.csv', index=False)
print(f"\nResults saved to riemannian_{classifier_choice}_{band_suffix}_results.csv")