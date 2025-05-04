import pickle
import numpy as np
# Import pyriemann components
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
# Import standard classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline # Useful for chaining steps
import os
import traceback
import pandas as pd

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
# Define the time window for covariance calculation (relative to epoch start)
# Let's use the same window as optimized CSP: 2.0s to 6.0s
cov_window_start_sec = 2.0
cov_window_end_sec = 6.0
# Classifier choice: 'LR' (Logistic Regression) or 'SVM' (Linear SVM)
classifier_choice = 'LR'
# Data key: Start with original data
data_key_to_use = 'epochs_whitened'

# --- Load Processed Data ---
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

# --- Pipeline Function ---
def run_riemannian_pipeline(patient_id, stage, data_key, all_data, fs,
                            cov_start_sample, cov_end_sample, classifier_type):
    """
    Runs the Riemannian Tangent Space pipeline for a given patient/stage.

    Args:
        # ... (similar args as before) ...
        classifier_type (str): 'LR' or 'SVM'.

    Returns:
        float: Classification accuracy (%), or None if an error occurs.
    """
    print(f"\n--- Running Riemannian ({classifier_type}) for {patient_id} - {stage} ({data_key}) ---")
    try:
        # Get data
        train_data = all_data.get(patient_id, {}).get(stage, {}).get('training')
        test_data = all_data.get(patient_id, {}).get(stage, {}).get('test')

        if not train_data or not test_data: return None # Basic check
        if data_key not in train_data or data_key not in test_data: return None

        train_epochs_list = train_data[data_key]
        train_labels = np.array(train_data['labels'])
        test_epochs_list = test_data[data_key]
        test_labels = np.array(test_data['labels'])

        if not train_epochs_list or not test_epochs_list: return None

        # Convert epoch lists to 3D numpy arrays [n_epochs, n_channels, n_samples]
        train_epochs_array = np.array(train_epochs_list).transpose(0, 2, 1)
        test_epochs_array = np.array(test_epochs_list).transpose(0, 2, 1)

        # --- 1. Select data within the covariance window ---
        train_epochs_windowed = train_epochs_array[:, :, cov_start_sample:cov_end_sample]
        test_epochs_windowed = test_epochs_array[:, :, cov_start_sample:cov_end_sample]
        print(f"  Data shape for covariance: {train_epochs_windowed.shape}")

        # --- 2. Build the pyriemann pipeline ---
        #    a) Calculate Covariances ('oas' is often robust)
        #    b) Project to Tangent Space
        #    c) Classify
        if classifier_type == 'LR':
            clf = LogisticRegression(solver='liblinear', random_state=42) # Liblinear often good for smaller datasets
        elif classifier_type == 'SVM':
            clf = SVC(kernel='linear', C=1.0, probability=False, random_state=42)
        else:
            print(f"  Error: Unknown classifier type '{classifier_type}'")
            return None

        # Create the scikit-learn compatible pipeline
        pipeline = make_pipeline(
            Covariances(estimator='oas'), # OAS: Oracle Approximating Shrinkage
            TangentSpace(metric='riemann'),
            clf # Your chosen classifier
        )

        # --- 3. Train the pipeline ---
        print("  Training Riemannian pipeline...")
        # The pipeline expects input shape [n_epochs, n_channels, n_samples]
        pipeline.fit(train_epochs_windowed, train_labels)

        # --- 4. Test the pipeline ---
        print("  Testing pipeline...")
        predictions = pipeline.predict(test_epochs_windowed)

        # --- 5. Calculate Accuracy ---
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

try:
    fs = all_data['P1']['pre']['training']['fs']
except KeyError:
    print("Error: Cannot determine sampling rate.")
    exit()

cov_start_sample = int(cov_window_start_sec * fs)
cov_end_sample = int(cov_window_end_sec * fs)
print(f"\nUsing covariance window: {cov_start_sample} to {cov_end_sample} samples ({cov_window_start_sec}s to {cov_window_end_sec}s within epoch)")
print(f"Using data key: '{data_key_to_use}'")
print(f"Using classifier: {classifier_choice}")

for patient in patients:
    for stage in stages:
        accuracy = run_riemannian_pipeline(patient, stage, data_key_to_use, all_data, fs,
                                           cov_start_sample, cov_end_sample, classifier_choice)
        results.append({
            'Patient': patient,
            'Stage': stage,
            'Pipeline': f'Riemannian({classifier_choice})',
            'Data': data_key_to_use,
            'Accuracy': accuracy if accuracy is not None else np.nan
        })

print("\n--- All Pipeline Runs Complete ---")

# --- Display Results ---
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df.to_string())

# Optional: Save results
results_df.to_csv(f'riemannian_{classifier_choice}_results.csv', index=False)
print(f"\nRiemannian ({classifier_choice}) results saved to riemannian_{classifier_choice}_results.csv")