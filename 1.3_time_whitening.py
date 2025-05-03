import scipy.io as sio
import numpy as np
import os
import glob
from scipy.signal import lfilter
from statsmodels.regression.linear_model import yule_walker

# --- Configuration ---
# Set the base directory where your patient data is stored
base_data_dir = '/Users/umairarshad/Projects/BCI Hackathon Datasets/BR41N_Hackathon/stroke-rehab/' # !!! IMPORTANT: Update this path if needed
epoch_start_offset_sec = 1.0
epoch_end_offset_sec = 8.0
ar_order = 10 # AR model order for whitening

# --- Function to process a single .mat file (from Phase 1.4) ---
def process_bci_file(file_path, epoch_start_offset_sec, epoch_end_offset_sec):
    """Loads BCI data, extracts epochs, returns epochs, labels, fs."""
    try:
        mat_data = sio.loadmat(file_path)
        fs = mat_data['fs'].flatten()[0]
        y = mat_data['y']
        trig = mat_data['trig'].flatten()
        epoch_start_offset_samples = int(epoch_start_offset_sec * fs)
        epoch_end_offset_samples = int(epoch_end_offset_sec * fs)
        trigger_onsets_idx = np.where((trig != 0) & (np.roll(trig, 1) == 0))[0]
        epochs = []
        labels = []
        for onset_sample_idx in trigger_onsets_idx:
            label = trig[onset_sample_idx]
            if label in [1, -1]:
                epoch_start_sample = onset_sample_idx + epoch_start_offset_samples
                epoch_end_sample = onset_sample_idx + epoch_end_offset_samples
                if epoch_start_sample >= 0 and epoch_end_sample <= y.shape[0]:
                    epoch_data = y[epoch_start_sample:epoch_end_sample, :]
                    epochs.append(epoch_data)
                    labels.append(label)
        return epochs, labels, fs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

# --- Whitening Functions ---

def estimate_ar_coeffs_per_channel(data_matrix, order):
    """
    Estimates AR coefficients for each channel using the Yule-Walker method
    via statsmodels.

    Args:
        data_matrix (np.ndarray): Data array of shape [n_samples x n_channels].
                                  Should be reasonably long for stable estimation.
        order (int): The order of the AR model (P).

    Returns:
        np.ndarray: Array of AR coefficients [a1, a2, ..., aP], shape [n_channels x order].
                    Returns None if estimation fails for any channel.
    """
    n_samples, n_channels = data_matrix.shape
    all_coeffs = np.zeros((n_channels, order))
    # print(f"  Estimating AR({order}) coefficients for {n_channels} channels...")

    for i in range(n_channels):
        try:
            # yule_walker returns rho (AR coefficients including the leading 1) and sigma
            # rho = [1, -a1, -a2, ..., -aP]
            rho, _ = yule_walker(data_matrix[:, i], order=order)
            # We need [a1, a2, ..., aP], so we take the negative of rho[1:]
            if len(rho) == order + 1:
                 all_coeffs[i, :] = -rho[1:]
            else:
                 # Handle cases where yule_walker might return fewer coefficients
                 # than requested if the data doesn't support the order.
                 # Padding with zeros or raising an error might be options.
                 # For simplicity here, let's print a warning and return None.
                 print(f"    Warning: yule_walker returned unexpected coefficient length for channel {i}. Expected {order}, got {len(rho)-1}")
                 # return None # Option 1: Fail completely
                 # Option 2: Pad with zeros (use with caution)
                 actual_order = len(rho) - 1
                 all_coeffs[i, :actual_order] = -rho[1:]
                 print(f"     Padded remaining coefficients with zeros.")


        except Exception as e:
            print(f"    Warning: AR estimation failed for channel {i}: {e}")
            return None # Return None if estimation fails

    # print("    AR estimation complete.")
    return all_coeffs

def apply_whitening_filter(data_matrix, ar_coeffs):
    """Applies the whitening FIR filter channel by channel."""
    # Check if ar_coeffs is valid before proceeding
    if ar_coeffs is None:
        print("  Warning: apply_whitening_filter received None for ar_coeffs. Returning original data.")
        return data_matrix # Return original data if coeffs are invalid

    n_samples, n_channels = data_matrix.shape
    whitened_data = np.zeros_like(data_matrix)

    # Ensure ar_coeffs has the correct dimensions [n_channels x order]
    if ar_coeffs.shape[0] != n_channels:
         print(f"  Error: Mismatch between number of channels in data ({n_channels}) and ar_coeffs ({ar_coeffs.shape[0]}).")
         return data_matrix # Or raise an error

    order = ar_coeffs.shape[1]

    for i in range(n_channels):
        # Construct the FIR filter coefficients for lfilter: b = [1, a1, a2, ..., aP]
        # where [a1, ..., aP] are the coefficients returned by estimate_ar_coeffs_per_channel.
        # Note: estimate_ar_coeffs_per_channel now returns [a1,...,aP] directly.
        filter_coeffs_b = np.concatenate(([1], ar_coeffs[i, :]))
        # The 'a' coefficients for lfilter are just [1] for an FIR filter
        filter_coeffs_a = [1]

        # Apply the filter
        whitened_data[:, i] = lfilter(filter_coeffs_b, filter_coeffs_a, data_matrix[:, i])

    return whitened_data

# --- Main Processing Script ---

# 1. Load all data first
all_data = {}
patients = ['P1', 'P2', 'P3']
stages = ['pre', 'post']
runs = ['training', 'test']

print("--- Phase 1.4: Loading and Epoching All Data ---")
for patient in patients:
    all_data[patient] = {}
    for stage in stages:
        all_data[patient][stage] = {}
        for run in runs:
            file_pattern = f"{patient}_{stage}_{run}.mat"
            file_path = os.path.join(base_data_dir, file_pattern)
            print(f"Processing {file_pattern}...")
            epochs, labels, fs = process_bci_file(file_path, epoch_start_offset_sec, epoch_end_offset_sec)
            if epochs is not None:
                all_data[patient][stage][run] = {'epochs': epochs, 'labels': labels, 'fs': fs}
                print(f"  Extracted {len(epochs)} epochs.")
            else:
                 all_data[patient][stage][run] = None # Mark as failed
                 print(f"  Skipping {file_pattern} due to processing error.")
print("--- Data Loading Complete ---")


# 2. Apply Whitening
print("\n--- Phase 1.3: Applying Whitening ---")
# Store AR coefficients per patient/stage to apply consistently
ar_coefficients_store = {}

for patient in patients:
    ar_coefficients_store[patient] = {}
    for stage in stages:
        print(f"\nProcessing Whitening for {patient} - {stage}")

        # Check if training data was loaded successfully
        if all_data.get(patient, {}).get(stage, {}).get('training'):
            training_data = all_data[patient][stage]['training']
            training_epochs = training_data['epochs']

            if not training_epochs:
                print(f"  No training epochs found for {patient} {stage}, skipping whitening.")
                ar_coefficients_store[patient][stage] = None
                continue

            # Estimate AR coefficients from concatenated training data
            print(f"  Estimating AR coefficients from {len(training_epochs)} training epochs...")
            concatenated_train_data = np.vstack(training_epochs)
            current_ar_coeffs = estimate_ar_coeffs_per_channel(concatenated_train_data, order=ar_order)
            ar_coefficients_store[patient][stage] = current_ar_coeffs

            if current_ar_coeffs is None:
                print(f"  AR estimation failed for {patient} {stage}, skipping whitening application.")
                continue

            # Apply whitening to training epochs
            print(f"  Applying whitening filter to training epochs...")
            whitened_train_epochs = []
            for epoch in training_epochs:
                whitened_epoch = apply_whitening_filter(epoch, current_ar_coeffs)
                whitened_train_epochs.append(whitened_epoch)
            all_data[patient][stage]['training']['epochs_whitened'] = whitened_train_epochs

            # Apply whitening to testing epochs (using same training AR coeffs)
            if all_data.get(patient, {}).get(stage, {}).get('test'):
                print(f"  Applying whitening filter to testing epochs...")
                testing_epochs = all_data[patient][stage]['test']['epochs']
                whitened_test_epochs = []
                for epoch in testing_epochs:
                    whitened_epoch = apply_whitening_filter(epoch, current_ar_coeffs)
                    whitened_test_epochs.append(whitened_epoch)
                all_data[patient][stage]['test']['epochs_whitened'] = whitened_test_epochs
            else:
                print(f"  Test data not found or failed loading for {patient} {stage}, skipping test whitening.")

        else:
            print(f"  Training data not found or failed loading for {patient} {stage}, skipping whitening.")
            ar_coefficients_store[patient][stage] = None


print("\n--- Whitening Application Complete ---")

# --- Verification ---
print(f"\nVerification:")
if all_data.get('P1', {}).get('pre', {}).get('training', {}).get('epochs_whitened'):
    print(f"P1 pre training whitened epochs count: {len(all_data['P1']['pre']['training']['epochs_whitened'])}")
    print(f"P1 pre training whitened epoch shape: {all_data['P1']['pre']['training']['epochs_whitened'][0].shape}")

if all_data.get('P3', {}).get('post', {}).get('test', {}).get('epochs_whitened'):
    print(f"P3 post test whitened epochs count: {len(all_data['P3']['post']['test']['epochs_whitened'])}")
