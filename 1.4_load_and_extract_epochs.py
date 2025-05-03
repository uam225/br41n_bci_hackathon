import scipy.io as sio
import numpy as np
import os
import glob 

# --- Configuration ---
base_data_dir = 'stroke-rehab/' 
epoch_start_offset_sec = 1.0
epoch_end_offset_sec = 8.0

# --- Function to process a single .mat file ---
def process_bci_file(file_path, epoch_start_offset_sec, epoch_end_offset_sec):
    """
    Loads BCI data from a .mat file, extracts epochs based on triggers,
    and returns epochs and labels.

    Args:
        file_path (str): Full path to the .mat file.
        epoch_start_offset_sec (float): Start of epoch relative to trigger onset (s).
        epoch_end_offset_sec (float): End of epoch relative to trigger onset (s).

    Returns:
        tuple: (epochs, labels, fs) where:
            epochs (list): List of numpy arrays, each [n_samples x n_channels].
            labels (list): List of trial labels (+1 or -1).
            fs (float): Sampling rate.
        Returns (None, None, None) if file loading or processing fails.
    """
    try:
        mat_data = sio.loadmat(file_path)
        fs = mat_data['fs'].flatten()[0]
        y = mat_data['y']
        trig = mat_data['trig'].flatten()

        # Convert times to samples
        epoch_start_offset_samples = int(epoch_start_offset_sec * fs)
        epoch_end_offset_samples = int(epoch_end_offset_sec * fs)

        # Find the sample indices where the trigger value changes from 0 to non-zero
        trigger_onsets_idx = np.where((trig != 0) & (np.roll(trig, 1) == 0))[0]

        epochs = []
        labels = []

        for onset_sample_idx in trigger_onsets_idx:
            label = trig[onset_sample_idx]

            if label in [1, -1]: # Only process Left MI (+1) and Right MI (-1)
                epoch_start_sample = onset_sample_idx + epoch_start_offset_samples
                epoch_end_sample = onset_sample_idx + epoch_end_offset_samples

                if epoch_start_sample >= 0 and epoch_end_sample <= y.shape[0]:
                    epoch_data = y[epoch_start_sample:epoch_end_sample, :]
                    epochs.append(epoch_data)
                    labels.append(label)

        return epochs, labels, fs

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

# --- Main processing loop ---

# Store the data in a nested dictionary:
# all_data[patient_id][stage][run] = {'epochs': list, 'labels': list, 'fs': float}
all_data = {}

# Patients, stages, and runs we need to process
patients = ['P1', 'P2', 'P3']
stages = ['pre', 'post']
runs = ['training', 'test']

print("Starting data loading and epoching for all patients, stages, and runs...")

for patient in patients:
    all_data[patient] = {} # Create entry for patient
    for stage in stages:
        all_data[patient][stage] = {} # Create entry for stage
        for run in runs:
            # Construct the expected file name pattern
            file_pattern = f"{patient}_{stage}_{run}.mat"
            file_path = os.path.join(base_data_dir, file_pattern)

            print(f"\nProcessing {file_pattern}...")

            epochs, labels, fs = process_bci_file(file_path, epoch_start_offset_sec, epoch_end_offset_sec)

            if epochs is not None: # Check if processing was successful
                all_data[patient][stage][run] = {
                    'epochs': epochs,
                    'labels': labels,
                    'fs': fs
                }
                print(f"  Extracted {len(epochs)} epochs.")
            else:
                 print(f"  Skipping {file_pattern} due to processing error.")


print("\nData loading and epoching complete.")

# --- Verification ---

# Example check:
print(f"\nVerification:")
if 'P1' in all_data and 'pre' in all_data['P1'] and 'training' in all_data['P1']['pre']:
    print(f"P1 pre training epochs count: {len(all_data['P1']['pre']['training']['epochs'])}")
    print(f"P1 pre training labels count: {len(all_data['P1']['pre']['training']['labels'])}")
    if len(all_data['P1']['pre']['training']['epochs']) > 0:
        print(f"P1 pre training epoch shape: {all_data['P1']['pre']['training']['epochs'][0].shape}")

# Add checks for other files
if 'P3' in all_data and 'post' in all_data['P3'] and 'test' in all_data['P3']['post']:
     if all_data['P3']['post']['test']: # Check if dictionary is not empty
        print(f"P3 post test epochs count: {len(all_data['P3']['post']['test']['epochs'])}")