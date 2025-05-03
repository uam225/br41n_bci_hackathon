import scipy.io as sio
import numpy as np
import os 


# --- Configuration
data_dir = 'stroke-rehab/'
file_name = 'P1_pre_training.mat'
file_path = os.path.join(data_dir, file_name)

try:
    mat_data = sio.loadmat(file_path)
    fs = mat_data['fs'].flatten()[0] # Get sampling rate as a scalar
    y = mat_data['y']
    trig = mat_data['trig'].flatten() # Flatten trig for easier processing

    print(f"Loaded data from {file_name} with fs={fs} Hz, y shape={y.shape}, trig shape={trig.shape}")

    # --- Event Extraction & Epoching ---

    # Define epoch window relative to trigger onset (in seconds)
    # Epoch starts 1.0s after trigger (t=0 in Figure 3)
    epoch_start_offset_sec = 1.0
    # Epoch ends 8.0s after trigger (t=0 in Figure 3)
    epoch_end_offset_sec = 8.0

    # Convert times to samples
    epoch_start_offset_samples = int(epoch_start_offset_sec * fs)
    epoch_end_offset_samples = int(epoch_end_offset_sec * fs)
    epoch_duration_samples = epoch_end_offset_samples - epoch_start_offset_samples

    print(f"\nEpoch window: {epoch_start_offset_sec}s to {epoch_end_offset_sec}s relative to trigger onset")
    print(f"Corresponding samples: {epoch_start_offset_samples} to {epoch_end_offset_samples} (duration: {epoch_duration_samples} samples)")


    # Find the sample indices where the trigger value changes from 0 to non-zero
    # This indicates the onset of a new trial/event
    # Look for points where trig[i] is non-zero AND trig[i-1] was zero
    trigger_onsets_idx = np.where((trig != 0) & (np.roll(trig, 1) == 0))[0]

    print(f"\nFound {len(trigger_onsets_idx)} potential trigger onsets.")

    epochs = []
    labels = []
    sampling_rate = fs # Use the loaded fs

    # Iterate through trigger onsets to extract epochs
    for onset_sample_idx in trigger_onsets_idx:
        # Get the label for this trial (should be +1 or -1)
        label = trig[onset_sample_idx]

        # We only care about +1 (Left MI) and -1 (Right MI) trials for classification
        if label in [1, -1]:
            # Calculate the start and end sample index for the epoch
            epoch_start_sample = onset_sample_idx + epoch_start_offset_samples
            epoch_end_sample = onset_sample_idx + epoch_end_offset_samples # End is exclusive in slicing

            # Ensure the epoch is within the bounds of the recording
            if epoch_start_sample >= 0 and epoch_end_sample <= y.shape[0]:
                # Extract the epoch data
                epoch_data = y[epoch_start_sample:epoch_end_sample, :] # Shape [timepoints x channels]

                epochs.append(epoch_data)
                labels.append(label)
            else:
                # print(f"Warning: Epoch starting at sample {onset_sample_idx} is out of bounds.")
                pass # Skip epochs that don't fit entirely

    print(f"\nExtracted {len(epochs)} epochs for labels +1 and -1.")
    if len(epochs) > 0:
        print(f"Shape of one epoch: {epochs[0].shape} (samples x channels)")
        print(f"Corresponding labels: {labels[:10]}...") # Print first 10 labels

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")