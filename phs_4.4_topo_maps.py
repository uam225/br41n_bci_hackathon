import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import traceback

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
data_key_to_use = 'epochs' # Use original data
patients_to_analyze = ['P2', 'P3']
stages_to_analyze = ['pre', 'post']

# Channel mapping based on the provided image (Number -> Label)
channel_names = [
    'FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Pz'
] # Assuming Number 1 = index 0, etc.

# Frequency band and time window for analysis
freq_band = [13, 30] # Beta band
analysis_window_start_sec = 2.0 # Start 2s into epoch (3s post-trigger)
analysis_window_end_sec = 6.0   # End 6s into epoch (7s post-trigger)
filter_order = 5

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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1 or low >= high: return data
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

# --- Analysis Loop ---
power_results = {} # Store average power per condition

try: fs = all_data['P1']['pre']['training']['fs']
except KeyError: print("Error: Cannot determine sampling rate."); exit()
print(f"Using sampling rate: {fs} Hz")

# Create MNE Info object with CORRECT channel names
ch_types = ['eeg'] * 16
info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types=ch_types)
try:
    info.set_montage('standard_1020', on_missing='warn') # Use standard 10-20 montage
except ValueError as e:
    print(f"Error setting montage (some channels might not be standard 10-20): {e}")
    print("Proceeding without precise montage locations, topomap might be approximate.")
except Exception as e:
     print(f"An unexpected error occurred setting montage: {e}")


# Calculate sample indices for analysis window
start_sample = int(analysis_window_start_sec * fs)
end_sample = int(analysis_window_end_sec * fs)
print(f"Analyzing window: {start_sample} to {end_sample} samples ({analysis_window_start_sec}s to {analysis_window_end_sec}s within epoch)")

for patient in patients_to_analyze:
    power_results[patient] = {}
    for stage in stages_to_analyze:
        print(f"\nCalculating Beta power for {patient} - {stage}...")
        try:
            # Combine train and test epochs
            train_epochs_list = all_data[patient][stage]['training'][data_key_to_use]
            test_epochs_list = all_data[patient][stage]['test'][data_key_to_use]
            all_epochs_list = train_epochs_list + test_epochs_list

            if not all_epochs_list: print(f"  No epochs found. Skipping."); continue

            # Stack epochs: [n_epochs, n_samples_epoch, n_channels]
            all_epochs_array = np.array(all_epochs_list)
            # MNE expects [n_epochs, n_channels, n_samples_epoch]
            all_epochs_mne = all_epochs_array.transpose(0, 2, 1)

            # Filter data to Beta band
            print(f"  Applying bandpass filter: {freq_band} Hz")
            low, high = freq_band
            filtered_epochs = bandpass_filter(all_epochs_mne, low, high, fs, order=filter_order)

            # Select time window
            epochs_windowed = filtered_epochs[:, :, start_sample:end_sample]

            # Calculate average power (variance) per channel across trials and time
            # Variance is proportional to power for zero-mean signals (like filtered EEG)
            avg_power_per_channel = np.mean(np.var(epochs_windowed, axis=2), axis=0) # Avg over epochs
            power_results[patient][stage] = avg_power_per_channel
            print(f"  Calculated average power for {len(avg_power_per_channel)} channels.")

        except Exception as e:
            print(f"  Error processing {patient} {stage}: {e}")
            traceback.print_exc()


print("\n--- Power Calculation Complete ---")

# --- Plotting Topomaps of Power Difference (Post - Pre) ---
print("\n--- Generating Topomaps ---")

for patient in patients_to_analyze:
    if 'pre' in power_results[patient] and 'post' in power_results[patient]:
        power_pre = power_results[patient]['pre']
        power_post = power_results[patient]['post']

        # Calculate difference (or log ratio for relative change)
        power_diff = power_post - power_pre
        # Optional: Log ratio power_diff = np.log10(power_post / power_pre)

        print(f"\nPlotting Beta Power Difference (Post-Pre) for {patient}")

        fig, ax = plt.subplots(figsize=(6, 5))
        im, cn = mne.viz.plot_topomap(
            data=power_diff,
            pos=info,
            axes=ax,
            cmap='RdBu_r', # Red=Increase Post, Blue=Decrease Post
            show=False,
            # You might need to adjust vmin/vmax based on the data range
            # vmin=-np.max(np.abs(power_diff)), vmax=np.max(np.abs(power_diff))
        )

        ax.set_title(f'{patient}: Beta Power Change (Post - Pre)\nduring Motor Imagery')
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Avg. Power Difference (Post - Pre)')

        plt.tight_layout()
        topo_filename = f'topomap_beta_diff_{patient}.png'
        plt.savefig(topo_filename)
        print(f"  Saved plot: {topo_filename}")
        plt.close(fig)

    else:
        print(f"\nSkipping topomap for {patient}: Missing pre or post power data.")

print("\n--- Topomap Plotting Complete ---")