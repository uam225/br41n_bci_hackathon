import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import traceback

# --- Configuration ---
input_data_file = 'all_processed_data.pkl'
data_key_to_use = 'epochs' # Use original data for visualization clarity
patients_to_analyze = ['P2', 'P3']
stages_to_analyze = ['pre', 'post']
# Channel indices to analyze (e.g., potentially C3, Cz, C4)
channels_to_analyze_idx = [6, 8, 10]
channel_names_generic = [f'EEG {i+1:03d}' for i in range(16)] # Generic names

# TFR parameters
freqs = np.arange(8, 36, 1)  # Frequencies from 8Hz to 35Hz
n_cycles = freqs / 2.0  # Number of cycles in Morlet wavelet, often freq-dependent

# Baseline period (relative to epoch start in seconds)
# Our epoch starts 1s after trigger. Let's use the first 0.5s of the epoch as baseline
# (i.e., 1.0s to 1.5s post-trigger)
baseline_period = (0, 0.5)

# Time window for plotting (relative to epoch start in seconds)
plot_tmin = 0
plot_tmax = 7.0 # Corresponds to 1s to 8s post-trigger

# --- Load Processed Data ---
print(f"Loading processed data from {input_data_file}...")
try:
    with open(input_data_file, 'rb') as f:
        all_data = pickle.load(f)
    print("Data loaded successfully.")
except FileNotFoundError: print(f"Error: Data file '{input_data_file}' not found."); exit()
except Exception as e: print(f"Error loading data: {e}"); exit()

# --- Analysis Loop ---
# Store TFR results to plot comparisons later
tfr_results = {} # tfr_results[patient][stage][label] = tfr_average

# Get sampling rate (assuming consistent)
try: fs = all_data['P1']['pre']['training']['fs']
except KeyError: print("Error: Cannot get sampling rate."); exit()
print(f"Using sampling rate: {fs} Hz")

# Create a basic MNE Info object
ch_types = ['eeg'] * 16
info = mne.create_info(ch_names=channel_names_generic, sfreq=fs, ch_types=ch_types)
#info.set_montage('standard_1020') # Use a standard montage for topo plots if needed later

for patient in patients_to_analyze:
    tfr_results[patient] = {}
    for stage in stages_to_analyze:
        tfr_results[patient][stage] = {}
        print(f"\nProcessing TFR for {patient} - {stage}...")

        # Combine training and testing epochs for more robust TFR
        try:
            train_epochs_list = all_data[patient][stage]['training'][data_key_to_use]
            train_labels = all_data[patient][stage]['training']['labels']
            test_epochs_list = all_data[patient][stage]['test'][data_key_to_use]
            test_labels = all_data[patient][stage]['test']['labels']

            all_epochs_list = train_epochs_list + test_epochs_list
            all_labels = train_labels + test_labels

            if not all_epochs_list:
                print(f"  No epochs found for {patient} {stage}. Skipping.")
                continue

            # Data needs to be in Volts for MNE (assuming original data might be microvolts or arbitrary units)
            # Let's scale it by 1e-6 assuming microvolts, adjust if units are different
            # If units are arbitrary, scaling might not matter much for relative plots, but good practice.
            all_epochs_array = np.array(all_epochs_list) * 1e-6 # Shape [n_epochs, n_samples, n_channels]

            # Create MNE EpochsArray object
            # MNE expects [n_epochs, n_channels, n_samples]
            mne_epochs = mne.EpochsArray(all_epochs_array.transpose(0, 2, 1), info, verbose=False)

            # Define event IDs for MNE
            event_id = {'Left': 1, 'Right': -1} # Map labels to names

            # Calculate TFR for Left and Right trials separately
            for label_name, label_val in event_id.items():
                print(f"  Calculating TFR for {label_name} trials...")
                # Select epochs for the current label
                current_labels_mask = [l == label_val for l in all_labels]
                epochs_for_label = mne_epochs[current_labels_mask]

                if len(epochs_for_label) == 0:
                    print(f"    No trials found for label {label_name}. Skipping.")
                    continue

                # Calculate TFR
                tfr = mne.time_frequency.tfr_morlet(
                    epochs_for_label,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    return_itc=False, # We only need power
                    average=True, # Average across trials
                    verbose=False
                )

                # Apply baseline correction (percent change)
                print(f"    Applying baseline correction {baseline_period}s...")
                tfr.apply_baseline(baseline=baseline_period, mode='percent')

                # Store the averaged, baseline-corrected TFR
                tfr_results[patient][stage][label_name] = tfr

        except Exception as e:
            print(f"  Error processing {patient} {stage}: {e}")
            traceback.print_exc()


print("\n--- TFR Calculation Complete ---")

# --- Plotting ---
print("\n--- Generating Plots ---")

for patient in patients_to_analyze:
    for label_name in ['Left', 'Right']:
        for ch_idx in channels_to_analyze_idx:
            ch_name = info['ch_names'][ch_idx]
            print(f"  Plotting comparison for {patient} - {label_name} - {ch_name}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            fig.suptitle(f'{patient} - {label_name} MI - Channel {ch_name} (Beta Focus)')

            # Plot Pre-therapy TFR
            tfr_pre = tfr_results.get(patient, {}).get('pre', {}).get(label_name)
            if tfr_pre:
                tfr_pre.plot(picks=[ch_idx], baseline=None, mode='percent',
                             tmin=plot_tmin, tmax=plot_tmax, fmin=13, fmax=30,
                             axes=axes[0], colorbar=False, show=False, verbose=False)
                # Use .nave to get the number of averaged trials
                axes[0].set_title(f'Pre-Therapy ({tfr_pre.nave} trials)') # MODIFIED HERE
                axes[0].axvline(x=1.0, color='white', linestyle='--', linewidth=1)
                axes[0].set_xlabel('Time (s) relative to Epoch Start')
            else:
                axes[0].set_title('Pre-Therapy (No Data)')
                axes[0].set_xlabel('Time (s) relative to Epoch Start')

            # Plot Post-therapy TFR
            tfr_post = tfr_results.get(patient, {}).get('post', {}).get(label_name)
            if tfr_post:
                im = tfr_post.plot(picks=[ch_idx], baseline=None, mode='percent',
                                   tmin=plot_tmin, tmax=plot_tmax, fmin=13, fmax=30,
                                   axes=axes[1], colorbar=True, show=False, verbose=False)
                # Use .nave to get the number of averaged trials
                axes[1].set_title(f'Post-Therapy ({tfr_post.nave} trials)') # MODIFIED HERE
                axes[1].axvline(x=1.0, color='white', linestyle='--', linewidth=1)
                axes[1].set_xlabel('Time (s) relative to Epoch Start')
            else:
                axes[1].set_title('Post-Therapy (No Data)')
                axes[1].set_xlabel('Time (s) relative to Epoch Start')

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            plot_filename = f'ERD_Compare_{patient}_{label_name}_Ch{ch_idx+1}.png'
            plt.savefig(plot_filename)
            print(f"    Saved plot: {plot_filename}")
            plt.close(fig)

print("\n--- Plotting Complete ---")