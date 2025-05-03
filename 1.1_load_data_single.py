import scipy.io as sio
import numpy as np

# Define the path to one of the .mat files.
file_path = 'stroke-rehab/P1_pre_training.mat'

try:
    # Load the .mat file
    mat_data = sio.loadmat(file_path)

    print(f"Successfully loaded data from: {file_path}")
    print("\nVariables found in the .mat file:")
    # Print the keys found in the dictionary
    for key in mat_data.keys():
        # Exclude standard .mat file metadata keys
        if not key.startswith('__'):
            print(f"- {key}")

    # Access and inspect the main variables: fs, y, trig
    if 'fs' in mat_data:
        fs = mat_data['fs']
        print(f"\nVariable 'fs':")
        print(f"  Type: {type(fs)}")
        print(f"  Shape: {fs.shape}")
        print(f"  Data type: {fs.dtype}")
        # fs is a scalar, but loadmat often returns it as a 1x1 array
        print(f"  Value: {fs.flatten()[0]} Hz") # Flatten to get the scalar value

    if 'y' in mat_data:
        y = mat_data['y']
        print(f"\nVariable 'y' (EEG signal matrix):")
        print(f"  Type: {type(y)}")
        print(f"  Shape: {y.shape} (Nsamples x 16 channels)")
        print(f"  Data type: {y.dtype}")
        print(f"  First 5 samples of first channel: {y[:5, 0]}")

    if 'trig' in mat_data:
        trig = mat_data['trig']
        print(f"\nVariable 'trig' (trigger channel):")
        print(f"  Type: {type(trig)}")
        print(f"  Shape: {trig.shape} (Nsamples x 1)")
        print(f"  Data type: {trig.dtype}")
        # Find and print the first few non-zero trigger values
        non_zero_triggers = trig[trig.flatten() != 0]
        print(f"  First 10 non-zero trigger values: {non_zero_triggers[:10].flatten()}")


except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")