import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Provided functions
def compute_trajectory_smoothness(trajectory):
    differences = np.diff(trajectory, axis=0)  # Differences between consecutive actions
    magnitudes = np.linalg.norm(differences, axis=1)  # Norm of differences
    smoothness = np.mean(magnitudes)  # Average magnitude
    return smoothness

def save_histogram(smoothness_values, output_path, bins=20):
    """
    Saves a histogram of smoothness values as a percentage of total.

    Args:
        smoothness_values (list or np.ndarray): List of smoothness values.
        output_path (str): Path to save the histogram image (without file extension).
        bins (int): Number of bins for the histogram.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Calculate the weights for percentage representation
    weights = (1 / len(smoothness_values)) * 100 * np.ones(len(smoothness_values))

    plt.figure(figsize=(8, 6))
    plt.hist(smoothness_values, bins=bins, color='blue', edgecolor='black', alpha=0.7, weights=weights)
    plt.title("Smoothness Distribution")
    plt.xlabel("Smoothness")
    plt.ylabel("Percentage (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the histogram
    print("Histogram saved to:", output_path)
    plt.savefig(output_path + '.png', format='png')
    plt.close()


def save_average_smoothness(average_smoothness, output_path):
    """
    Saves the average smoothness value to a text file.

    Args:
        average_smoothness (float): The computed average smoothness value.
        output_path (str): Path to save the average smoothness file.
    """
    with open(output_path, 'w') as f:
        f.write(f"Average Smoothness: {average_smoothness:.6f}\n")
    print("Average smoothness saved to:", output_path)

# Smoothness calculation and histogram plotting
def analyze_smoothness(hdf5_file_path, output_dir):
    smoothness_values = []

    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as hdf_file:
        # Iterate over all demos in the "data" group
        for demo in hdf_file["data"].keys():
            actions = hdf_file[f"data/{demo}/actions"][:]
            
            # Compute the smoothness for the current demo's trajectory
            smoothness = compute_trajectory_smoothness(actions)
            smoothness_values.append(smoothness)

            print(f"Demo {demo}: Smoothness = {smoothness}")

    # Compute average smoothness
    average_smoothness = np.mean(smoothness_values)
    print("Average Smoothness:", average_smoothness)

    # Save the histogram of smoothness values
    histogram_path = os.path.join(output_dir, "smoothness_histogram")
    save_histogram(smoothness_values, histogram_path)

    # Save the average smoothness to a text file
    average_smoothness_path = os.path.join(output_dir, "average_smoothness.txt")
    save_average_smoothness(average_smoothness, average_smoothness_path)

# Example usage
if __name__ == "__main__":
    # Path to the HDF5 file
    hdf5_file_path = "/iris/u/rheamal/bid2/data/diverse/test_randomstart_medium/demos.hdf5"

    # Output directory to save the results
    output_dir = "/iris/u/rheamal/bid2/data/diverse/test_randomstart_medium/results"

    # Analyze smoothness and save the results
    analyze_smoothness(hdf5_file_path, output_dir)
