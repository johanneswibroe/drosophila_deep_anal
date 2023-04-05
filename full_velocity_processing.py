import os
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        if np.isnan(y).all():
            continue

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        mask = np.isnan(x)
        x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel


# Set up input and output paths
input_folder = "/home/joeh/real_experiment_I_guess/anal"
output_file = "raw.csv"

# Initialize output array
output_data = []

# Loop through files in input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".h5"):
        # Load data from file
        file_path = os.path.join(input_folder, file_name)
        with h5py.File(file_path, "r") as f:
            locations = f["tracks"][:].T

        # Fill in missing values
        locations = fill_missing(locations)

        # Extract thorax location and compute velocity
        thorax_loc = locations[:, 0, :, :]
        thorax_vel = smooth_diff(thorax_loc[:, :, 0])

        # Get frame numbers
        frame_numbers = np.arange(1, len(thorax_vel)+1)

        # Get filename for each row
        filenames = np.full(len(thorax_vel), file_name)

        # Get condition for each row
        if "cont" in file_name:
            condition = np.full(len(thorax_vel), "control")
        elif "stim" in file_name:
            condition = np.full(len(thorax_vel), "stimulus")
        else:
            condition = np.full(len(thorax_vel), "unknown")

        # Combine the arrays
        output_data_temp = np.column_stack((filenames, frame_numbers, thorax_vel, condition))

        # Append to output array
        output_data.append(output_data_temp)

# Save output to CSV
np.savetxt(output_file, np.concatenate(output_data), delimiter=",", fmt="%s")

import pandas as pd

# read in the csv file
df = pd.read_csv('raw.csv')

# add column titles to first row
df.columns = ['Filenames', 'Frame', 'Velocity', 'Condition']

# add "Onset" column with initial value of ""
df["Onset"] = ""

# loop through rows of "Frame" column and set value of "Onset" column accordingly
for i in range(len(df)):
    if df.loc[i, "Frame"] < 250:
        df.loc[i, "Onset"] = "Pre_onset"
    else:
        df.loc[i, "Onset"] = "Post_onset"

# save the updated dataframe to a new csv file
df.to_csv('filtered.csv', index=False)

import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("filtered.csv")

# Split the DataFrame into two based on the Onset value
df_pre_onset = df[df["Onset"] == "Pre_onset"]
df_post_onset = df[df["Onset"] == "Post_onset"]

# Save the two DataFrames to separate CSV files
df_pre_onset.to_csv("your_pre_onset_file.csv", index=False)
df_post_onset.to_csv("your_post_onset_file.csv", index=False)


import pandas as pd

# Load the two CSV files into pandas DataFrames
df_pre_onset = pd.read_csv("your_pre_onset_file.csv")
df_post_onset = pd.read_csv("your_post_onset_file.csv")

# Rename the "Velocity" column of each DataFrame
df_pre_onset = df_pre_onset.rename(columns={"Velocity": "Pre_onset_velocity"})
df_post_onset = df_post_onset.rename(columns={"Velocity": "Post_onset_velocity"})

# Concatenate the two DataFrames side by side
df_merged = pd.concat([df_pre_onset, df_post_onset["Post_onset_velocity"]], axis=1)

# Save the merged DataFrame to a CSV file
df_merged.to_csv("your_merged_file.csv", index=False)

####################



# read in the CSV file
df = pd.read_csv('your_merged_file.csv')

# calculate the average pre-onset velocity for each filename and condition
pre_avg = df.groupby(['Filenames', 'Condition'])['Pre_onset_velocity'].mean().reset_index()
pre_avg = pre_avg.rename(columns={'Pre_onset_velocity': 'Average_Pre_onset_velocity'})

# calculate the average post-onset velocity for each filename and condition
post_avg = df.groupby(['Filenames', 'Condition'])['Post_onset_velocity'].mean().reset_index()
post_avg = post_avg.rename(columns={'Post_onset_velocity': 'Average_Post_onset_velocity'})

# merge the two DataFrames
merged = pd.merge(pre_avg, post_avg, on=['Filenames', 'Condition'])

# output the results to a new CSV file
merged.to_csv('merged_with_averages_file.csv', index=False)