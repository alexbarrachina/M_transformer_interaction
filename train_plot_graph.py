#===================================================================================================
# Monster Genie train_plot_graph.py Python module
# Plot graph of training and validation losses
# 
# Alex Barrachina 2025
#===================================================================================================
# License: Apache 2.0
#===================================================================================================

import matplotlib.pyplot as plt
import numpy as np
from midiUtils import Any_Pickle_File_Reader

# Define a function for moving average
def moving_average(data, window_size):
    """Calculate moving average with specified window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# Choose window size (adjust as needed)
window_size = 100  # Averages over 10 epochs

data = Any_Pickle_File_Reader(input_file_name='./save_models/big_mar28_decoder_only.pickle')

train_losses = np.array(data[0])
train_accs = np.array(data[1])
val_losses = np.array(data[2])
val_accs = np.array(data[3])

data_to_plot = val_accs

# Create x-axis that aligns with the averaged data
avg_data = moving_average(data_to_plot, window_size)
epochs = range(window_size-1, len(data_to_plot))

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(epochs, avg_data, label='val accuracy')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()