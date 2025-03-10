import os
import matplotlib.pyplot as plt
from TMIDIX import Tegridy_Any_Pickle_File_Writer, Tegridy_Any_Pickle_File_Reader

data = Tegridy_Any_Pickle_File_Reader(input_file_name='./save_models/losses_accs')

train_losses = data[0]
train_accs = data[1]
val_losses = data[2]
val_accs = data[3]


# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()