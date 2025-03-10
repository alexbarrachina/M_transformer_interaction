from datasets import load_dataset
import os
# Download the dataset with a specific cache directory
cache_dir = "/Volumes/DADES/Deep/Symbolic/monster_genie/dataset_cache"
os.makedirs(cache_dir, exist_ok=True)

monster_piano = load_dataset('asigalov61/Monster-Piano', cache_dir=cache_dir)

# Print the cache location for future reference
print(f"Dataset cached at: {cache_dir}")

# Save it to disk for future use
monster_piano.save_to_disk("/Volumes/DADES/Datasets/MIDI/asigalov61___monster-piano")