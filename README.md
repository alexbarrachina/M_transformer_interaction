## Transformer version of the piano-genie 

Separate Embeddings: Each feature (dtime, vel, pitch, dur, button) has its own embedding layer.
Summation: The embeddings for each feature are summed to form a single embedding tensor per timestep.

