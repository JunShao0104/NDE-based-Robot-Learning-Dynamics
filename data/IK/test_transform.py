import numpy as np

# Baxter
Baxter_data = np.load('BaxterRand.npy', allow_pickle=True)
print(Baxter_data.shape) # (15000, 28)