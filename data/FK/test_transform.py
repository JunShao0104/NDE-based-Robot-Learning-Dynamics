import numpy as np

# Baxter
Baxter_data = np.load('BaxterDirectDynamics.npy', allow_pickle=True)
print(Baxter_data.shape) # (17410, 35)

# Kuka
Kuka_data = np.load('KukaDirectDynamics.npy', allow_pickle=True)
print(Kuka_data.shape) # (18140, 35)