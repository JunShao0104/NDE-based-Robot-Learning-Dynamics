import scipy.io as io
import numpy as np

Baxter_data = io.loadmat('BaxterDirectDynamics.mat')

# Check the type and keys of the data
print("type of Baxter_data: ", type(Baxter_data)) # dict
print("keys of Baxter_data: ", Baxter_data.keys())
print("header of Baxter_data: ", Baxter_data['__header__']) # b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Fri Feb 26 16:51:50 2016'
print("version of Baxter_data: ", Baxter_data['__version__']) # 1.0
print("globals of Baxter_data: ", Baxter_data['__globals__']) # []
print("baxtertraj1 type: ", type(Baxter_data['baxtertraj1'])) # numpy array
print("baxtertraj1 shape: ", Baxter_data['baxtertraj1'].shape) # (2001, 35)
# Check the neighbor sample data are related
# [[state, action, next_state],
#  [state, action, next_state],
#  [state, action, next_state]]
print(Baxter_data['baxtertraj1'][0, 21:])
print(Baxter_data['baxtertraj1'][1, :14])

# Transform the data into numpy array
Baxter_data_lst = [Baxter_data['baxtertraj1'], Baxter_data['baxtertraj2'], Baxter_data['baxtertraj3'], Baxter_data['baxtertraj4'],
                    Baxter_data['baxtertraj5'], Baxter_data['baxtertraj6'], Baxter_data['baxtertraj7'], Baxter_data['baxtertraj8'],
                    Baxter_data['baxtertraj9'], Baxter_data['baxtertraj10']]
print("length of Baxter_data_lst: ", len(Baxter_data_lst))
for data in Baxter_data_lst:
    print(data.shape)

# Concatenation
Baxter_data_array = np.concatenate(Baxter_data_lst, axis=0)
print(Baxter_data_array.shape) # (19295, 35)

# Save the data as npy form
np.save('BaxterDirectDynamics.npy', Baxter_data_array)


