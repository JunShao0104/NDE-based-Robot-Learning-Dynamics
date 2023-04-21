import scipy.io as io
import numpy as np

Transform_Baxter = True

# Baxter data
if Transform_Baxter:
    Baxter_data = io.loadmat('BaxterRand.mat')

    # Check the type and keys of the data
    print("type of Baxter_data: ", type(Baxter_data)) # dict
    print("keys of Baxter_data: ", Baxter_data.keys()) # '__header__', '__version__', '__globals__', 'baxter_rand', 'X_train', 'Y_train', 'X_test', 'Y_test'
    print("header of Baxter_data: ", Baxter_data['__header__']) # b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Jan 27 17:10:50 2015'
    print("version of Baxter_data: ", Baxter_data['__version__']) # 1.0
    print("globals of Baxter_data: ", Baxter_data['__globals__']) # []
    print("baxter_rand type: ", type(Baxter_data['baxter_rand'])) # numpy array
    print("baxter_rand shape: ", Baxter_data['baxter_rand'].shape) # (19992, 28)
    # Train
    print("X_train type: ", type(Baxter_data['X_train'])) # numpy array
    print("X_train shape: ", Baxter_data['X_train'].shape) # (15000, 21)
    print("Y_train type: ", type(Baxter_data['Y_train'])) # numpy array
    print("Y_train shape: ", Baxter_data['Y_train'].shape) # (15000, 7)
    # Test
    print("X_test type: ", type(Baxter_data['X_test'])) # numpy array
    print("X_test shape: ", Baxter_data['X_test'].shape) # (4992, 21)
    print("Y_test type: ", type(Baxter_data['Y_test'])) # numpy array
    print("Y_test shape: ", Baxter_data['Y_test'].shape) # (4992, 7)

    # Concatenation
    Baxter_data_array = np.concatenate((Baxter_data['X_train'], Baxter_data['Y_train']), axis=1)
    print(Baxter_data_array.shape) # (15000, 28) = (15000, 21) + (15000, 7)

    # # Save the data as npy form
    np.save('BaxterRand.npy', Baxter_data_array)



