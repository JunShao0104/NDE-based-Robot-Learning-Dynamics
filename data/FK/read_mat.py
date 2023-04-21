import scipy.io as io
import numpy as np

# Two bool variable to control tranform which dataset
Transform_Baxter = True
Transform_Kuka = False

# Baxter data
if Transform_Baxter:
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

    # Find the minimum traj length
    min_traj_len = 3000
    for data in Baxter_data_lst:
        print(data.shape)
        if data.shape[0] < min_traj_len:
            min_traj_len = data.shape[0]
    print("minimum traj length: ", min_traj_len)

    # Transform the trajectory length to the same number
    Baxter_data_lst_shorten = []
    Baxter_data_lst_val = []
    for data_item in Baxter_data_lst:
        data = data_item[:min_traj_len, :]
        data_val = data_item[min_traj_len:, :]
        Baxter_data_lst_shorten.append(data)
        Baxter_data_lst_val.append(data_val)
        # print(data.shape)
        # print(data_val.shape)

    # Concatenation
    Baxter_data_array = np.concatenate(Baxter_data_lst_shorten, axis=0)
    print(Baxter_data_array.shape) # (17410, 35)

    Baxter_data_val = np.concatenate(Baxter_data_lst_val, axis=0)
    print(Baxter_data_val.shape)

    # Save the data as npy form
    # np.save('BaxterDirectDynamics.npy', Baxter_data_array)
    np.save('BaxterDirectDynamics_val.npy', Baxter_data_val)


# Kuka data
if Transform_Kuka:
    Kuka_data = io.loadmat('KukaDirectDynamics.mat')

    # Check the type and keys of the data
    print("type of Kuka_data: ", type(Kuka_data)) # dict
    print("keys of Kuka_data: ", Kuka_data.keys())
    print("header of Kuka_data: ", Kuka_data['__header__']) # b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Fri Feb 26 16:51:50 2016'
    print("version of Kuka_data: ", Kuka_data['__version__']) # 1.0
    print("globals of Kuka_data: ", Kuka_data['__globals__']) # []
    print("kukatraj1 type: ", type(Kuka_data['kukatraj1'])) # numpy array
    print("kukatraj1 shape: ", Kuka_data['kukatraj1'].shape) # (1814, 35)

    # Check the neighbor sample data are related
    # [[state, action, next_state],
    #  [state, action, next_state],
    #  [state, action, next_state]]
    print(Kuka_data['kukatraj1'][0, 21:])
    print(Kuka_data['kukatraj1'][1, :14])

    # Transform the data into numpy array
    Kuka_data_lst = [Kuka_data['kukatraj1'], Kuka_data['kukatraj2'], Kuka_data['kukatraj3'], Kuka_data['kukatraj4'],
                        Kuka_data['kukatraj5'], Kuka_data['kukatraj6'], Kuka_data['kukatraj7'], Kuka_data['kukatraj8'],
                        Kuka_data['kukatraj9'], Kuka_data['kukatraj10']]
    print("length of Kuka_data_lst: ", len(Kuka_data_lst))
    
    # Find the minimum traj length
    min_traj_len = 3000
    for data in Kuka_data_lst:
        print(data.shape)
        if data.shape[0] < min_traj_len:
            min_traj_len = data.shape[0]
    print("minimum traj length: ", min_traj_len)

    # Transform the trajectory length to the same number
    Kuka_data_lst_shorten = []
    for data in Kuka_data_lst:
        data = data[:min_traj_len, :]
        Kuka_data_lst_shorten.append(data)
        print(data.shape)
    
    # Concatenation
    Kuka_data_array = np.concatenate(Kuka_data_lst_shorten, axis=0)
    print(Kuka_data_array.shape) # (18140, 35)

    # Save the data as npy form
    np.save('KukaDirectDynamics.npy', Kuka_data_array)


