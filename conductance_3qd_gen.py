from copy import deepcopy
import os
import numpy as np

import utils

defaults = utils.Defaults()

dirpath = './3qd_train'

for s in range(400):
    parameters = utils.Parameters(no_dots=3, no_levels=1, default_parameters=defaults)
    parameters.set_random_parameters_free_reduced()
    reference = deepcopy(parameters)
    system = utils.System(parameters)
    transport = utils.Transport(system, gamma=.1, density=101, reference_point=reference)

    C_map = []
    for k in [0,1,2]:
        for i in [0,1]:
            for j in [0,1]:
                C_map.append(transport.C_ij_map21(i,j,k)[:,2].reshape((101,101)))
    for i in [0,1]:
        for j in [0,1]:
            C_map.append(transport.C_ij_map22(i,j)[:,2].reshape((101,101)))

    C_map = np.stack(C_map)
    np.save(os.path.join(dirpath, 'sample_'+str(s)+'.npy'), C_map)
    
    # mu_L, mu_C, mu_R, t_LC, t_CR, l_LC, l_CR
    par = parameters.get_parameters_reduced()
    np.savetxt(os.path.join(dirpath, 'label_'+str(s)+'.txt'), par)