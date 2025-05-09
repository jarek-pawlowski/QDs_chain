from copy import deepcopy

import utils

defaults = utils.Defaults()

# defaults.l_ksi_default = 2.3
# defaults.l_rho_default = 1.6
# defaults.l_default = 1.9
# defaults.t_default *= 2.

parameters = utils.Parameters(no_dots=3, no_levels=1, default_parameters=defaults)
parameters.set_random_parameters_free_reduced()
reference = deepcopy(parameters)
system = utils.System(parameters)
plot = utils.Plotting()

paramsweep, eigenvalues, occupations, polarizations = system.parameter_sweeping(parameter_name='mu', start=-1.5, stop=1.5, num=101, majorana_repr=False)
#paramsweep, eigenvalues, occupations, polarizations = system.parameter_sweeping(parameter_name='t', start=.1, stop=.5, num=101, majorana_repr=True)
plot.plot_eigenvalues(paramsweep, eigenvalues, polarizations, range=[-1.,1.])


transport = utils.Transport(system, gamma=.1, density=201, reference_point=reference)
C_map = transport.C_ij_map21(0, 0, 2)
plot.plot_conductance_map(C_map, xlabel="$\Delta{}V_L$ (mV)", ylabel="$E_F$ (meV)", suffix="2QD_det_ef")

transport = utils.Transport(system, gamma=.1)
C_map = transport.C_ij_map22(0, 0)
plot.plot_conductance_map(C_map, xlabel="$\Delta{}B_z$ (mV)", ylabel="$E_F$ (meV)", suffix="2QD_bz_ef")