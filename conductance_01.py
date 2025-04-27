import utils

defaults = utils.Defaults()

# defaults.l_ksi_default = 2.3
# defaults.l_rho_default = 1.6
# defaults.l_default = 1.9
# defaults.t_default *= 2.

parameters = utils.Parameters(no_dots=2, no_levels=1, default_parameters=defaults)
system = utils.System(parameters)
plot = utils.Plotting()

paramsweep, eigenvalues, occupations, polarizations = system.parameter_sweeping(parameter_name='mu', start=-1.5, stop=1.5, num=101, majorana_repr=False)
#paramsweep, eigenvalues, occupations, polarizations = system.parameter_sweeping(parameter_name='t', start=.1, stop=.5, num=101, majorana_repr=True)
plot.plot_eigenvalues(paramsweep, eigenvalues, polarizations, range=[-1.,1.])


transport = utils.Transport(system, gamma=.1)
C_map = transport.C_ij_map20(1, 1)
plot.plot_conductance_map(C_map, xlabel="$E_F$ (mV)", ylabel="$V_L$ (meV)", suffix="2QD_mul_ef")