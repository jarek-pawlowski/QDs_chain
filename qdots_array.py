import utils

defaults = utils.Defaults()

# defaults.l_ksi_default = 2.3
# defaults.l_rho_default = 1.6
# defaults.l_default = 1.9

parameters = utils.Parameters(no_dots=3, no_levels=1, default_parameters=defaults)
system = utils.System(parameters)
plot = utils.Plotting()

hamiltonian = system.full_hamiltonian()
plot.plot_hamiltonian(hamiltonian)

# eigs = eigh(hamiltonian, eigvals_only=True)

paramsweep, eigenvalues, occupations, polarizations = system.parameter_sweeping(parameter_name='mu', start=-1.5, stop=1.5, num=101, majorana_repr=False)
#paramsweep, eigenvalues, occupations, polarizations = system.parameter_sweeping(parameter_name='t', start=.1, stop=.5, num=101, majorana_repr=True)
plot.plot_eigenvalues(paramsweep, eigenvalues, polarizations, range=[-1.,1.])

#majorana = system.majorana_representation(hamiltonian)