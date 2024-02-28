import utils

defaults = utils.Defaults()
parameters = utils.Parameters(no_dots=7, no_levels=2, default_parameters=defaults)
system = utils.System(parameters)
plot = utils.Plotting()

hamiltonian = system.full_hamiltonian()
plot.plot_hamiltonian(hamiltonian)

# eigs = eigh(hamiltonian, eigvals_only=True)

paramsweep, eigenvalues, occupations = system.parameter_sweeping(parameter_name='mu', start=-1.5, stop=.5, num=101)
plot.plot_eigenvalues(paramsweep, eigenvalues, occupations, range=[-1.,1.])


