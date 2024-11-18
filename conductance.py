import utils

defaults = utils.Defaults()

# defaults.l_ksi_default = 2.3
# defaults.l_rho_default = 1.6
# defaults.l_default = 1.9

parameters = utils.Parameters(no_dots=3, no_levels=1, default_parameters=defaults)
system = utils.System(parameters)
plot = utils.Plotting()

transport = utils.Transport(system, gamma=.1)

C_map = transport.C_ij_map0(0, 0)
plot.plot_conductance_map(C_map, xlabel="$V$ (mV)", ylabel="$E_F$ (meV)", suffix="mu_Ef")

C_map = transport.C_ij_map1(0, 0)
plot.plot_conductance_map(C_map, xlabel="$V_L$ (mV)", ylabel="$V_R$ (meV)", suffix="mul_mur")
