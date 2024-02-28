import os
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib import colors


def abs2(c): return c.real**2 + c.imag**2


class AtomicUnits:
    """Class storing atomic units.

    All variables, arrays in simulations are in atomic units.

    Attributes
    ----------
    Eh : float
        Hartree energy (in meV)
    Ah : float
        Bohr radius (in nanometers)
    Th : float
        time (in picoseconds)
    Bh : float
        magnetic induction (in Teslas)
    """
    # atomic units
    Eh=27211.4 # meV
    Ah=0.05292 # nm
    Th=2.41888e-5 # ps
    Bh=235051.76 # Teslas

au = AtomicUnits()


class Defaults:
    def __init__(self, mu_max=100., t_max=100., b_max=2., d_max=5., lambda_max=2.):
        # potential within a dot (meV)
        self.mu_default = 0./au.Eh
        self.mu_range = [-mu_max/au.Eh, mu_max/au.Eh]
        # energy level separation within the dot (meV)
        self.dot_split = 1./au.Eh
        # hopping amplitude (meV)
        self.t_default = .2/au.Eh
        self.t_range = [0., t_max/au.Eh]
        # local Zeeman field (meV)
        self.b_default = .5/au.Eh
        self.b_range = [-b_max/au.Eh, b_max/au.Eh]
        # superconducting gap (meV)
        self.d_default = .5/au.Eh
        self.d_range = [0., d_max/au.Eh]
        # superconducting phase step
        self.ph_d_default = 0.
        self.ph_d_range = [-np.pi, np.pi]
        # SOI field
        # amplitude:
        self.l_default = .1*np.pi*2.
        self.l_range = np.array([0., lambda_max])*np.pi*2.
        #angles:
        self.l_rho_default = np.pi/2.
        self.l_rho_range = [0., np.pi]
        self.l_ksi_default = 0.
        self.l_ksi_range = [0., np.pi*2.]
 
        
def rand_sample(length=None, range=[0.,1.]):
    if length in None:
        return np.random.random()*(range[1]-range[0])+range[0]
    else:
        return np.random.rand(length)*(range[1]-range[0])+range[0]
         
class Parameters:
    def __init__(self, no_dots, no_levels, default_parameters):
        self.no_dots = no_dots
        self.no_levels = no_levels
        self.def_par = default_parameters
        self.mu = np.ones(self.no_dots)*self.def_par.mu_default
        self.t = np.ones(self.no_dots)*self.def_par.t_default
        self.b = np.ones(self.no_dots)*self.def_par.b_default
        self.d = np.ones(self.no_dots)*self.def_par.d_default
        self.ph_d = self.def_par.ph_d_default
        self.l = np.ones(self.no_dots)*self.def_par.l_default
        self.l_rho = np.ones(self.no_dots)*self.def_par.l_rho_default
        self.l_ksi = np.ones(self.no_dots)*self.def_par.l_ksi_default
        
    def set_random_parameters_const(self):
        self.mu = np.ones(self.no_dots)*rand_sample(range=self.def_par.mu_range)
        self.t = np.ones(self.no_dots)*rand_sample(range=self.def_par.t_range)
        self.b = np.ones(self.no_dots)*rand_sample(range=self.def_par.b_range)
        self.d = np.ones(self.no_dots)*rand_sample(range=self.def_par.d_range)
        self.ph_d = rand_sample(range=self.def_par.ph_d_range)
        self.l = np.ones(self.no_dots)*rand_sample(range=self.def_par.l_range)
        self.l_rho = np.ones(self.no_dots)*rand_sample(range=self.def_par.l_rho_range)
        self.l_ksi = np.ones(self.no_dots)*rand_sample(range=self.def_par.l_ksi_range)
        
    def set_random_parameters_free(self):
        self.mu = rand_sample(self.no_dots, self.def_par.mu_range)
        self.t = rand_sample(self.no_dots, self.def_par.t_range)
        self.b = rand_sample(self.no_dots, self.def_par.b_range)
        self.d = rand_sample(self.no_dots, self.def_par.d_range)
        self.ph_d = rand_sample(range=self.def_par.ph_d_range)
        self.l = rand_sample(self.no_dots, self.def_par.l_range)
        self.l_rho = rand_sample(self.no_dots, self.def_par.l_rho_range)
        self.l_ksi = rand_sample(self.no_dots, self.def_par.l_ksi_range)


class System:
    def __init__(self, parameters):
        self.parameters = parameters
        self.dimB = 4  # Bogoliubov block
        self.dim0 =  self.parameters.no_levels*self.dimB  # single dot block
        self.dim = self.dim0*self.parameters.no_dots

    def onsite_matrix(self, i):
        par = self.parameters
        # Nambu spinor = [\Psi^dag_up, \Psi^dag_down, \Psi_up, \Psi_down]
        onsite = np.zeros((self.dimB,self.dimB), dtype=np.complex128)
        onsite[0,0] = -par.mu[i]+par.b[i]
        onsite[1,1] = -par.mu[i]-par.b[i]
        onsite[2,2] =  par.mu[i]-par.b[i]
        onsite[3,3] =  par.mu[i]+par.b[i]
        onsite[0,3] =  par.d[i]*np.exp( 1.j*par.ph_d*i)/2.
        onsite[1,2] = -par.d[i]*np.exp( 1.j*par.ph_d*i)/2.
        onsite[2,1] = -par.d[i]*np.exp(-1.j*par.ph_d*i)/2.
        onsite[3,0] =  par.d[i]*np.exp(-1.j*par.ph_d*i)/2.
        if par.no_levels > 1:
            onsite = np.kron(np.eye(par.no_levels), onsite)
            for l in range(par.no_levels):
                for ld in range(self.dimB):
                    if ld < self.dimB/2:  # particle
                        onsite[l*self.dimB+ld, l*self.dimB+ld] -= par.def_par.dot_split*l
                    else:  # hole
                        onsite[l*self.dimB+ld, l*self.dimB+ld] += par.def_par.dot_split*l
        return onsite
        
    def hopping_matrix(self, i):
        par = self.parameters
        # lambda_versor = [np.sin(par.l_rho[i])*np.cos(par.l_ksi[i]), np.sin(par.l_rho[i])*np.sin(par.l_ksi[i]), np.cos(par.l_rho[i])]
        hopping = np.zeros((self.dimB,self.dimB), dtype=np.complex128)
        hopping[0,0] = -par.t[i]*(np.cos(par.l[i]) + 1.j*np.cos(par.l_rho[i])*np.sin(par.l[i]))
        hopping[0,1] = -par.t[i]*(1.j*np.sin(par.l_rho[i])*np.cos(par.l_ksi[i])*np.sin(par.l[i]) + np.sin(par.l_rho[i])*np.sin(par.l_ksi[i])*np.sin(par.l[i]))
        hopping[1,0] = -par.t[i]*(1.j*np.sin(par.l_rho[i])*np.cos(par.l_ksi[i])*np.sin(par.l[i]) - np.sin(par.l_rho[i])*np.sin(par.l_ksi[i])*np.sin(par.l[i]))
        hopping[1,1] = -par.t[i]*(np.cos(par.l[i]) - 1.j*np.cos(par.l_rho[i])*np.sin(par.l[i]))
        hopping[2,2] =  par.t[i]*(np.cos(par.l[i]) - 1.j*np.cos(par.l_rho[i])*np.sin(par.l[i]))
        hopping[2,3] =  par.t[i]*(-1.j*np.sin(par.l_rho[i])*np.cos(par.l_ksi[i])*np.sin(par.l[i]) - np.sin(par.l_rho[i])*np.sin(par.l_ksi[i])*np.sin(par.l[i]))
        hopping[3,2] =  par.t[i]*(-1.j*np.sin(par.l_rho[i])*np.cos(par.l_ksi[i])*np.sin(par.l[i]) + np.sin(par.l_rho[i])*np.sin(par.l_ksi[i])*np.sin(par.l[i]))
        hopping[3,3] =  par.t[i]*(np.cos(par.l[i]) + 1.j*np.cos(par.l_rho[i])*np.sin(par.l[i]))
        if par.no_levels > 1:
            hopping = np.kron(np.eye(par.no_levels), hopping)
        return hopping #+ np.conjugate(hopping.T)

    def full_hamiltonian(self):
        """
        create Hamiltonian for the whole array
        """
        hamiltonian = np.zeros((self.dim,self.dim), dtype=np.complex128)
        for i in range(self.parameters.no_dots-1):
            hamiltonian[i*self.dim0:(i+1)*self.dim0, (i+1)*self.dim0:(i+2)*self.dim0] += self.hopping_matrix(i)
        for i in range(self.parameters.no_dots):
            hamiltonian[i*self.dim0:(i+1)*self.dim0, i*self.dim0:(i+1)*self.dim0] += self.onsite_matrix(i) 
        hamiltonian += np.conjugate(np.triu(hamiltonian, k=self.dimB)).T
        return hamiltonian
    
    def parameter_sweeping(self, parameter_name, start, stop, num=101):
        values = np.linspace(start/au.Eh, stop/au.Eh, num=num)
        eigenvalues = []
        occupations = []
        for mu in values:
            setattr(self.parameters, parameter_name, np.ones(self.parameters.no_dots)*mu)
            hamiltonian = self.full_hamiltonian()
            eigs, eigv = eigh(hamiltonian, eigvals_only=False)
            eigenvalues.append(eigs)
            edge_occ = np.zeros(len(eigs))
            for i in range(len(eigs)):
                dots_occ = np.sum(abs2(eigv[:,i]).reshape(-1, self.dim0), axis=1)
                edge_occ[i] = dots_occ[0]+dots_occ[-1]  # edge states
            occupations.append(edge_occ)
        return values, eigenvalues, occupations


class Plotting:
    def __init__(self, directory=None):
        if directory is not None:
            self.directory = os.path.join('./', directory)
            os.makedirs(directory, exist_ok=True)
        else:
            self.directory = './'
        self.pointsize = .2
        self.pointcolor='tab:blue'

    def plot_eigenvalues(self, parameters_sweeping, eigenvalues, occupations=None, xlabel=None, range=None, filename='eigenvalues.png'):
        fig, ax = plt.subplots()
        if range is not None:
            ax.set_ylim(range[0], range[1])
        ax.set_xlabel("subsequent eigenstates")
        if xlabel is not None: ax.set_xlabel(xlabel)
        ax.set_ylabel("energy (meV)")
        if occupations is not None:
            for i,p in enumerate(parameters_sweeping): 
                occupy = ax.scatter(x=np.tile(p*au.Eh, len(eigenvalues[i])), y=eigenvalues[i]*au.Eh, c=occupations[i], s=self.pointsize)
            cbar = fig.colorbar(occupy, ax=ax)
            cbar.set_label(r'edges occupation')
        else:
            for i,p in enumerate(parameters_sweeping): 
                occupations = ax.scatter(x=np.tile(p*au.Eh, len(eigenvalues[i])), y=eigenvalues[i]*au.Eh, c=self.pointcolor, s=self.pointsize)
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
        
    def plot_hamiltonian(self, hamiltonian):
        f_, axs = plt.subplots(1,2) 
        axs[0].imshow(hamiltonian.real)
        axs[1].imshow(hamiltonian.imag)
        plt.savefig('hamiltonian.png')
        plt.close()