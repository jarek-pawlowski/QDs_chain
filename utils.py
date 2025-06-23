import os
import numpy as np
from scipy.linalg import eigh, inv
import matplotlib.pyplot as plt
from matplotlib import colors


def abs2(c): return c.real**2 + c.imag**2

def rand_sample(length=None, range=[0.,1.]):
    if length is None:
        return np.random.random()*(range[1]-range[0])+range[0]
    else:
        return np.random.rand(length)*(range[1]-range[0])+range[0]


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
        self.mu_default = 0.62/au.Eh
        self.mu_range = [.2/au.Eh, 1./au.Eh]
        #self.mu_range = [.61/au.Eh, .63/au.Eh]
        #self.mu_range = [.62/au.Eh, .62/au.Eh]
        # energy level separation within the dot (meV)
        self.dot_split = 1./au.Eh
        # hopping amplitude (meV)
        self.t_default = .25/au.Eh
        self.t_range = [.1/au.Eh, .4/au.Eh]
        #self.t_range = [.24/au.Eh, .26/au.Eh]
        #self.t_range = [.25/au.Eh, .25/au.Eh]
        # local Zeeman field (meV)
        self.b_default = .5/au.Eh  #.5/au.Eh
        self.b_range = [-b_max/au.Eh, b_max/au.Eh]
        # superconducting gap (meV)
        self.d_default = 0.25/au.Eh
        self.d_range = [0., d_max/au.Eh]
        # superconducting phase step
        self.ph_d_default = 0.
        self.ph_d_range = [-np.pi, np.pi]
        # SOI field
        # amplitude:
        self.l_default = 0.133*np.pi*2.  # 0.166*np.pi*2. for 2QD, for 3QD: 0.133*np.pi*2.  # optimal: d/b = cos(l)
        self.l_range = np.array([0.,.5])*np.pi*2.
        #self.l_range = np.array([0.130, 0.136])*np.pi*2.
        #self.l_range = np.array([0.133, 0.133])*np.pi*2.
        #angles:
        self.l_rho_default = np.pi/2.
        self.l_rho_range = [0., np.pi]
        self.l_ksi_default = np.pi/2.
        self.l_ksi_range = [0., np.pi*2.]
         
         
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
        
    def set_random_parameters_free_reduced(self):
        self.mu = rand_sample(self.no_dots, self.def_par.mu_range)
        self.t = rand_sample(self.no_dots-1, self.def_par.t_range)
        self.l = rand_sample(self.no_dots-1, self.def_par.l_range)
        
    def get_parameters_reduced(self):
        return np.concatenate([self.mu, self.t, self.l])


class System:
    def __init__(self, parameters):
        self.parameters = parameters
        self.dimB = 4  # Bogoliubov block
        self.dim0 =  self.parameters.no_levels*self.dimB  # single dot block
        self.dim = self.dim0*self.parameters.no_dots
        # auxiliary matrices
        U = np.array([[1,-1j],[1,1j]]/np.sqrt(2.))
        U1 = np.linalg.inv(U)
        self.U = np.kron(U, np.eye(2))
        self.U1 = np.kron(U1, np.eye(2))
        self.sx = np.array([[0,1],[1,0]])
        self.sy = np.array([[0,-1j],[1j,0]])
        self.sz = np.array([[1,0],[0,-1]])

    def onsite_matrix(self, i):
        par = self.parameters
        onsite = np.zeros((self.dimB,self.dimB), dtype=np.complex128)
        onsite[0,0] = -par.mu[i]+par.b[i]
        onsite[1,1] = -par.mu[i]-par.b[i]
        onsite[2,2] =  par.mu[i]-par.b[i]
        onsite[3,3] =  par.mu[i]+par.b[i]
        onsite[0,3] =  par.d[i]*np.exp( 1.j*par.ph_d*i)
        onsite[1,2] = -par.d[i]*np.exp( 1.j*par.ph_d*i)
        onsite[2,1] = -par.d[i]*np.exp(-1.j*par.ph_d*i)
        onsite[3,0] =  par.d[i]*np.exp(-1.j*par.ph_d*i)
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
        hopping = np.zeros((self.dimB,self.dimB), dtype=np.complex128)
        lambda_versor = [np.sin(par.l_rho[i])*np.cos(par.l_ksi[i]), np.sin(par.l_rho[i])*np.sin(par.l_ksi[i]), np.cos(par.l_rho[i])]
        ph = np.cos(par.l[i])*np.eye(2)+1j*np.sin(par.l[i])*(self.sx*lambda_versor[0]+self.sy*lambda_versor[1]+self.sz*lambda_versor[2])
        hopping[:2,:2] = par.t[i]*ph
        ph = np.cos(par.l[i])*np.eye(2)-1j*np.sin(par.l[i])*(self.sx*lambda_versor[0]+self.sy.T*lambda_versor[1]+self.sz*lambda_versor[2])
        hopping[2:4,2:4] = -par.t[i]*ph
        if par.no_levels > 1:
            hopping = np.kron(np.eye(par.no_levels), hopping)
        return hopping

    def full_hamiltonian(self):
        """
        create Hamiltonian for the whole array
        # Nambu spinor representation: Psi = [\psi_up, \psi_down, \psi^dag_up, \psi^dag_down]^T
        """
        hamiltonian = np.zeros((self.dim,self.dim), dtype=np.complex128)
        for i in range(self.parameters.no_dots-1):
            hamiltonian[i*self.dim0:(i+1)*self.dim0, (i+1)*self.dim0:(i+2)*self.dim0] += self.hopping_matrix(i)
        for i in range(self.parameters.no_dots):
            hamiltonian[i*self.dim0:(i+1)*self.dim0, i*self.dim0:(i+1)*self.dim0] += self.onsite_matrix(i) 
        hamiltonian = np.triu(hamiltonian) + np.conjugate(np.triu(hamiltonian, 1)).T
        return hamiltonian
    
    def majorana_representation(self, hamiltonian):
        """
        Representation discussed in:
        https://arxiv.org/pdf/2006.10153
        """
        d = int(self.dim/self.dimB)
        H = hamiltonian.reshape(d, self.dimB, d, self.dimB)
        M = np.zeros_like(H)
        for i in range(d):
            for j in range(d):
                M[i,:,j,:] = (self.U1 @ H[i,:,j,:] @ self.U)*-1j  # (M-M^T)_ij, Eq. (14)

        print(np.abs(M.imag).sum())        
        return M.reshape(self.dim, self.dim)
    
    def parameter_sweeping(self, parameter_name, start, stop, num=101, majorana_repr=False):
        values = np.linspace(start/au.Eh, stop/au.Eh, num=num)
        eigenvalues = []
        occupations = []
        polarizations = []
        for mu in values:
            setattr(self.parameters, parameter_name, np.ones(self.parameters.no_dots)*mu)
            hamiltonian = self.full_hamiltonian()
            #
            if majorana_repr:
                majorana = self.majorana_representation(hamiltonian)
                hamiltonian = -majorana @ majorana
            #
            eigs, eigv = eigh(hamiltonian, eigvals_only=False)
            if majorana_repr: eigs = np.sqrt(eigs)
            eigenvalues.append(eigs)
            edge_occ = np.zeros(len(eigs))
            polars = np.zeros(len(eigs))
            for i in range(len(eigs)):
                dots_occ = np.sum(abs2(eigv[:,i]).reshape(-1, self.dim0), axis=1)
                edge_occ[i] = dots_occ[0]+dots_occ[-1]  # edge states
                # polarization
                V = eigv[:,i].reshape(-1, self.parameters.no_levels, self.dimB)
                ai = V[:,:,0]
                bi = V[:,:,1]
                ci = V[:,:,2]
                di = V[:,:,3]
                if majorana_repr: 
                    Pi = np.sum(bi*np.conjugate(di)-ai*np.conjugate(ci), axis=1).real*2.
                else:
                    Pi = np.sum(bi*np.conjugate(di)-ai*np.conjugate(ci), axis=1).imag*2. 
                polars[i] = Pi[0]-Pi[-1]
            #if np.abs(eigs[i]) < 0.00004: breakpoint()
            occupations.append(edge_occ)
            polarizations.append(polars)
        return values, eigenvalues, occupations, polarizations
    
    def update_mu(self, new_mu):
        self.parameters.mu = new_mu
        
    def update_t(self, new_t):
        self.parameters.t = new_t
        
    def update_l(self, new_l):
        self.parameters.l = new_l
        
    def update_b(self, new_b):
        self.parameters.b = new_b


class Transport:
    def __init__(self, system, gamma=1., density=201, reference_point=None):
        self.s = system
        self.d = density
        if reference_point is not None:
            self.ref_mu = reference_point.mu
            self.ref_b = reference_point.b
        else:
            self.ref_mu = np.ones(self.s.parameters.no_dots)*self.s.parameters.def_par.mu_default
            self.ref_b = np.ones(self.s.parameters.no_dots)*self.s.parameters.def_par.b_default
        # build W matrix
        self.W = np.zeros((self.s.dim,self.s.dim))
        sg = np.sqrt(gamma/au.Eh)
        site = np.diag([sg,sg,-sg,-sg])
        if self.s.parameters.no_levels > 1:
            site = np.kron(np.eye(self.s.parameters.no_levels), site)
        i = 0  # left lead:
        self.W[i*self.s.dim0:(i+1)*self.s.dim0, i*self.s.dim0:(i+1)*self.s.dim0] += site 
        i = self.s.parameters.no_dots-1  # right lead
        self.W[i*self.s.dim0:(i+1)*self.s.dim0, i*self.s.dim0:(i+1)*self.s.dim0] += site 
    def S_matrix(self, ef, hamiltonian):
        Smat = inv(np.eye(self.s.dim)*ef - hamiltonian + self.W @ np.conjugate(self.W.T)*0.5j)
        Smat = np.conjugate(self.W.T) @ Smat @ self.W
        Smat = np.eye(self.s.dim) - Smat*1.j
        return Smat
    def C_ij_(self, i, j, S):
        """
        C = dI_i/dV_j 
        i = {0,1} = {L,R} <- current via L/R lead
        j = {0,1} = {L,R} <- L/R lead (electrode) voltage
        """
        dij = np.eye(2)[i,j]
        i *= self.s.parameters.no_dots-1
        j *= self.s.parameters.no_dots-1
        #
        Sij = S.reshape(np.rint(self.s.dim/4).astype(int),2,2,np.rint(self.s.dim/4).astype(int),2,2)
        r_ee = Sij[i,0,:,j,0,:]
        r_he = Sij[i,1,:,j,0,:]
        return (2.*dij*self.s.parameters.no_levels - np.trace(r_ee @ np.conjugate(r_ee.T)) + np.trace(r_he @ np.conjugate(r_he.T))).real
    def C_ij(self, i, j, ef):
        """
        ef = Fermi energy
        """
        return self.C_ij_(i, j, self.S_matrix(ef, self.s.full_hamiltonian()))
    def kitaevH(self, m):
        H = [[m/2, 0.5, 0, 0, -0.5, 0], 
             [0.5, m/2, 0.5, 0.5, 0, -0.5], 
             [0, 0.5, m/2, 0, 0.5, 0], 
             [0, 0.5, 0, -(m/2), -0.5, 0], 
             [-0.5, 0, 0.5, -0.5, -(m/2), -0.5], 
             [0, -0.5, 0, 0, -0.5, -(m/2)]]
        return np.kron(np.array(H).reshape(2,3,2,3).transpose(1,0,3,2).reshape(6,6), np.eye(2)).astype(complex)
    def C_ij_test(self, i, j, ef, mu):
        return self.C_ij_(i, j, self.S_matrix(ef, self.kitaevH(mu)))
    def C_ij_map0(self, i, j):
        C_map = []
        efs = np.linspace(-2./au.Eh, 2./au.Eh, num=self.d, endpoint=True)
        mus = np.linspace(-2./au.Eh, 2./au.Eh, num=self.d, endpoint=True)
        for mu in mus:
            for ef in efs:
                self.s.update_mu(np.ones(self.s.parameters.no_dots)*mu)
                #C_map.append([ef, mu, self.C_ij_test(i, j, ef, mu)])
                C_map.append([mu, ef, self.C_ij(i, j, ef)])
        return np.array(C_map)
    def C_ij_map1(self, i, j):
        C_map = []
        efs = np.linspace(-2./au.Eh, 2./au.Eh, num=self.d, endpoint=True)
        mus = np.linspace(-2./au.Eh, 2./au.Eh, num=self.d, endpoint=True)
        for mul in efs:
            for mur in mus:
                self.s.update_mu(np.array([mul, 0., mur]))
                #C_map.append([ef, mu, self.C_ij_test(i, j, ef, mu)])
                C_map.append([mul, mur, self.C_ij(i, j, 0.)])
        return np.array(C_map)
    def C_ij_map2(self, i, j):
        C_map = []
        bs = np.linspace(0./au.Eh, 2./au.Eh, num=self.d, endpoint=True)
        efs = np.linspace(-1./au.Eh, 1./au.Eh, num=self.d, endpoint=True)
        for b in bs:
            for ef in efs:
                self.s.update_b(np.ones(self.s.parameters.no_dots)*b)
                C_map.append([b, ef, self.C_ij(i, j, ef)])
        return np.array(C_map)
    def C_ij_map20(self, i, j):
        C_map = []
        efs = np.linspace(-2./au.Eh, 2./au.Eh, num=self.d, endpoint=True)
        mus = np.linspace(-2./au.Eh, 2./au.Eh, num=self.d, endpoint=True)
        for mul in mus:
            for ef in efs:
                self.s.update_mu(np.array([mul-0.65/au.Eh, -0.65/au.Eh]))
                C_map.append([ef, mul, self.C_ij(i, j, ef)])
        return np.array(C_map)
    def C_ij_map21(self, i, j, k):
        C_map = []
        efs = np.linspace(-1.5/au.Eh, 1.5/au.Eh, num=self.d, endpoint=True)
        dets = np.linspace(-1./au.Eh, 1./au.Eh, num=self.d, endpoint=True)
        mask = [0]*self.s.parameters.no_dots
        mask[k] = 1
        self.s.update_b(self.ref_b)
        for det in dets:
            for ef in efs:
                self.s.update_mu(self.ref_mu+np.array(mask)*det)
                C_map.append([det, ef, self.C_ij(i, j, ef)])
        return np.array(C_map)
    def C_ij_map22(self, i, j):
        C_map = []
        efs = np.linspace(-1.5/au.Eh, 1.5/au.Eh, num=self.d, endpoint=True)
        bs = np.linspace(-.5/au.Eh, .5/au.Eh, num=self.d, endpoint=True)
        self.s.update_mu(self.ref_mu)
        for b in bs:
            for ef in efs:
                self.s.update_b(self.ref_b+np.ones(self.s.parameters.no_dots)*b)
                C_map.append([b, ef, self.C_ij(i, j, ef)])
        return np.array(C_map)


class Poor:
    def kitaev2H(self, vl, vr, tt, dd):
        H = [[ vl, tt, 0 ,-dd], 
             [ tt, vr, dd, 0 ], 
             [ 0 , dd,-vl,-tt], 
             [-dd, 0 ,-tt,-vr]]
        return np.array(H)/2.
    def kitaev3H(self, vl, vc, vr, t1, t2, d1, d2, fi):
        z2 = d2*np.exp(1.j*fi) 
        z3 = d2*np.exp(-1.j*fi) 
        H = [[ vl, t1, 0 , 0 ,-d1, 0 ], 
             [ t1, vc, t2, d1, 0 ,-z2], 
             [ 0 , t2, vr, 0 , z2, 0 ], 
             [ 0 , d1, 0 ,-vl,-t1, 0 ], 
             [-d1, 0 , z3,-t1,-vc,-t2], 
             [ 0 ,-z3, 0 , 0 ,-t2,-vr]]
        return np.array(H)/2.


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
        
    def plot_conductance_map(self, C_map, xlabel="", ylabel="", suffix=""):
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cc = ax.scatter(x=C_map[:,0]*au.Eh, y=C_map[:,1]*au.Eh, c=C_map[:,2], s=2.)  #, norm=colors.LogNorm())
        cbar = fig.colorbar(cc, ax=ax)
        cbar.set_label(r'conductance')
        plt.savefig(os.path.join(self.directory, 'conductance_'+suffix+'.png'))
        plt.close()
        
    def plot_conductance_map_(self, C_map, xlabel="", ylabel="", suffix=""):
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cc = ax.imshow(C_map.T)
        cbar = fig.colorbar(cc, ax=ax)
        cbar.set_label(r'conductance')
        plt.savefig(os.path.join(self.directory, 'conductance_'+suffix+'.png'))
        plt.close()