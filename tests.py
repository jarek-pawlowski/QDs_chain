import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import utils

au = utils.au


# Kitaev 2QD
aux = utils.Poor()
m2l = np.array([1,0,1,0]).T/np.sqrt(2.)
m2l_ = np.array([1,0,-1,0]).T*1.j/np.sqrt(2.)
m2r = np.array([0,1,0,-1]).T*1.j/np.sqrt(2.)
m2r_ = np.array([0,1,0,1]).T/np.sqrt(2.)

points = []
for v in np.linspace(-5., 5., num=101):
    eigs, eigvs = eigh(aux.kitaev2H(v, v, 1., -1.), eigvals_only=False)
    for ie, eig in enumerate(eigs):
        ml = eigvs[:,ie]@m2l
        mr = eigvs[:,ie]@m2r
        zm = np.exp(-np.abs(eig)/0.1)
        points.append([v, eig+ie*0.1, (np.abs(ml-mr))*zm])

points = np.array(points)

fig, ax = plt.subplots()
ax.set_xlabel("v_gate")
ax.set_ylabel("energy")
cc = ax.scatter(x=points[:,0], y=points[:,1], c=points[:,2], s=1.)
cbar = fig.colorbar(cc, ax=ax)
cbar.set_label(r'majoranization')
plt.savefig('kitayev_2QD.png', dpi=200)
plt.close()


# Kitaev 3QD
m3l = np.array([1,0,0,1,0,0]).T/np.sqrt(2.)
m3l_ = np.array([1,0,0,-1,0,0]).T*1.j/np.sqrt(2.)
m3r = np.array([0,0,1,0,0,-1]).T*1.j/np.sqrt(2.)
m3r_ = np.array([0,0,1,0,0,1]).T/np.sqrt(2.)

points = []
majoranas = []
for v in np.linspace(-5., 5., num=101):
    eigs, eigvs = eigh(aux.kitaev3H(v, 0., 0., 1., 1., 1., 1., 0.*np.pi), eigvals_only=False)
    ms = []
    for ie, eig in enumerate(eigs):
        ml = np.conj(eigvs[:,ie])@m3l
        mr = np.conj(eigvs[:,ie])@m3r
        ml_ = np.conj(eigvs[:,ie])@m3l_
        mr_ = np.conj(eigvs[:,ie])@m3r_
        zm = np.exp(-np.abs(eig)/0.2)
        points.append([v, eig+ie*0.05, (np.abs(ml_)-np.abs(mr_))*zm])
        ms.append(np.abs(ml_-mr_)*zm)  # mode projection on left - right majoranas
    ms = np.array(ms)[np.abs(eigs).argsort()]  # sort MZM_i using |E_i|
    majoranas.append([v, np.amax([0., ms[0]+ms[1]-ms[2:].sum()])])  # max(0, MZM_1+MZM_2 - all_others), everything is weighed by exp(-E_i/E_0)

points = np.array(points)
majoranas = np.array(majoranas)

fig, ax = plt.subplots()
ax.set_xlabel("v_gate")
ax.set_ylabel("energy")
cc = ax.scatter(x=points[:,0], y=points[:,1], c=points[:,2], s=1., cmap='coolwarm', vmin=-1, vmax=1.)
ax.plot(majoranas[:,0], majoranas[:,1], c='orange')
cbar = fig.colorbar(cc, ax=ax)
cbar.set_label(r'majoranization')
plt.savefig('kitayev_3QD.png', dpi=200)
plt.close()


# Rashba 3-5QDs
no_dots = 3
defaults = utils.Defaults()
parameters = utils.Parameters(no_dots=no_dots, no_levels=1, default_parameters=defaults)
system = utils.System(parameters)
plot = utils.Plotting()

hamiltonian = system.full_hamiltonian()

m3l = np.array([1,1,1,1]+[0,0,0,0]*(no_dots-1)).T/2.
m3l_ = np.array([1,1,-1,-1]+[0,0,0,0]*(no_dots-1)).T*1.j/2.
m3r = np.array([0,0,0,0]*(no_dots-1)+[1,1,-1,-1]).T*1.j/2.
m3r_ = np.array([0,0,0,0]*(no_dots-1)+[1,1,1,1]).T/2.

m1 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(no_dots-2)), np.array([1,1,1,1])]).T/2.
m2 = np.concatenate([np.array([1,1,-1,-1])*-1., np.array([0,0,0,0]*(no_dots-2)), np.array([1,1,1,1])]).T/2.

points = []
majoranas = []
for v in np.linspace(-1./au.Eh, 1./au.Eh, num=101):
    system.update_mu(np.array([2.8721024815950908e-05, -2.570004880207832e-05, -3.383476842390834e-05]) + np.array([1,1,1])*v)
    system.update_t(np.array([9.187325900174191e-06, 2.71874078071042e-05]))
    system.update_l(np.array([4.616630964537762, 2.085409866967125]))
    hamiltonian = system.full_hamiltonian()
    eigs, eigvs = eigh(hamiltonian, eigvals_only=False)
    ms = []
    ehs = []
    for ie, eig in enumerate(eigs):
        ml = np.conj(eigvs[:,ie])@m1
        mr = np.conj(eigvs[:,ie])@m2
        zm = np.exp(-np.abs(eig)/(0.05/au.Eh))  # 0.1
        eh = (np.abs(eigvs[:,ie])**2).reshape(-1,2,2).sum(axis=(0,2))
        points.append([v, eig+ie*0.005/au.Eh, (np.abs(ml)-np.abs(mr))*zm])
        #points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=0).reshape(2,2).sum(axis=1)[0]])
        #points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=1)[0]*2])
        #points.append([v, eig+ie*0.005/au.Eh, eh[1]*eh[0]*4])
        ms.append(np.abs(ml-mr)*zm)  # mode projection on left - right majoranas
        ehs.append(np.amax([0., eh[1]*eh[0]*4-0.5])*2.)
    ms = np.array(ms)[np.abs(eigs).argsort()]  # sort MZM_i using |E_i|
    ehs = np.array(ehs)[np.abs(eigs).argsort()]
    majoranas.append([v, np.amax([0., ms[0]+ms[1]-ms[2:].sum()])*ehs[0]])  # max(0, MZM_1+MZM_2 - all_others), everything is weighed by exp(-E_i/E_0)

points = np.array(points)
majoranas = np.array(majoranas)

fig, ax = plt.subplots()
ax.set_xlabel("v_gate")
ax.set_ylabel("energy")
ax.set_ylim([-2.5,2.5])
cc = ax.scatter(x=points[:,0]*au.Eh, y=points[:,1]*au.Eh, c=points[:,2], s=1., cmap='coolwarm')
ax.plot(majoranas[:,0]*au.Eh, majoranas[:,1]/2., c='orange')
cbar = fig.colorbar(cc, ax=ax)
cbar.set_label(r'majoranization')
plt.savefig('rashba_3QD.png', dpi=200)
plt.close()