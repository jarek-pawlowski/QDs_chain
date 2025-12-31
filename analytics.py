import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import utils

au = utils.au


# Rashba 3QDs
no_dots = 3
defaults = utils.Defaults()
parameters = utils.Parameters(no_dots=no_dots, no_levels=1, default_parameters=defaults)
system = utils.System(parameters)
plot = utils.Plotting()

hamiltonian = system.full_hamiltonian()

m1 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(no_dots-2)), np.array([1,1,1,1])]).T/2.
m2 = np.concatenate([np.array([1,1,-1,-1])*-1., np.array([0,0,0,0]*(no_dots-2)), np.array([1,1,1,1])]).T/2.

m11 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(no_dots-1))]).T/2.
m22 = np.concatenate([np.array([0,0,0,0]*(no_dots-1)), np.array([1,1,-1,-1])]).T/2.

points = []
majoranas = []
vs = np.linspace(0./au.Eh, 1.5/au.Eh, num=101)
for u in np.linspace(0./au.Eh, 1.5/au.Eh, num=101):
    for v in vs:
        system.update_b(np.array([1,1,1])*u)
        system.update_mu(np.array([1,1,1])*v)
        hamiltonian = system.full_hamiltonian()
        eigs, eigvs = eigh(hamiltonian, eigvals_only=False)
        ms = []
        ehs = []
        for ie, eig in enumerate(eigs):
            ml = np.conj(eigvs[:,ie])@m11
            mr = np.conj(eigvs[:,ie])@m22
            zm = np.exp(-np.abs(eig)/(0.05/au.Eh))  # 0.1
            eh = (np.abs(eigvs[:,ie])**2).reshape(-1,2,2).sum(axis=(0,2))
            points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=0).reshape(2,2).sum(axis=1)[0]])
            ms.append(np.abs(ml+mr)*zm)  # mode projection on left - right majoranas
            ehs.append(np.amax([0., eh[0]*eh[1]*4-0.5])*2.)  # smaller than 0.5 are filtered out
        ms = np.array(ms)[np.abs(eigs).argsort()]  # sort MZM_i using |E_i|
        ehs = np.array(ehs)[np.abs(eigs).argsort()]  # same here
        majoranas.append([u, v, np.amax([0., ms[0]+ms[1]-ms[2:].sum()])*ehs[0]/2.])  # max(0, MZM_1+MZM_2 - all_others), everything is weighed by exp(-E_i/E_0)

points = np.array(points)
majoranas = np.array(majoranas)

#plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(3,3))
ax.set_xlabel("$\mu$ (mV)", labelpad=-1.)
ax.set_ylabel("$V_Z$ (meV)")
#ax.set_ylim([-1.,1.])
ax.set_aspect('equal')
ax.set_yticks([-1.,-.5,0.,.5,1.])
cc = ax.scatter(x=majoranas[:,1]*au.Eh, y=majoranas[:,0]*au.Eh, c=majoranas[:,2], s=.5)
ax.plot(vs*au.Eh, np.sqrt(system.parameters.d[0]**2+vs**2)*au.Eh, '-', c='orange')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.45)
cbar = fig.colorbar(cc, cax=cax, orientation='horizontal')
cbar.set_label(r'majoranization')
plt.savefig('rashba_3QD_a1.png', bbox_inches='tight', dpi=300)
plt.close()


system.update_b(np.array([1,1,1])*.5/au.Eh)
points = []
majoranas = []
vs = np.linspace(0./au.Eh, 1.5/au.Eh, num=101)
for u in np.linspace(0./au.Eh, 1.5/au.Eh, num=101):
    for v in vs:
        system.update_d(np.array([1,1,1])*u)
        system.update_mu(np.array([1,1,1])*v)
        hamiltonian = system.full_hamiltonian()
        eigs, eigvs = eigh(hamiltonian, eigvals_only=False)
        ms = []
        ehs = []
        for ie, eig in enumerate(eigs):
            ml = np.conj(eigvs[:,ie])@m11
            mr = np.conj(eigvs[:,ie])@m22
            zm = np.exp(-np.abs(eig)/(0.05/au.Eh))  # 0.1
            eh = (np.abs(eigvs[:,ie])**2).reshape(-1,2,2).sum(axis=(0,2))
            points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=0).reshape(2,2).sum(axis=1)[0]])
            ms.append(np.abs(ml+mr)*zm)  # mode projection on left - right majoranas
            ehs.append(np.amax([0., eh[0]*eh[1]*4-0.5])*2.)  # smaller than 0.5 are filtered out
        ms = np.array(ms)[np.abs(eigs).argsort()]  # sort MZM_i using |E_i|
        ehs = np.array(ehs)[np.abs(eigs).argsort()]  # same here
        majoranas.append([u, v, np.amax([0., ms[0]+ms[1]-ms[2:].sum()])*ehs[0]/2.])  # max(0, MZM_1+MZM_2 - all_others), everything is weighed by exp(-E_i/E_0)

points = np.array(points)
majoranas = np.array(majoranas)

#plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(3,3))
ax.set_xlabel("$\mu$ (mV)", labelpad=-1.)
ax.set_ylabel("$\Delta$ (meV)")
#ax.set_ylim([-1.,1.])
ax.set_aspect('equal')
ax.set_yticks([-1.,-.5,0.,.5,1.])
cc = ax.scatter(x=majoranas[:,1]*au.Eh, y=majoranas[:,0]*au.Eh, c=majoranas[:,2], s=.5)
ax.plot(vs*au.Eh, np.sqrt(system.parameters.b[0]**2-vs**2)*au.Eh, '-', c='orange')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.45)
cbar = fig.colorbar(cc, cax=cax, orientation='horizontal')
cbar.set_label(r'majoranization')
plt.savefig('rashba_3QD_a2.png', bbox_inches='tight', dpi=300)
plt.close()


system.update_b(np.array([1,1,1])*.5/au.Eh)
system.update_d(np.array([1,1,1])*.25/au.Eh)
points = []
majoranas = []
vs = np.linspace(0./au.Eh, 1.5/au.Eh, num=101)
for u in np.linspace(0./au.Eh, 1.5/au.Eh, num=101):
    for v in vs:
        system.update_t(np.array([1,1,1])*u)
        system.update_mu(np.array([1,1,1])*v)
        hamiltonian = system.full_hamiltonian()
        eigs, eigvs = eigh(hamiltonian, eigvals_only=False)
        ms = []
        ehs = []
        for ie, eig in enumerate(eigs):
            ml = np.conj(eigvs[:,ie])@m11
            mr = np.conj(eigvs[:,ie])@m22
            zm = np.exp(-np.abs(eig)/(0.05/au.Eh))  # 0.1
            eh = (np.abs(eigvs[:,ie])**2).reshape(-1,2,2).sum(axis=(0,2))
            points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=0).reshape(2,2).sum(axis=1)[0]])
            ms.append(np.abs(ml+mr)*zm)  # mode projection on left - right majoranas
            ehs.append(np.amax([0., eh[0]*eh[1]*4-0.5])*2.)  # smaller than 0.5 are filtered out
        ms = np.array(ms)[np.abs(eigs).argsort()]  # sort MZM_i using |E_i|
        ehs = np.array(ehs)[np.abs(eigs).argsort()]  # same here
        majoranas.append([u, v, np.amax([0., ms[0]+ms[1]-ms[2:].sum()])*ehs[0]/2.])  # max(0, MZM_1+MZM_2 - all_others), everything is weighed by exp(-E_i/E_0)

points = np.array(points)
majoranas = np.array(majoranas)

#plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(3,3))
ax.set_xlabel("$\mu$ (mV)", labelpad=-1.)
ax.set_ylabel("$t$ (meV)")
ax.set_ylim([0.,1.5])
ax.set_aspect('equal')
ax.set_yticks([0.,.5,1.])
cc = ax.scatter(x=majoranas[:,1]*au.Eh, y=majoranas[:,0]*au.Eh, c=majoranas[:,2], s=.6)
ax.plot(vs*au.Eh, -0.62+0.25+vs*au.Eh, '-', c='orange')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.45)
cbar = fig.colorbar(cc, cax=cax, orientation='horizontal')
cbar.set_label(r'majoranization')
plt.savefig('rashba_3QD_a3.png', bbox_inches='tight', dpi=300)
plt.close()



system.update_b(np.array([1,1,1])*.5/au.Eh)
system.update_d(np.array([1,1,1])*.25/au.Eh)
system.update_t(np.array([1,1,1])*.25/au.Eh)
points = []
majoranas = []
vs = np.linspace(0./au.Eh, 1.5/au.Eh, num=101)
for u in np.linspace(0., 1.0, num=101):
    for v in vs:
        system.update_l(np.array([1,1,1])*u*np.pi*2.)
        system.update_mu(np.array([1,1,1])*v)
        hamiltonian = system.full_hamiltonian()
        eigs, eigvs = eigh(hamiltonian, eigvals_only=False)
        ms = []
        ehs = []
        for ie, eig in enumerate(eigs):
            ml = np.conj(eigvs[:,ie])@m1
            mr = np.conj(eigvs[:,ie])@m2
            zm = np.exp(-np.abs(eig)/(0.05/au.Eh))  # 0.1
            eh = (np.abs(eigvs[:,ie])**2).reshape(-1,2,2).sum(axis=(0,2))
            points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=0).reshape(2,2).sum(axis=1)[0]])
            ms.append(np.abs(ml-mr)*zm)  # mode projection on left - right majoranas
            ehs.append(np.amax([0., eh[0]*eh[1]*4-0.5])*2.)  # smaller than 0.5 are filtered out
        ms = np.array(ms)[np.abs(eigs).argsort()]  # sort MZM_i using |E_i|
        ehs = np.array(ehs)[np.abs(eigs).argsort()]  # same here
        majoranas.append([u, v, np.amax([0., ms[0]+ms[1]-ms[2:].sum()])*ehs[0]/2.])  # max(0, MZM_1+MZM_2 - all_others), everything is weighed by exp(-E_i/E_0)

points = np.array(points)
majoranas = np.array(majoranas)

#plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(3,3))
ax.set_xlabel("$\mu$ (mV)", labelpad=-1.)
ax.set_ylabel("$\lambda$ (meV)")
ax.set_ylim([0.,1.])
ax.set_aspect('equal')
ax.set_yticks([0.,.5,1.])
cc = ax.scatter(x=majoranas[:,1]*au.Eh, y=majoranas[:,0]*2, c=majoranas[:,2], s=1.5)
ax.plot(vs*au.Eh, np.arctan(vs*au.Eh/.25)/np.pi, '-', c='orange')
ax.vlines(np.sqrt(0.5**2-0.25**2), -10., 10., colors='orange', linestyles='-')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.45)
cbar = fig.colorbar(cc, cax=cax, orientation='horizontal')
cbar.set_label(r'majoranization')
plt.savefig('rashba_3QD_a4.png', bbox_inches='tight', dpi=300)
plt.close()