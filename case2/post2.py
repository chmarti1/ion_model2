#!/usr/bin/python3
import numpy as np
import sheath1d as s1d
import os
from miscpy import lplot as lp

datadir = 'data'
postdir = 'post'

lp.set_defaults(font_size=16, legend_font_size=16)

ax1 = lp.init_fig('$z$', '$\\eta$, $\\nu$, $\\psi$', label_size=16)

bybeta = {}

for thisfile in os.listdir(datadir):
    base,_,ext = thisfile.rpartition('.')
    if ext == 'gz':
        source = os.path.join(datadir, thisfile)
        
        print(source)
        
        p = s1d.load(source)
        
        ax = lp.init_fig('$z$', '$\\eta$, $\\nu$, $\\psi$', label_size=16)
        ax.plot(p.z, p.eta, 'k', label='$\\eta$')
        ax.plot(p.z, p.nu, 'k--', label='$\\nu$')
        ax.plot(p.z, p.psi, 'k:', label='$\\psi$')
        ax.set_title(f'$\\Omega$={p.param.omega} $\\beta$={p.param.beta}')
    
        ax.legend(loc=0)
        
        fig = ax.get_figure()
        fig.savefig(os.path.join(postdir, base+'.png'))
        lp.plt.close(fig)

        # Build a dictionary of simulation results organized by the
        # alpha value
        if not p.param.beta in bybeta:
            bybeta[p.param.beta] = []
        bybeta[p.param.beta].append(p)

marker_list = [
    {'ls':'none', 'marker':'o', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'s', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'d', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'o', 'markersize':8, 'mfc':'k', 'mec':'k'},
    {'ls':'none', 'marker':'s', 'markersize':8, 'mfc':'k', 'mec':'k'}]

blist = list(bybeta.keys())
blist.sort()

ax1 = lp.init_fig('$\\beta^{2}\\Omega$', '$\\phi_\\infty$', label_size=16)
ax1.set_xscale('log')
ax1.grid(True, which='both')

ax2 = lp.init_fig('$\\Omega$', '$\\phi_\\infty$', label_size=16)
ax2.set_xscale('log')
ax2.grid(True, which='both')
for beta in blist:
    plist = bybeta[beta]
    
    ax = lp.init_fig('$z$', '$\\eta$, $\\nu$, $\\psi$', label_size=16)

    omega = []
    phi = []
    
    for p in plist:
        omega.append(p.param.omega)
        phi.append(p.phi[-1])
        
        ax.plot(p.z, p.eta, 'k', label='$\\eta$')
        ax.plot(p.z, p.nu, 'k--', label='$\\nu$')
        ax.plot(p.z, p.psi, 'k:', label='$\\psi$')
        
    ax.set_title(f'$\\beta$={beta}')
    fig = ax.get_figure()
    
    lp.floating_legend(fig, (0.9, 0.9),
        [[{'color':'k', 'marker':None, 'ls':'-'}, '$\\eta$'],
        [{'color':'k', 'marker':None, 'ls':'--'}, '$\\nu$'],
        [{'color':'k', 'marker':None, 'ls':':'}, '$\\psi$']],
        loc_edge = 'rt')
    fig.savefig(os.path.join(postdir, f'beta{int(beta*10):02d}.png'))

    mm = marker_list.pop(0)
    ax1.plot(np.array(omega)*beta**2, phi, **mm, label = f'$\\beta$ = {beta:0.2f}')
    ax2.plot(omega, phi, **mm, label = f'$\\beta$ = {beta:0.1f}')
ax1.legend(loc=0)
fig = ax1.get_figure()
fig.savefig(os.path.join(postdir, 'phib2w.png'))

ax2.legend(loc=0)
fig = ax2.get_figure()
fig.savefig(os.path.join(postdir, 'phiw.png'))
