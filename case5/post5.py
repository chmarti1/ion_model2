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
        # study value
        if not p.param.beta in bybeta:
            bybeta[p.param.beta] = []
        bybeta[p.param.beta].append(p)

marker_list = [
    {'ls':'none', 'marker':'o', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'s', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'d', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'o', 'markersize':8, 'mfc':'k', 'mec':'k'},
    {'ls':'none', 'marker':'s', 'markersize':8, 'mfc':'k', 'mec':'k'},
    {'ls':'none', 'marker':'d', 'markersize':8, 'mfc':'k', 'mec':'k'},
    ]
    
ax1 = lp.init_fig('$\\beta \\Omega$', '$J$', label_size=16)
#ax1.set_xscale('log')
ax1.grid(True, which='both')
blist = list(bybeta.keys())
blist.sort()
for beta in blist:
    plist = bybeta[beta]
    ax = lp.init_fig('$z$', '$\\eta$, $\\nu$, $\\psi$', label_size=16)

    J = []
    phi = []
    omega = []
    
    for p in plist:
        J.append(p.J)
        phi.append(p.phi[-1])
        omega.append(p.param.omega)
        
        ax.plot(p.z, p.eta, 'k', label='$\\eta$')
        ax.plot(p.z, p.nu, 'k--', label='$\\nu$')
        ax.plot(p.z, p.psi, 'k:', label='$\\psi$')
        
    J = np.array(J)
    omega = np.array(omega)
    phi = np.array(phi)
        
    ax.set_title(f'$\\beta$={beta}')
    fig = ax.get_figure()
    
    lp.floating_legend(fig, (0.9, 0.9),
        [[{'color':'k', 'marker':None, 'ls':'-'}, '$\\eta$'],
        [{'color':'k', 'marker':None, 'ls':'--'}, '$\\nu$'],
        [{'color':'k', 'marker':None, 'ls':':'}, '$\\psi$']],
        loc_edge = 'rt')
    fig.savefig(os.path.join(postdir, f'beta{int(beta*10):02d}.png'))

    ax1.plot(omega*beta, J, **marker_list.pop(), label = f'$\\beta$ = {beta:0.1f}')
    #ax1.plot(omega, phi, **marker_list.pop(), label = f'$\\beta$ = {beta:0.1f}')
ax1.legend(loc=0)
ax1.set_xscale('log')
ax1.set_yscale('log')
fig = ax1.get_figure()
fig.savefig(os.path.join(postdir, 'psiphi.png'))
