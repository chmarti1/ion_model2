#!/usr/bin/python3
import numpy as np
import sheath1d as s1d
import os
from miscpy import lplot as lp

datadir = 'data'
postdir = 'post'

lp.set_defaults(font_size=16, legend_font_size=16)

ax1 = lp.init_fig('$z$', '$\\eta$, $\\nu$, $\\psi$', label_size=16)

byomega = {}

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
        ax.set_title(f'$\\Omega$={p.param.omega} $\\phi$={p.param.phia}')
    
        ax.legend(loc=0)
        
        fig = ax.get_figure()
        fig.savefig(os.path.join(postdir, base+'.png'))
        lp.plt.close(fig)

        # Build a dictionary of simulation results organized by the
        # study value
        if not p.param.omega in byomega:
            byomega[p.param.omega] = []
        byomega[p.param.omega].append(p)

#marker_list = [
#    {'ls':'none', 'marker':'o', 'markersize':8, 'mfc':'w', 'mec':'k'},
#    {'ls':'none', 'marker':'s', 'markersize':8, 'mfc':'w', 'mec':'k'},
#    {'ls':'none', 'marker':'d', 'markersize':8, 'mfc':'k', 'mec':'k'},
#    {'ls':'none', 'marker':'^', 'markersize':8, 'mfc':'k', 'mec':'k'}]
marker_list = [
    {'ls':'-', 'color':'k', 'marker':'o', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'-', 'color':'k', 'marker':'s', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'-', 'color':'k', 'marker':'d', 'markersize':8, 'mfc':'k', 'mec':'k'},
    {'ls':'-', 'color':'k', 'marker':'^', 'markersize':8, 'mfc':'k', 'mec':'k'}]
    
ax1 = lp.init_fig('$\\phi_\\infty$', '$J$', label_size=16)
#ax1.set_xscale('log')
ax1.grid(True, which='both')
Wlist = list(byomega.keys())
Wlist.sort()
for omega in Wlist:
    plist = byomega[omega]
    ax = lp.init_fig('$z$', '$\\eta$, $\\nu$, $\\psi$', label_size=16)

    J = []
    phi = []
    
    for p in plist:
        J.append(p.J)
        phi.append(p.phi[-1])
        
        ax.plot(p.z, p.eta, 'k', label='$\\eta$')
        ax.plot(p.z, p.nu, 'k--', label='$\\nu$')
        ax.plot(p.z, p.psi, 'k:', label='$\\psi$')
        
    ax.set_title(f'$\\Omega$={omega}')
    fig = ax.get_figure()
    
    lp.floating_legend(fig, (0.9, 0.9),
        [[{'color':'k', 'marker':None, 'ls':'-'}, '$\\eta$'],
        [{'color':'k', 'marker':None, 'ls':'--'}, '$\\nu$'],
        [{'color':'k', 'marker':None, 'ls':':'}, '$\\psi$']],
        loc_edge = 'rt')
    fig.savefig(os.path.join(postdir, f'omega{int(omega*10):02d}.png'))

    # Sort by phi value
    I = np.argsort(phi)
    phi = np.array(phi)[I]
    J = np.array(J)[I]

    ax1.plot(phi, J, **marker_list.pop(0), label = f'$\\Omega$ = {omega:0.1f}')
    #ax1.plot(omega, phi, **marker_list.pop(), label = f'$\\beta$ = {beta:0.1f}')

ax1.set_ylim([-5,5])
ax1.legend(loc=0)
fig = ax1.get_figure()
fig.savefig(os.path.join(postdir, 'jphi.png'))
