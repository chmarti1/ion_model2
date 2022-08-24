#!/usr/bin/python3

import sheath1d as s1d
import os
from miscpy import lplot as lp

datadir = 'data'
postdir = 'post'

lp.set_defaults(font_size=16, legend_font_size=16)

ax1 = lp.init_fig('$z$', '$\\eta$, $\\nu$, $\\psi$', label_size=16)

byalpha = {}

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
        ax.set_title(f'$R$={p.param.R} $\\alpha$={p.param.alpha}')
    
        ax.legend(loc=0)
        
        fig = ax.get_figure()
        fig.savefig(os.path.join(postdir, base+'.png'))
        lp.plt.close(fig)

        # Build a dictionary of simulation results organized by the
        # alpha value
        if not p.param.alpha in byalpha:
            byalpha[p.param.alpha] = []
        byalpha[p.param.alpha].append(p)

marker_list = [
    {'ls':'none', 'marker':'o', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'s', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'d', 'markersize':8, 'mfc':'w', 'mec':'k'},
    {'ls':'none', 'marker':'^', 'markersize':8, 'mfc':'k', 'mec':'k'}]
    
ax1 = lp.init_fig('$R$', '$\\psi_\\infty$', label_size=16)
#ax1.set_xscale('log')
ax1.grid(True, which='both')

alist = list(byalpha.keys())
alist.sort()
for alpha in alist:
    plist = byalpha[alpha]
    
    ax = lp.init_fig('$z$', '$\\eta$, $\\nu$, $\\psi$', label_size=16)

    R = []
    psi = []
    
    for p in plist:
        R.append(p.param.R)
        psi.append(p.psi[-1])
        
        ax.plot(p.z, p.eta, 'k', label='$\\eta$')
        ax.plot(p.z, p.nu, 'k--', label='$\\nu$')
        ax.plot(p.z, p.psi, 'k:', label='$\\psi$')
        
    ax.set_title(f'$\\alpha$={alpha}')
    fig = ax.get_figure()
    
    lp.floating_legend(fig, (0.9, 0.9),
        [[{'color':'k', 'marker':None, 'ls':'-'}, '$\\eta$'],
        [{'color':'k', 'marker':None, 'ls':'--'}, '$\\nu$'],
        [{'color':'k', 'marker':None, 'ls':':'}, '$\\psi$']],
        loc_edge = 'rt')
    fig.savefig(os.path.join(postdir, f'alpha{int(alpha*10):02d}.png'))
    
    ax1.plot(R, psi, **marker_list.pop(), label = f'$\\alpha$ = {alpha:0.1f}')
    ax1.legend(loc=0)
    fig = ax1.get_figure()
    fig.savefig(os.path.join(postdir, 'psi.png'))
