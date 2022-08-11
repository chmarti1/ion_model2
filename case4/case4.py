#!/usr/bin/python3
"""CASE 4

Case 4 is an i-v study with various values of omega
"""
import sheath1d as s1d
import os

if __name__ == '__main__':
    
    # Generate a list of R and alpha value combinations
    Wlist = [0, .5, 1, 2]
    plist = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
    
    phi = []
    psi = []
    omega = []
    for w in Wlist:
        omega += [w] * len(plist)
        phi += plist
    
    ipm = s1d.IonParamManager(\
            R=1.,
            alpha=1.,
            omega=omega,
            beta=1.,
            phia=phi,
            gamma=10)

    s = s1d.Sheath1DV()
    first=True
    for index,ip in ipm.items():
        print(index)
        s.init_param(ip)
        s.init_grid(.005)
        s.init_mat()
        if first:
            s.init_solution()
            first=False
        else:
            s.init_solution(eta=s.eta, nu=s.nu, psi=s.psi)
        
        try:
            s.solve()
            p = s.init_post()
            p.expand_post()
            p.save(f'data/{index:03d}.gz', overwrite=True)
        except:
            pass
        
