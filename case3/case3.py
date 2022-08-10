#!/usr/bin/python3
"""CASE 3

Case 3 is a study of i-v characteristic with different values of omega.
R = 1, alpha = 1, beta = 1
"""
import sheath1d as s1d
import os

if __name__ == '__main__':
    
    # Generate a list of R and alpha value combinations
    Wlist = [0, .5, 1, 2]
    plist = [0, 0.001, 0.002, 0.003, .004, .005, .006]
    
    psia = []
    omega = []
    for p in plist:
        omega += Wlist
        psia += [p]*len(Wlist)
    
    ipm = s1d.IonParamManager(\
            R=1.,
            alpha=1.,
            omega=omega,
            beta=1.,
            psia=psia,
            gamma=10)

    s = s1d.Sheath1D()
    first=True
    for index,ip in ipm.items():
        print(index)
        s.init_param(ip)
        s.init_grid(.001)
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
        
