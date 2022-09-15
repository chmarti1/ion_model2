#!/usr/bin/python3
"""CASE 5

Case 5 is a saturation current study with various values of beta and 
omega
"""
import sheath1d as s1d
import os

if __name__ == '__main__':
    
    # Generate a list of R and alpha value combinations
    Wblist = [.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    blist = [0.1, 0.5, 1, 5]
    
    beta = [0.]
    omega = [0.]
    for wb in Wblist:
        omega += [wb / b for b in blist]
        beta += blist
    
    ipm = s1d.IonParamManager(\
            R=1.,
            alpha=1.,
            omega=omega,
            beta=beta,
            phia=40.,
            gamma=10)

    s = s1d.Sheath1DV()
    first=True
    for index,ip in ipm.items():
        print(index)
        s.init_param(ip)
        s.init_grid(.002)
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
        
