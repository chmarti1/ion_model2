#!/usr/bin/python3
"""CASE 2

Case 2 is a study of the effects of secondary ion formation rate and the
formation rate length scale, beta.  Reynolds number is 1 and alpha is 1.
"""
import sheath1d as s1d
import os

if __name__ == '__main__':
    
    # Generate a list of R and alpha value combinations
    Wblist = [0.1, .2, .5, 1., 2., 5., 10.]
    blist = [0.1, 0.2, 0.5, 1., 2.]
    
    beta = []
    omega = []
    for b in blist:
        omega += [wb/b/b for wb in Wblist]
        beta += [b]*len(Wblist)
    
    ipm = s1d.IonParamManager(\
            R=1.,
            alpha=1.,
            omega=omega,
            beta=beta,
            gamma=10)

    s = s1d.Sheath1D()
    for index,ip in ipm.items():
        print(index)
        s.init_param(ip)
        s.init_grid(.001)
        s.init_mat()
        s.init_solution()
        
        s.solve()
        
        p = s.init_post()
        p.expand_post()
        p.save(f'data/{index:03d}.gz', overwrite=True)
