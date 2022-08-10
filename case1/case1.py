#!/usr/bin/python3
"""CASE 1

Case 1 is a study of the effects of Reynolds number and alpha 
(stagnation thickness).  The individual simulations are run with 
combinations of alpha and R.  The applied electric field is zero.
"""
import sheath1d as s1d
import os

if __name__ == '__main__':
    
    # Generate a list of R and alpha value combinations
    Rlist = [.2, .5, 1., 2., 5., 10., 20.]
    alist = [0.2, 0.5, 1., 2.]
    
    alpha = []
    R = []
    for a in alist:
        R += Rlist
        alpha += [a]*len(Rlist)
    
    ipm = s1d.IonParamManager(\
            R=R,
            alpha=alpha,
            omega=0,
            beta=1,
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
