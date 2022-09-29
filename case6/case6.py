#!/usr/bin/python3
"""CASE 6

Case 6 is a study of the effects of Reynolds number and alpha 
(stagnation thickness) on the saturation current.  The individual 
simulations are run with combinations of alpha and R.  The applied 
voltage is 15.
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
            gamma=10,
            phia = 15)

    s = s1d.Sheath1DV()
    for index,ip in ipm.items():
        print(index)
        s.init_param(ip)
        s.init_grid(.002)
        s.init_mat()
        s.init_solution()
        
        try:
            s.solve()
        
            p = s.init_post()
            p.expand_post()
            p.save(f'data/{index:03d}.gz', overwrite=True)
        except:
            print("ERROR!")
