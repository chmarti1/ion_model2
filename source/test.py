import sheath1d as s1d

def run():
    s = s1d.Sheath1D()
    s.init_param(R=1, alpha=.25, psia=0.,gamma=10)
    s.init_grid(.001)
    s.init_mat()
    s.init_solution()
    while not s.test_solution():
        s.step_solution()
    
    
    p = s1d.PostIon1D(s)
    p.expand_post()

    return s,p
