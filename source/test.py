import sheath1d as s1d

def run():
    s = s1d.Sheath1D()
    s.init_param()
    s.init_grid(.01)
    s.init_mat()
    s.init_solution()
    while not s.test_solution():
        s.step_solution()
    
        
    p = s1d.PostIon1D(s)
    p.expand_post()

    return s,p
