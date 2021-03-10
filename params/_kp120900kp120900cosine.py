import numpy as np

def kp120900kp120900cosine(self):
    ### -----kp120900kp120900-----
    self.lattice      = 'Feyn-Hell_kp120900kp120900/'
    self.kappa        = 'kp120900kp120900'
    self.geom         = '32x64'
    self.beta         = 'b5p50'
    self.csw          = 'c2p6500'
    self.L            = 32
    # self.lmbstring    = 'lp025'
    # self.lmblist      = ['lp0001', 'lp001', 'lp005', 'lp01', 'lp02', 'lp04', 'lp08']
    # self.lmbvals      = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08]
    # self.lmblist      = ['lp025', 'lp05']
    # self.lmbvals      = [0.025, 0.05]
    self.lmblist      = ['lp05']
    self.lmbvals      = [0.05]
    self.almb         = 2.5e-2
    self.a            = 0.074
    self.baryons      = {"proton" : ["p", [2/3, -1/3], [(-0.2,1.2), (-0.1,1.1), (-0.4,0.7), (-0.1,1.1), (-0.1,1.1)]], #Baryon type, electric charge, ylimits GE, GM, GE/GM, F1, F2
                       "neutron": ["n", [-1/3, 2/3], [(-0.2,0.2), (-1.1,0.1), (-0.7,0.1), (-0.2,0.2), (-0.9,0.1)]]} #Baryon type, electric charge, ylimits GE, GM, GE/GM, F1, F2
    self.quarks       = ['u','d']
    self.momfold      = ['mass/', 'p+1+0+0/q+0+0+0/', 'p+1+1+1/q+0+0+0/']
    # self.momfold      = ['p+0+0+0/q+0+0+0/', 'p+1+0+0/q+0+0+0/', 'p+1+1+0/q+0+0+0/', 'p+1+1+1/q+0+0+0/', 'p+2+0+0/q+0+0+0/', 'p+2+1+0/q+0+0+0/', 'p+2+1+1/q+0+0+0/']
    self.qval         = [np.array([0,0,0]), np.array([2,0,0]), np.array([2,2,2])]
    self.snkfold      = ['']
    self.sink         = 0
    self.xlim         = 32
    self.norma        = np.array([1e39])
    self.initpar0     = np.array([[1., 4.5e-1],
                                 [1., 5e-1],
                                 [1., 6e-1],
                                 [1., 6e-1],
                                 [1., 7.5e-1],
                                 [1., 7.5e-1],
                                 [1., 8.5e-1]])
    self.initpar1     = np.array([[1.,1.8e-4],
                                 [1.,1.8e-5],
                                 [1.,5.8e-5],
                                 [1.,5.8e-5],
                                 [1.,1.8e-6],
                                 [1.,1.8e-6],
                                 [1.,1.8e-6]])
    self.initpar2pt   = np.array([[-1., 4.5e-1, -1., 8.0e-1]])
    self.initparratio = np.array([[[[1.0e-1, 4.0e-4, 5.0e-1, 4.5e-4]],
                               [[1.0e-1, 2.3e-4, 5.0e-1, 2.4e-4]]]])


    self.bounds2pt = [(-np.inf,np.inf),(-1.,1.),(-np.inf,np.inf),(-1.,3.)]
    self.boundsratio  = [(-np.inf,np.inf),(-1e-1,1e-1),(-np.inf,np.inf),(-1e-1,1e-1)]
    self.tminenergy   = [0]
    self.tminchoice   = [[[ 0,  0,  0,  0,  0,  0,  0],  #g4 uquark
                         [ 0,  0,  0,  0,  0,  0,  0]]] #g4 dquark
    self.tminchoice1  = [[[ 8,  7,  9,  7,  6,  6,  6],  #g4 uquark
                         [ 8,  7,  9,  7,  6,  6,  6]]] #g4 dquark
    self.tmaxenergy   = [24]  #energy fit
    self.tmaxratio    = [[[17],  #g4 uquark
                          [17]]] #g4 dquark
    
    self.ylims  = [[[(0.98,1.05)],   #u quark  g4
                         [(0.98,1.05)]]]   #d quark  g4
    self.ylimde = [[[(3.7e-4, 5.7e-4)],  #u quark  g4
                         [(1.2e-4, 3.2e-4)]]]  #d quark g4

    self.tmin_quarks   = [[[[1,6]],  #u quark g4
                           [[1,6]]]] #d quark g4
    self.tmin_energy   = [[2,8], [3,7], [1,7], [1,4], [1,4], [1,3], [1,3]]
    self.tmin_quarks_1 = [[[[7,14], [7,16], [6,13], [6,12], [6,11], [6,10], [5,10]],  #u quark g4
                           [[7,13], [7,14], [6,13], [6,12], [6,11], [6,10], [4,10]]]] #d quark g4
    self.tmin_energy_1 = [[9,18], [12,18], [8,15], [7,12], [5,10], [4,10], [1,9]]
    
