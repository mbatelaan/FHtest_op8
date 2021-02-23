# -*- eval: (comment-tags-mode) -*-
import numpy as np
import os

class params:
    # beta=5.50
    # 32x64
    from ._kp120900kp120900 import kp120900kp120900
    
    def __init__(self, kappas, sinktype, momentum=None):
        """
        This class will set the fitting parameters of for each of the lattice ensembles used. It will import the specific parameters from the other files in this folder.
        To add a new ensemble, simply make a new params file, import it above and add it to the lattifefn list below.
        kappas: integer 0-10 (in the order of the import statements above)
        sinktype: integer 0-3 for ptsnk, smsnk30, smsnk60 (or other combination that is present in the files)
        momentum: integer value of the momentum in order of smallest to largest (starting with 0)
        """
        self.kappas     = kappas
        self.sinktype   = sinktype
        self.momentum   = momentum
        self.basedir    = '/home/mischa/Documents/PhD/analysis_results/FHtest/'
        self.nboot      = 500 #700
        self.nbin       = 1 #10
        self.confidence = 68
        self.xlim       = 32
        self.colors     = ['b', 'r', 'y', 'k', 'g', 'm']
        self.markers    = ['s','o','^','*', 'v', '>', '<']
        self.quarks     = ['u','d']
        self.operators  = ['g35']
        self.opchoice   = [0]
        self.tminmin     = 3
        self.tminmax     = 14
        self.tmaxmin     = 11
        self.tmaxmax     = 18
        self.ff_number   = 0
        self.datafiles = ['Ratio_fit', 'One-exp_fit', 'corr_fit', 'combined_corr_fit', 'wa_weightedavg']

        self.latticefn = [
            self.kp120900kp120900, #0
            # self.kp121040kp121040, #1
            # self.kp121040kp120620, #2
            # self.kp120620kp121040, #3
            # self.kp121095kp121095, #4
            # self.kp121095kp120512, #5
            # self.kp120512kp121095, #6
            # self.kp122005kp122005, #7
            # self.kp122130kp122130, #8
            # self.kp122130kp121756, #9
            # self.kp121756kp122130, #10
            # self.kp122078kp122078, #11
            # self.kp122078kp121859, #12
            # self.kp121859kp122078, #13
            # self.kp119930kp119930, #14
            # self.kp120084kp119623, #15
            # self.kp120084kp120084, #16
            # self.kp119623kp120084,  #17
            # self.kp122005kp122005lp001 #18
            ]
        self.latticefn[self.kappas]()

        if self.kappas==18: # So that this doesn't overwrite the lambda=1e-4 data.
            self.workdir    = self.basedir+self.beta+self.lattice[-17:-1]+self.csw+'/'+self.kappa+self.snkfold[0]+'lp001'+'/'  # Folder for the current lattice and sink type
        else:
            self.workdir    = self.basedir+self.beta+self.lattice[-17:-1]+self.csw+'/'+self.kappa+self.snkfold[0]+'/'  # Folder for the current lattice and sink type

        # Folders for pandapickle:
        # self.evxptdir     = '/home/mischa/Documents/PhD/lattice_results/FHtest2/'  # folder with evxpt resultdump files
        self.evxptdir     = '/home/mischa/Documents/PhD/lattice_results/Feyn-Hell_kp120900kp120900/clover_nf2p1_feyn-hell/b5p50kp120900kp120900c2p6500-32x64/nucleon/kp120900kp120900/mass/rel/FHtest2/'
        # self.evxptdir     = '/home/mischa/Documents/PhD/lattice_results/FHtest2/'  # folder with evxpt resultdump files
        self.sinkfold     = self.beta+self.lattice[-17:-1]+self.csw+self.snkfold[0]+'-'+self.geom+'/nucleon/'+self.kappa+'/'  # Location of the unperturbed correlators
        
        # These probably shouldn't be in the class
        self.fit        = "Aexp"
        self.fit2       = "TwoexpRatio4"
        self.numbers    = 10  #2 for mom=0, 10 for mom!=0

    def makeresultdir(self, plots=None):
        if self.momentum!=None:
            self.data_dir     = self.workdir+ self.fit +'/'+self.momfold[self.momentum][:7]+'/data_'+str(self.nboot)+'/'  # Folder where the fitting data will be saved
            self.fit_data_dir = self.workdir+self.fit +'/'+self.momfold[self.momentum][:7]+'/fit_data_'+str(self.nboot)+'/'  # Folder where the complete fitting data will be saved
            self.plot_dir     = [ [self.workdir+self.fit+'/'+self.momfold[self.momentum][:7]+'/plots/'+ op + '_quark1/',
                                   self.workdir+self.fit+'/'+self.momfold[self.momentum][:7]+'/plots/'+ op + '_quark2/'] for op in self.operators ]   # Folders where the plots will be saved
        else:
            self.data_dir     = self.workdir+ self.fit +'/data_'+str(self.nboot)+'/'  # Folder where the fitting data will be saved
            #self.fit_data_dir = self.workdir+self.fit +'/fit_data_'+str(self.nboot)+'/'  # Folder where the complete fitting data will be saved
            self.plot_dir     = [[self.workdir+self.fit+'/plots/'+ op + '/'] for op in self.operators ]   # Folders where the plots will be saved

        if self.fit=='FFplot':
            self.basedir0 = self.workdir+'Twoexp/'
            self.basedir1 = self.workdir+'Aexp/'
            self.fitratio    = "TwoexpRatio"
            self.fitoneexp   = "Aexp"
            self.choosetmins = True
            self.choosetmins_1 = True

        elif self.fit=='fanplots':
            self.savedir  = self.workdir+'fanplots/'
            self.ffplotdir = self.workdir+'FFplot/data_500/'
            self.plot_dir  = [[self.workdir+self.fit+'/plots/']]
            
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        if plots:
            try:
                if not os.path.isdir(self.fit_data_dir):
                    os.makedirs(self.fit_data_dir)
            except AttributeError:
                pass
            for lis in self.plot_dir:
                for fold in lis:
                    if not os.path.isdir(fold):
                        os.makedirs(fold)
                        if self.fit=='FFplot':
                            os.makedirs(fold+'/noavg')

        # # fanplots
        # self.kappavalues   = ['kp120900kp120900', 'kp121040kp121040', 'kp120512kp121095']
        # self.latticevalues = ['Feyn-Hell_'+i+'/' for i in self.kappavalues]
        # self.savedir       = self.basedir+'Fanplots/'+self.kappa +'/'
        # self.datadir       = self.workdir+'FFplot2/data/'
