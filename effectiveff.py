#!/bin/python
import numpy as np
import pandas as pd
import csv
import sys
import time as tm
import matplotlib.pyplot as pypl
from matplotlib import rcParams
import os

from BootStrap3 import BootStrap
from evxptreaders import ReadEvxptdump
import params as pr
import fitfunc as ff
import stats as stats
from formatting import err_brackets

def read_data(pars):
    """Read data files in pickle format with Bootstrap ensembles"""
    BSdata_dir = pars.workdir+'pandapickle/'+pars.momfold[pars.momentum][:7]+'/data_'+str(pars.nboot)+'/'
    bsfiles    = [ [BSdata_dir + 'TwoptBS_' + pars.momfold[pars.momentum][:7] + '_num' + str(i) + '.pkl' for i in range(pars.numbers)],
                   [BSdata_dir + 'TwoptBS_' + pars.momfold[pars.momentum][:7] + '_quark1' + '_num'+str(i)+'.pkl' for i in range(pars.numbers)],
                   [BSdata_dir + 'TwoptBS_' + pars.momfold[pars.momentum][:7] + '_quark2' + '_num'+str(i)+'.pkl' for i in range(pars.numbers)] ]
    nucleon,nucleon_q1,nucleon_q2 = [],[],[] #TODO: Maybe don't even need the seperate lists, could just use one list?
    nucleons = [nucleon,nucleon_q1,nucleon_q2]
    for i in range(pars.numbers):
        for j,nu in enumerate(nucleons):
            rdata = pd.read_pickle(bsfiles[j][i]) #Unpickle
            nu.append([])
            for t in range(len(rdata.index)):
                nu[i].append(BootStrap(len(rdata.columns), pars.confidence))
                nu[i][-1].values=rdata.loc[t,:]
                nu[i][-1].Stats()
    return nucleons

def read_datag3_ratio(pars, sgn=-1.0):
    """Read data files in pickle format with Bootstrap ensembles for the data using the gamma3 operator
    Output the ratio of correlators and the unperturbed correlator"""
    BSdata_dir = pars.workdir+'pandapickle/'+pars.momfold[pars.momentum][:7]+'/data_'+str(pars.nboot)+'/'
    bsfiles    = [ [BSdata_dir + 'TwoptBS_g3_' + pars.momfold[pars.momentum][:7] + '_quark1' + '_num'+str(i)+'.pkl' for i in range(16)],
                   [BSdata_dir + 'TwoptBS_g3_' + pars.momfold[pars.momentum][:7] + '_quark2' + '_num'+str(i)+'.pkl' for i in range(16)] ]
    
    nucleon_q1,nucleon_q2 = [],[] #TODO: Maybe don't even need the seperate lists, could just use one list?
    nucleons = [nucleon_q1,nucleon_q2]

    for i in range(16):
        for j,nu in enumerate(nucleons):
            rdata = pd.read_pickle(bsfiles[j][i]) #Unpickle
            nu.append([])
            for t in range(len(rdata.index)):
                nu[i].append(BootStrap(len(rdata.columns), pars.confidence))
                nu[i][-1].values=rdata.loc[t,:]
                if i%2:
                    nu[i][-1] = sgn*nu[i][-1] #Necessary to give the energy shift the same sign as the one from gamma2
                nu[i][-1].Stats()
            # print(i)
            # print(f"{nu[i][2].Avg=}")

    return nucleons

def ratioplotter(pars, opnum, fitfnc, momlist):
    # fold = '/home/mischa/Documents/PhD/analysis_results/effectiveFF/'+pars.kappa+'/'
    # if not os.path.isdir(fold):
    #         os.makedirs(fold)
    lmb=1e-4
    
    energylist=[]
    ratiolist = [[],[]]
    #for imom, nclns in enumerate(nucleonlist):
    for i, imom in enumerate(momlist):
        print(f"{imom=}")
        pars.momentum = imom
        if pars.momentum==0:
            pars.numbers=2
            if opnum==0:
                continue
        else:
            pars.numbers=10

        # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        nucleon_data = read_data(pars)
        ratios, unpert_2p = makeratio(nucleon_data,opnum)
        # If kappa=kp119930kp119930 then also read the gamma3 files
        if pars.kappas==11 and pars.momentum!=0:
            nucleon_datag3 = read_datag3_ratio(pars,sgn=-1.0)
            ratiosg3, unpert_2pg3 = makeratiog3(nucleon_datag3                # effmass(pars, ratiosg3[0], plot=True)
            # effmass(pars, ratiosg3[1], plot=True)
)
            ratios, unpert_2p = combine_ratios(ratios, ratiosg3, unpert_2p, unpert_2pg3)            # # Take the average of the (pos. parity, trev=0) and (neg parity, trev=1) two-point functions to get an energy value
        ratiolist[0].append(ratios[0])
        ratiolist[1].append(ratios[1])
    dEplotter(pars, energylist, ratiolist[0], momlist, lmb, opnum, 0, fold=pars.plot_dir[opnum][0])
    dEplotter(pars, energylist, ratiolist[1], momlist, lmb, opnum, 1, fold=pars.plot_dir[opnum][0])
    # FFplotter(pars, energylist, ratiolist[quark], momlist, lmb, opnum, quark, fold)
            

def dEplotter(pars, energylist, ratiolist, momlist, lmb, opnum, quark, fold):
    pypl.figure(figsize=(9,6))
    for imom, ratio in zip(momlist,ratiolist):
        ydata   = ratio
        time    = np.arange(0,len(ydata))
        efftime = time[:-1]+0.5
        if opnum==1 and imom==0 and pars.lattice != 'Feyn-Hell_kp121040kp120620/' and pars.lattice != 'Feyn-Hell_kp119930kp119930/':
            effratio = stats.effectivemass(ydata, factor=0.5*lmb**(-1))
            yavgeff = np.array([y.Avg for y in effratio])
            yerreff = np.array([y.Std for y in effratio])
        # elif opnum==1 and imom==0 and pars.lattice != 'Feyn-Hell_kp119930kp119930/':
        #     print('here')
        #     effratio = stats.effectivemass(ydata, factor=lmb**(-1))
        #     yavgeff = np.array([y.Avg for y in effratio])
        #     yerreff = np.array([y.Std for y in effratio])
        else:
            effratio = stats.effectivemass(ydata, factor=lmb**(-1))
            yavgeff = np.array([y.Avg for y in effratio])
            yerreff = np.array([y.Std for y in effratio])

        pypl.errorbar(efftime[:pars.xlim]+imom/10, yavgeff[:pars.xlim], yerreff[:pars.xlim], fmt='.', capsize=2, elinewidth=1, color=pars.colors[imom], marker=',', markerfacecolor='none', label=pars.snkfold[pars.sink]+' q='+str(pars.qval[imom]))
    
    op = pars.operators[opnum]
    filename = '_gamma'+op[-1]+'_quark'+str(quark+1)
    title    = r'Energy shift '+', $\gamma_{'+op[-1]+'}$, quark '+str(quark+1)
    #ylabel   = r'$\Delta E_{q_'+str(quark+1)+', \gamma_{'+op[-1]+'}}$'
    ylabel   = r'$\Delta E_{q_'+str(quark+1)+', \gamma_{'+op[-1]+'}}$'
    ylim     = pars.ylimde[opnum][quark][pars.momentum]
 
    pypl.legend(fontsize='xx-small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(ylabel,labelpad=5,fontsize=18)
    pypl.title(title)
    pypl.ylim([[(-0.2,0.4), (-0.1,0.05)],[(-0.2,1.2), (-0.2,0.7)]][opnum][quark])
    # pypl.ylim(ylim)
    pypl.xlim(0,17)
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    #pypl.savefig(fold + 'Eff_dEshift_' + pars.momfold[pars.momentum][:7] +  '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + filename + '.pdf')
    pypl.savefig(fold + 'Eff_dEshift_' +  pars.snkfold[pars.sink] + filename + '.pdf')
    pypl.close()

def FFplotter(pars, energylist, ratiolist, momlist, lmb, opnum, quark, fold):
    #xmax=[20,20,17,15,12]
    xmax=[20,20,20,20,20]
    pypl.figure(figsize=(9,6))
    for imom, ratio in zip(momlist,ratiolist):
        ydata   = ratio
        time    = np.arange(0,len(ydata))
        efftime = time[:-1]+0.5
        if opnum==1 and imom==0 and pars.lattice != 'Feyn-Hell_kp121040kp120620/' and pars.lattice != 'Feyn-Hell_kp119930kp119930/':
            effratio = stats.effectivemass(ydata, factor=0.5*lmb**(-1))
            for k in range(len(effratio)):
                effratio[k] = effratio[k]*energylist[imom]*energylist[0]**(-1)
                effratio[k].Stats()
            yavgeff = np.array([y.Avg for y in effratio])
            yerreff = np.array([y.Std for y in effratio])
        # elif opnum==1 and imom==0 and pars.lattice != 'Feyn-Hell_kp119930kp119930/':
        #     print('here')
        #     effratio = stats.effectivemass(ydata, factor=lmb**(-1))
        #     for k in range(len(effratio)):
        #         effratio[k] = effratio[k]*energylist[imom]*energylist[0]**(-1)
        #         effratio[k].Stats()
        #     yavgeff = np.array([y.Avg for y in effratio])
        #     yerreff = np.array([y.Std for y in effratio])
        else:
            effratio = stats.effectivemass(ydata, factor=lmb**(-1))
            for k in range(len(effratio)):
                effratio[k] = effratio[k]*energylist[imom]*energylist[0]**(-1)
                effratio[k].Stats()
            yavgeff = np.array([y.Avg for y in effratio])
            yerreff = np.array([y.Std for y in effratio])

        pypl.errorbar(efftime[:xmax[imom]]+imom/10, yavgeff[:xmax[imom]], yerreff[:xmax[imom]], fmt='.', capsize=2, elinewidth=1, color=pars.colors[imom], marker=',', markerfacecolor='none', label=pars.snkfold[pars.sink]+' q='+str(pars.qval[imom]))
    
    op = pars.operators[opnum]
    filename = '_gamma'+op[-1]+'_quark'+str(quark+1)
    title    = r'Electric form factor '+', $\gamma_{'+op[-1]+'}$, quark '+str(quark+1) + r'$(\kappa_{1}, \kappa_{2})=(0.'+pars.kappa[2:8]+',0.'+pars.kappa[10:]+')$'#+pars.geom 
    #ylabel   = r'$\Delta E_{q_'+str(quark+1)+', \gamma_{'+op[-1]+'}}$'
    ylabel   = r'$G_{E, q_'+str(quark+1)+', \gamma_{'+op[-1]+'}}$'
    ylim     = pars.ylimde[opnum][quark][pars.momentum]
 
    pypl.legend(fontsize='xx-small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(ylabel,labelpad=5,fontsize=18)
    pypl.title(title)
    pypl.ylim([(-0.2,1.2), (-0.2,0.7)][quark])
    pypl.xlim(0,pars.xlim)
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    #pypl.savefig(fold + 'Eff_dEshift_' + pars.momfold[pars.momentum][:7] +  '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + filename + '.pdf')
    pypl.savefig(fold + 'Eff_FF_' +  pars.snkfold[pars.sink] + filename + '.pdf')
    pypl.close()

def makeratio(nclns, opnum):
    """Construct the ratio of correlators"""
    # The ratio of perturbed and unperturbed two-point functions is different for the two operators
    if opnum==0: #g2
        ratiou = stats.feynhellratio4(nclns[0], nclns[1])
        ratiod = stats.feynhellratio4(nclns[0], nclns[2])
    elif opnum==1: #g4
        ratiou = stats.feynhellratio(nclns[1][0], nclns[0][1], nclns[0][0], nclns[1][1])
        ratiod = stats.feynhellratio(nclns[2][0], nclns[0][1], nclns[0][0], nclns[2][1])
    ratios = [ratiou, ratiod]

    # Take the average of the (pos. parity, trev=0) and (neg parity, trev=1) two-point functions to get an energy value
    unpert_2p =[]
    for i in range(len(nclns[0][0])): #64
        unpert_2p.append(BootStrap(nclns[0][0][i].nboot, 68))
        unpert_2p[-1] = (nclns[0][0][i] + nclns[0][1][i])*0.5
        unpert_2p[-1].Stats()
    return ratios, unpert_2p

def makeratiog3(nclns, sgn=-1.0):
    """Construct the ratio of correlators"""
    fullratio=[[],[]]
    for quark in range(2):
        # print(f"{quark=}")
        for i in range(len(nclns[quark][0])):
            # Take average over trev=0 and trev=1
            # Take sum or difference of unpolarized and polarized projections (spin up, spin down)
            spinupposm = 0.25*(nclns[quark][0][i]+nclns[quark][1][i]+nclns[quark][2][i]+nclns[quark][3][i])
            spindnposm = 0.25*(nclns[quark][0][i]-nclns[quark][1][i]+nclns[quark][2][i]-nclns[quark][3][i])
            spinupnegm = 0.25*(nclns[quark][4][i]+nclns[quark][5][i]+nclns[quark][6][i]+nclns[quark][7][i])
            spindnnegm = 0.25*(nclns[quark][4][i]-nclns[quark][5][i]+nclns[quark][6][i]-nclns[quark][7][i])
            # print(f"{i=}")

            # Take the sam combinations for the correlators with FH perturbations
            spinupposmlmb = 0.25*(nclns[quark][8][i]+nclns[quark][9][i]+nclns[quark][10][i]+nclns[quark][11][i])
            spindnposmlmb = 0.25*(nclns[quark][8][i]-nclns[quark][9][i]+nclns[quark][10][i]-nclns[quark][11][i])
            spinupnegmlmb = 0.25*(nclns[quark][12][i]+nclns[quark][13][i]+nclns[quark][14][i]+nclns[quark][15][i])
            spindnnegmlmb = 0.25*(nclns[quark][12][i]-nclns[quark][13][i]+nclns[quark][14][i]-nclns[quark][15][i])
            # print('i1')

            # Construct the ratio of the various correlators defined above
            ratio1 = ((spinupposmlmb*spindnposm)*(spinupposm*spindnposmlmb)**(-1))**(1/4)
            # print('i2')
            # print(f"{spinupnegmlmb.Avg=}")
            # print(f"{spindnnegm.Avg=}")
            ratio2 = ((spinupnegm*spindnnegmlmb)*(spinupnegmlmb*spindnnegm)**(-1))**(1/4)
            # print('i3')
            fullratio[quark].append( ratio1*ratio2 )
            fullratio[quark][-1].Stats()
            # print('i4')
    # Take the average of the (pos. parity, trev=0) and (neg parity, trev=1) two-point functions to get an energy value
    unpert_2p =[]
    for i in range(len(nclns[0][0])): #64
        unpert_2p.append(BootStrap(nclns[0][0][i].nboot, 68))
        unpert_2p[-1] = (nclns[0][0][i] + nclns[0][2][i])*0.5
        unpert_2p[-1].Stats()
    return fullratio, unpert_2p

def combine_ratios(ratios, ratiosg3, unpert_2p, unpert_2pg3):
    """Combine the data from the gamma3 current and the gamma2 current"""
    fullratio=[[],[]]
    unpert_2pfull = []
    for q, quark in enumerate(pars.quarks): #2
        for i in range(len(ratios[q])): #64
            fullratio[q].append( 0.5*(ratios[q][i] + ratiosg3[q][i])) 
            # fullratio[q].append(BootStrap(nclns[0][0][i].nboot, 68))
            # fullratio[q][-1] = (ratios[q][i] + ratiosg3[q][i])*0.5
            fullratio[q][-1].Stats()
    for i in range(len(unpert_2p)): #64
        unpert_2pfull.append( 0.5*(unpert_2p[i] + unpert_2pg3[i])) 
        unpert_2pfull[-1].Stats()
    return fullratio, unpert_2pfull


if __name__ == "__main__":
    if len(sys.argv)>2:
        print("Starting effectiveff")
        start = tm.time()

        pypl.rc('font', size=18, **{'family': 'sans-serif','serif': ['Computer Modern']})
        pypl.rc('text', usetex=True)
        rcParams.update({'figure.autolayout': True})
        
        kappas     = int(sys.argv[1]) 
        sinktype   = int(sys.argv[2]) # ptsnk smsnk30 smsnk60
        pars       = pr.params(kappas, sinktype)
        # momentum   = int(sys.argv[1])
        # kappas     = int(sys.argv[2])
        # sinktype   = int(sys.argv[3]) # ptsnk smsnk30 smsnk60
        # pars = pr.params(kappas, sinktype, momentum)
        pars.fit = 'effectiveff'
        pars.makeresultdir(plots=True)

        fitfunction  = ff.initffncs('Aexp') #Initialise the fitting function

        for opnum in range(2):
            if opnum==0:
                momlist = np.arange(len(pars.momfold)-1)+1
            else:
                momlist = np.arange(len(pars.momfold))
            ratioplotter(pars, opnum, fitfunction, momlist)
    else:
        print("arg[1] = kappas")
        print("arg[2] = sinktype")
