# -*- eval: (comment-tags-mode) -*-
import numpy as np
import dill as pickle
import csv
import sys
import time as tm
import matplotlib.pyplot as pypl
from matplotlib import rcParams

from BootStrap3 import BootStrap
from evxptreaders import ReadEvxptdump
import params as pr
import fitfunc as ff
import stats as stats
from formatting import err_brackets

def read_data(pars):
    """Read data files in pickle format with Bootstrap ensembles"""
    BSdata_dir = pars.workdir+'pandapickle/'+pars.momfold[pars.momentum][:7]+'/data_'+str(pars.nboot)+'/'
    qval = np.array2string(pars.qval[pars.momentum],separator='')[1:-1]
    bsfile  = BSdata_dir + 'TwoptBS_q' + qval + '.pkl'
    bsfile1 = BSdata_dir + 'TwoptBS_q' + qval + '_quark1_' + pars.lmbstring + '.pkl'
    bsfile2 = BSdata_dir + 'TwoptBS_q' + qval + '_quark2_' + pars.lmbstring +'.pkl'

    with open(bsfile, 'rb') as fileout:
        nucleon = pickle.load(fileout)
    with open(bsfile1, 'rb') as fileout:
        nucleon_q1  = pickle.load(fileout)
    with open(bsfile2, 'rb') as fileout:
        nucleon_q2 = pickle.load(fileout)
    nucleons_raw = [nucleon,nucleon_q1,nucleon_q2]

    nucleons = [[],[],[]]
    for num in range(len(nucleons_raw[0])):
        for type,nu in enumerate(nucleons):
            nu.append([])
            for t in range(len(nucleons_raw[type][num])):
                nu[num].append(BootStrap(len(nucleons_raw[type][num][t]), pars.confidence))
                nu[num][-1].values=nucleons_raw[type][num][t]
                nu[num][-1].Stats()

    return nucleons

def oneexpfitter(ratios, unpert_2p, fitfnc, pars, opnum, plots=1):
    """
    Loop over the the tmin values and fit the two-point functions with a one-exponential function to obtain the energy shift
    pars:  param class object
    """
    print('\n',pars.operators[opnum])
    op        = pars.operators[opnum]
    fitlist1  = [[],[]]
    energyfit = []

    #-------------------------------------------------------------------------------------------------
    # Set the initial guess values from effective energies and amplitudes
    # For the nucleon energy
    effmass0 = effmass(pars, unpert_2p, plot=False)
    effamp0  = effamp(pars, unpert_2p, plot=False)
    # Set the priors by taking the value of the effective amp. and mass at t=timeslice (hoerz2020)
    timeslice=12
    pars.initpar0[pars.momentum] = np.array([effamp0[timeslice].Avg, effmass0[timeslice].Avg])
    #pars.bounds0[pars.momentum] = [(-np.inf,np.inf),(-1.,effmass[timeslice].Avg*2)]
    # For the FH ratio
    effmass1 = effmass(pars, ratios[0], plot=False, FH=True)
    effamp1  = effamp(pars, ratios[0], plot=False, FH=True)
    # Set the priors by taking the value of the effective amp. and mass at t=timeslice with the width being 10 times the uncertainty of this point. (hoerz2020)
    timeslice=12
    pars.initpar1[pars.momentum] = np.array([effamp1[timeslice].Avg, effmass1[timeslice].Avg])
    
    #-------------------------------------------------------------------------------------------------
    #Loop over the tmin values for the fit
    dictenergy = {}
    dictenergyshift = [{},{}]
    fitcounter = 0
    for index0, tmax in enumerate(np.arange(pars.tmaxmin, pars.tmaxmax)):
        for index, tmin in enumerate(np.arange(pars.tminmin, pars.tminmax)):
            print("\n------------------------------", "\nfit range = ",tmin,"-",tmax)
            if (tmax-tmin)>fitfnc.npar+1: #Check whether there are more data points than fit parameters
                #---------------------------------------------
                #INFO: Fit to the 2pt. function to get a value for the energy
                extension = 7 # The correlator itself can be fit to later tmax values as the StN is better, so we increase the tmax values for this fit
                tdataE = np.arange(tmin,tmax+extension)
                fitparam,BSen0,redchisq = stats.BootFitclass3(fitfnc.eval, pars.initpar0[pars.momentum], tdataE, np.array(unpert_2p)[tdataE], bounds=pars.bounds2pt[:fitfnc.npar], fullcov=True)
                # Save the fit results in a dictionary
                dictenergy[fitcounter] = {}
                dictenergy[fitcounter]['tmin'] = tmin
                dictenergy[fitcounter]['tmax'] = tmax+extension
                dictenergy[fitcounter]['A0'] = BSen0[0].values
                dictenergy[fitcounter]['E0'] = BSen0[1].values
                dictenergy[fitcounter]['dof'] = len(tdataE)-len(pars.initpar0[pars.momentum])
                dictenergy[fitcounter]['chisq'] = redchisq*(len(tdataE)-len(pars.initpar0[pars.momentum]))
                energyfit.append(BSen0)
                #---------------------------------------------
                #INFO: Fit to the ratio of correlators to get the energy shift
                tdata = np.arange(tmin,tmax)
                for quark in np.arange(2):
                    #INFO: Fit over all bootstraps for 1-exp
                    fitboot,BSen1,redchisq1=stats.BootFitclass3(fitfnc.eval, pars.initpar1[pars.momentum], tdata, np.array(ratios[quark])[tdata],bounds=fitfnc.bounds,time=True, fullcov=True)
                    dictenergyshift[quark][fitcounter] = {}
                    dictenergyshift[quark][fitcounter]['tmin'] = tmin
                    dictenergyshift[quark][fitcounter]['tmax'] = tmax
                    dictenergyshift[quark][fitcounter]['A0'] = BSen1[0].values
                    dictenergyshift[quark][fitcounter]['E0'] = BSen1[1].values
                    dictenergyshift[quark][fitcounter]['dof'] = len(tdata)-len(pars.initpar1[pars.momentum])
                    dictenergyshift[quark][fitcounter]['chisq'] = redchisq1*(len(tdata)-len(pars.initpar1[pars.momentum]))
                    # Save bootstrap objects of parameters for the plot
                    fitlist1[quark].append(BSen1)
                fitcounter+=1

    # Save the energy fit to the pickle file 
    energyfitfile = pars.data_dir + 'energyBS_' + op +'_'+ pars.momfold[pars.momentum][:7]+'_'+pars.fit + pars.snkfold[pars.sink] + '_' + pars.lmbstring + '.pkl'
    with open(energyfitfile, 'wb') as fileout:
        pickle.dump(dictenergy, fileout)

    # Save the energy shift fit to the pickle file
    energyshiftfiles = [ pars.data_dir + 'energyshiftfit_'+op+'_q'+str(quark+1) + pars.momfold[pars.momentum][:7]+'_'+pars.fit+pars.snkfold[pars.sink]+'_'+pars.lmbstring+'.pkl' for quark in range(2) ]
    for i, outfile in enumerate(energyshiftfiles):
        with open(outfile, 'wb') as fileout:
            pickle.dump(dictenergyshift[i], fileout)

    #-----------------------------------------------------------------------------
    if plots==1: #Make only one version of the plots which shows the fit chosen in the params file.
        index2 = min(dictenergy, key=(lambda x : abs(dictenergy[x]['chisq']-1) ) )
        indexu = min(dictenergyshift[0], key=(lambda x : abs(dictenergyshift[0][x]['chisq']-1) ) )
        indexd = min(dictenergyshift[1], key=(lambda x : abs(dictenergyshift[1][x]['chisq']-1) ) )
        # Plot the effective energy + fit
        effmassplotter( unpert_2p
                        , np.arange(dictenergy[index2]['tmin'], dictenergy[index2]['tmax'])
                        , energyfit[index2]
                        , pars.plot_dir[0][0]+'../'
                        , 'unpert_'+op
                        , fitfnc
                        , dictenergy[index2]['chisq'])
        indices = [indexu,indexd]
        for quark in range(2):
            effdEplotter(pars
                         , ratios[quark]
                         , np.arange(dictenergyshift[quark][indices[quark]]['tmin'], dictenergyshift[quark][indices[quark]]['tmax'])
                         , fitlist1[quark][indices[quark]]
                         , pars.plot_dir[opnum][quark]
                         , fitfnc
                         , dictenergyshift[quark][indices[quark]]['chisq']
                         , filename = '_gamma'+op[1:]+'_quark'+str(quark+1)
                         , title    = r'Energy shift '+pars.momfold[pars.momentum][:7]+r', $\gamma_{'+op[1:]+r'}$, quark '+str(quark+1)
                         , ylabel   = r'$dE_{q_'+str(quark+1)+', \gamma_{'+op[1:]+'}}$'
                         , ylim     = pars.ylimde[opnum][quark][pars.momentum]
            )
    #-----------------------------------------------------------------------------
    return dictenergy, dictenergyshift


def plotratios(ratios1, ratios2, dictenergyshift1, dictenergyshift2, fitfnc, pars, opnum, quarknum):
    """ Plot the effective mass of the ratio of correlators for both lambdas and plot their fits """
    op = pars.operators[opnum]
    fold = pars.plot_dir[opnum][0]+'../'
    time       = np.arange(0,len(ratios1[quarknum]))
    efftime    = time[:-1]+0.5

    #lp05
    # Get the index of the fit with the chi-squared value closest to 1
    index = min(dictenergyshift1, key=(lambda x : abs(dictenergyshift1[x]['chisq']-1) ) )
    fitrange   = np.arange(dictenergyshift1[index]['tmin'], dictenergyshift1[index]['tmax'])
    redchisq   = dictenergyshift1[index]['chisq']
    fitBS    = np.array([ stats.effmass(fitfnc.eval(time[:pars.xlim], [dictenergyshift1[index]['A0'][nbo], dictenergyshift1[index]['E0'][nbo]] )) for nbo in range(len(dictenergyshift1[index]['A0'])) ])
    fitBSavg   = np.average(fitBS,axis=0)
    fitBSstd   = fitBS.std(axis=0, ddof=1)
    fitBSlower = fitBSavg - fitBSstd
    fitBSupper = fitBSavg + fitBSstd
    #lp025
    # Get the index of the fit with the chi-squared value closest to 1
    index2 = min(dictenergyshift2, key=(lambda x : abs(dictenergyshift2[x]['chisq']-1) ) )
    fitrange2   = np.arange(dictenergyshift2[index2]['tmin'], dictenergyshift2[index2]['tmax'])
    redchisq2   = dictenergyshift2[index2]['chisq']
    fitBS2   =np.array([stats.effmass(fitfnc.eval(time[:pars.xlim], [dictenergyshift2[index2]['A0'][nbo], dictenergyshift2[index2]['E0'][nbo]] )) for nbo in range(len(dictenergyshift2[index2]['A0'])) ])
    fitBSavg2   = np.average(fitBS2,axis=0)
    fitBSstd2   = fitBS2.std(axis=0, ddof=1)
    fitBSlower2 = fitBSavg2 - fitBSstd2
    fitBSupper2 = fitBSavg2 + fitBSstd2

    #lp05
    yavgeff1 = np.array([y.Avg for y in stats.effectivemass(ratios1[quarknum])])
    yerreff1 = np.array([y.Std for y in stats.effectivemass(ratios1[quarknum])])
    #lp025
    yavgeff2 = np.array([y.Avg for y in stats.effectivemass(ratios2[quarknum])])
    yerreff2 = np.array([y.Std for y in stats.effectivemass(ratios2[quarknum])])
    
    pypl.figure(figsize=(9,6))
    pypl.errorbar(efftime[:pars.xlim], yavgeff1[:pars.xlim], yerreff1[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=r'$\lambda=0.025$')
    pypl.errorbar(efftime[:pars.xlim], yavgeff2[:pars.xlim], yerreff2[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='b', marker=pars.markers[pars.sink], markerfacecolor='none', label=r'$\lambda=0.05$')

    pypl.plot(fitrange[:-1]+0.5,fitBSavg[fitrange[:-1]], linestyle='-', color=pars.colors[1], label="OneExp fit $\chi^2=${:0.2f}".format(redchisq))
    pypl.fill_between(fitrange[:-1]+0.5, fitBSlower[fitrange[:-1]], fitBSupper[fitrange[:-1]], color=pars.colors[1], alpha=0.3, linewidth=0)

    pypl.plot(fitrange2[:-1]+0.5,fitBSavg2[fitrange2[:-1]], linestyle='-', color=pars.colors[2], label="OneExp fit $\chi^2=${:0.2f}".format(redchisq2))
    pypl.fill_between(fitrange2[:-1]+0.5, fitBSlower2[fitrange2[:-1]], fitBSupper2[fitrange2[:-1]], color=pars.colors[2], alpha=0.3, linewidth=0)

    pypl.legend(fontsize='x-small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(r'$dE_{q_'+str(1)+', \gamma_{'+op[1:]+'}}$',labelpad=5,fontsize=18)
    # pypl.ylabel(ylabel,labelpad=5,fontsize=18)
    pypl.title(r'Energy shift '+pars.momfold[pars.momentum][:7]+r', $\gamma_{'+op[1:]+r'}$')
    # pypl.ylim(fitBSavg[-1]-15*(fitBSupper[-1]-fitBSlower[-1]), fitBSavg[-1]+15*(fitBSupper[-1]-fitBSlower[-1]))
    pypl.ylim(-0.02,0.18)
    pypl.legend(fontsize='small')
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'Eff_dEshift_' + pars.momfold[pars.momentum][:7] +  '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + '.pdf')
    pypl.close()

        # for quark in range(2):
        #     effdEplotter(pars
        #                  , ratios[quark]
        #                  , np.arange(dictenergyshift[quark][indices[quark]]['tmin'], dictenergyshift[quark][indices[quark]]['tmax'])
        #                  , fitlist1[quark][indices[quark]]
        #                  , pars.plot_dir[opnum][quark]
        #                  , fitfnc
        #                  , dictenergyshift[quark][indices[quark]]['chisq']
        #                  , filename = '_gamma'+op[1:]+'_quark'+str(quark+1)
        #                  , title    = r'Energy shift '+pars.momfold[pars.momentum][:7]+r', $\gamma_{'+op[1:]+r'}$, quark '+str(quark+1)
        #                  , ylabel   = r'$dE_{q_'+str(quark+1)+', \gamma_{'+op[1:]+'}}$'
        #                  , ylim     = pars.ylimde[opnum][quark][pars.momentum]
        #     )
    #-----------------------------------------------------------------------------
    return 



def writeBS(filename, headernames, bootstraps):
    """filename to be written to, headernames to print in the first row, bootstraps a list of bootstrap elements of the same length as headernames."""
    with open(filename,'w') as csvfile:
        datawrite = csv.writer(csvfile,delimiter=',',quotechar='|')
        datawrite.writerow(headernames)
        for i in range(bootstraps[-1].nboot):
            datawrite.writerow([bs.values[i] for bs in bootstraps])
    return

def writedata(filename, headernames, data):
    """filename to be written to, headernames to print in the first row, data of same length as headernames"""
    with open(filename,'w') as csvfile:
        datawrite = csv.writer(csvfile,delimiter=',',quotechar='|')
        datawrite.writerow(headernames)
        datawrite.writerow(data)
    return

def writefitdata(filename, headernames, data):
    """filename to be written to, headernames to print in the first row, data of same length as headernames"""
    with open(filename,'w') as csvfile:
        datawrite = csv.writer(csvfile,delimiter=',',quotechar='|')
        datawrite.writerow(headernames)
        for item in data:
            datawrite.writerow(item)
    return

def effmassplotter(ydata, fitrange, fitdata, fold, name, fitfnc, redchisq):
    """Plot the fit to the FH ratio and the data itself"""
    time    = np.arange(0,len(ydata))
    efftime = time[:-1]+0.5
    yavg    = np.array([y.Avg for y in ydata])
    yerr    = np.array([y.Std for y in ydata])
    yavgeff = np.array([y.Avg for y in stats.effectivemass(ydata)])
    yerreff = np.array([y.Std for y in stats.effectivemass(ydata)])

    fitBS1       = np.array([ stats.effmass(fitfnc.eval(time[:pars.xlim], [fd.values[nbo] for fd in fitdata])) for nbo in range(fitdata[0].nboot) ])
    fitBSavg1    = np.average(fitBS1,axis=0)
    fitBSstd1    = fitBS1.std(axis=0, ddof=1)
    fitBSlower1  = fitBSavg1 - fitBSstd1
    fitBShigher1 = fitBSavg1 + fitBSstd1
    #paramnames  = ['$dA_0$', '$dE_0$', '$dA_1$', '$dE_1$', '$dA_2$', '$dE_2$', '$dA_3$', '$dE_3$'][:len(fitdata)]
    paramnames  = ['$A_0$', '$E_0$', '$A_1$', '$E_1$', '$A_2$', '$E_2$', '$A_3$', '$E_3$'][:len(fitdata)]
    ratioavg    = np.array([fd.Avg for fd in fitdata])
    ratiostd    = np.array([fd.Std for fd in fitdata])
    print(f"{float(ratioavg[0])=}")
    print(f"{float(ratiostd[0])=}")
    print(f"{float(ratioavg[1])=}")
    print(f"{float(ratiostd[1])=}") 
    textstr = '\n'.join( [ paramnames[i]+' = '+err_brackets(float(ratioavg[i]), float(ratiostd[i]), form='Sci', texify=True) for i in range(len(ratioavg)) ] )
    props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='0.8')

    pypl.figure("effmassplot", figsize=(9,6))
    pypl.errorbar(efftime[:pars.xlim], yavgeff[:pars.xlim], yerreff[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])
    # print(f"{fitrange[:-1]=}")
    # print(f"{fitBSavg1=}")
    # print(f"{pars.xlim=}")
    # print(f"{fitBSavg1[fitrange[:-1]]=}")
    # print(f"{fitBSavg1=}")

    pypl.plot(fitrange[:-1]+.5, fitBSavg1[fitrange[:-1]], linestyle='-', color=pars.colors[1], label="correlator fit $\chi^2=${:0.2f}".format(redchisq))
    pypl.fill_between(efftime[:pars.xlim-1], fitBSlower1, fitBShigher1, color=pars.colors[1], alpha=0.3, linewidth=0)
    pypl.legend(fontsize='x-small',loc='upper left')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(r'$E_{\mathrm{eff}}$',labelpad=5,fontsize=18)
    pypl.title(r'Effective energy '+pars.momfold[pars.momentum][:7])
    pypl.ylim(bottom=0.1,top=2.0)
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'Eff_energy_' + pars.momfold[pars.momentum][:7] +  '_' + pars.fit + '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + name + pars.lmbstring +'.pdf')
    pypl.close()


def effdEplotter(pars, ydata, fitrange, fitdata, fold, fitfnc, redchisq, filename='', title=r'Energy shift', ylabel=r'dE', ylim=None):
    """Plotting the effective energy of the ratio with the fit and error ranges."""
    time       = np.arange(0,len(ydata))
    efftime    = time[:-1]+0.5
    fitBS      = np.array([ stats.effmass(fitfnc.eval(time[:pars.xlim], [fd.values[nbo] for fd in fitdata])) for nbo in range(fitdata[0].nboot) ])
    fitBSavg   = np.average(fitBS,axis=0)
    fitBSstd   = fitBS.std(axis=0, ddof=1)
    fitBSlower = fitBSavg - fitBSstd
    fitBSupper = fitBSavg + fitBSstd

    yavgeff = np.array([y.Avg for y in stats.effectivemass(ydata)])
    yerreff = np.array([y.Std for y in stats.effectivemass(ydata)])

    paramnames = ['$dA_0$', '$dE_0$', '$dA_1$', '$dE_1$', '$dA_2$', '$dE_2$', '$dA_3$', '$dE_3$'][:len(fitdata)]
    ratioavg   = np.array([fd.Avg for fd in fitdata])
    ratiostd   = np.array([fd.Std for fd in fitdata])
    textstr    = '\n'.join( [ paramnames[i]+' = '+err_brackets(float(ratioavg[i]), float(ratiostd[i]), form='Sci', texify=True) for i in range(len(ratioavg)) ] )
    props      = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='0.8')
    
    pypl.figure(figsize=(9,6))
    pypl.errorbar(efftime[:pars.xlim], yavgeff[:pars.xlim], yerreff[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])
    pypl.plot(fitrange[:-1]+0.5,fitBSavg[fitrange[:-1]], linestyle='-', color=pars.colors[1], label="OneExp fit $\chi^2=${:0.2f}".format(redchisq))
    pypl.fill_between(fitrange[:-1]+0.5, fitBSlower[fitrange[:-1]], fitBSupper[fitrange[:-1]], color=pars.colors[1], alpha=0.3, linewidth=0)
    pypl.legend(fontsize='x-small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(ylabel,labelpad=5,fontsize=18)
    pypl.title(title)
    # ax1.set_ylim((lower+upper)/2-3*(upper-lower), (lower+upper)/2+3*(upper-lower))
    pypl.ylim(fitBSavg[-1]-15*(fitBSupper[-1]-fitBSlower[-1]), fitBSavg[-1]+15*(fitBSupper[-1]-fitBSlower[-1]))
    # pypl.ylim(ylim)
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'Eff_dEshift_' + pars.momfold[pars.momentum][:7] +  '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + filename + pars.lmbstring+ '.pdf')
    pypl.close()
    #---------------------------------------------
    
def dEplotter(pars, ydata, fitrange, fitdata, fold, fitfnc, redchisq, fitparamfullcov, filename='', title=r'Energy shift', ylabel=r'dE', ylim=None):
    """Plotting the the ratio with the fit and error ranges."""
    time       = np.arange(0,len(ydata))
    fitBS      = np.array([ fitfnc.eval(time[:pars.xlim], [fd.values[nbo] for fd in fitdata]) for nbo in range(fitdata[0].nboot) ])
    fitBSavg   = np.average(fitBS,axis=0)
    fitBSstd   = fitBS.std(axis=0, ddof=1)
    fitBSlower = fitBSavg - fitBSstd
    fitBSupper = fitBSavg + fitBSstd
    fullcovfit = fitfnc.eval(fitrange,fitparamfullcov)
    yavg       = np.array([y.Avg for y in ydata])
    yerr       = np.array([y.Std for y in ydata])

    paramnames = ['$dA_0$', '$dE_0$', '$dA_1$', '$dE_1$', '$dA_2$', '$dE_2$', '$dA_3$', '$dE_3$'][:len(fitdata)]
    ratioavg   = np.array([fd.Avg for fd in fitdata])
    ratiostd   = np.array([fd.Std for fd in fitdata])
    textstr    = '\n'.join( [ paramnames[i]+' = '+err_brackets(float(ratioavg[i]), float(ratiostd[i]), form='Sci', texify=True) for i in range(len(ratioavg)) ] )
    props      = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='0.8')
    
    pypl.figure(figsize=(9,6))
    pypl.errorbar(time[:pars.xlim], yavg[:pars.xlim], yerr[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])
    pypl.plot(fitrange,fitBSavg[fitrange], linestyle='-', color=pars.colors[1], label="OneExp fit $\chi^2=${:0.2f}".format(redchisq))
    pypl.fill_between(fitrange, fitBSlower[fitrange], fitBSupper[fitrange], color=pars.colors[1], alpha=0.3, linewidth=0)
    pypl.plot(fitrange,fullcovfit, linestyle='-', linewidth=0.3, color=pars.colors[3], label="$\chi^2=${:0.2f}".format(redchisq))
    pypl.legend(fontsize='x-small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(ylabel,labelpad=5,fontsize=18)
    pypl.title(title)
    pypl.ylim(ylim)
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'dEshift_' + pars.momfold[pars.momentum][:7] +  '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + filename + pars.lmbstring +'.pdf')
    pypl.close()
    #---------------------------------------------
    
def effmass(pars, ydata, plot=False, FH=False):
    """Return the effective mass and plot it if plot==True"""
    effmass = stats.effectivemass(ydata)
    if plot:
        time    = np.arange(0,len(ydata))
        efftime = time[:-1]+0.5
        yavgeff = np.array([y.Avg for y in effmass])
        yerreff = np.array([y.Std for y in effmass])
        pypl.figure("effmassplot", figsize=(9,6))
        pypl.errorbar(efftime[:pars.xlim], yavgeff[:pars.xlim], yerreff[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])
        pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
        pypl.ylabel(r'$E_{\mathrm{eff}}$',labelpad=5,fontsize=18)
        pypl.title(r'Effective mass')
        # pypl.ylim(bottom=pars.ylimeffmass[0],top=pars.ylimeffmass[1])
        timeslice=12
        pypl.ylim(yavgeff[timeslice]-20*yerreff[timeslice], yavgeff[timeslice]+10*yerreff[timeslice])
        pypl.grid(True, alpha=0.4)
        ax = pypl.gca()
        pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
        #pypl.savefig(pars.plot_dir + 'Eff_energy_' + pars.momfold[pars.momentum][:7] +  '_'+ pars.snkfold[pars.sink] + '.pdf')
        if FH:
            pypl.savefig(pars.plot_dir[0][0] + '/Eff_energy_' + pars.momfold[pars.momentum][:7] + pars.lmbstring + '.pdf')
        else:
            pypl.savefig(pars.plot_dir[0][0] + '../Eff_energy_' + pars.momfold[pars.momentum][:7] + pars.lmbstring + '.pdf')
        pypl.close()
    return effmass

def effamp(pars, ydata, plot=False,timeslice=10, FH=False):
    """
    Return the effective amplitude and plot it if plot==True
    Idea comes from Hoerz2020 paper
    """
    effmass = stats.effectivemass(ydata)
    effamp =[]
    for i, meff in enumerate(effmass):
        effamp.append(ydata[i]*meff.exp((len(effamp)-1)))
        effamp[-1].Stats()
    if plot:
        time    = np.arange(0,len(ydata))
        efftime = time[:-1]+0.5
        yavg    = np.array([y.Avg for y in effamp])
        yerr    = np.array([y.Std for y in effamp])
        pypl.figure("effampplot", figsize=(9,6))
        pypl.errorbar(efftime[:pars.xlim], yavg[:pars.xlim], yerr[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])
        pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
        pypl.ylabel(r'$A_{\mathrm{eff}}$',labelpad=5,fontsize=18)
        pypl.title(r'Effective amplitude')
        #pypl.ylim(bottom=pars.ylimeffmass[0],top=pars.ylimeffmass[1])
        # pypl.ylim(-3000,-250)
        pypl.ylim(yavg[timeslice]-20*yerr[timeslice], yavg[timeslice]+10*yerr[timeslice])
        pypl.grid(True, alpha=0.4)
        ax = pypl.gca()
        pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
        if FH:
            pypl.savefig(pars.plot_dir[0][0] + '/Eff_amplitude_' + pars.momfold[pars.momentum][:7] + pars.lmbstring + '.pdf')
        else:
            pypl.savefig(pars.plot_dir[0][0] + '../Eff_amplitude_' + pars.momfold[pars.momentum][:7] + pars.lmbstring + '.pdf')
        # pypl.savefig(pars.plot_dir[0][0] + '../Eff_amplitude_' + pars.momfold[pars.momentum][:7] +  '_'+ pars.snkfold[pars.sink] + '.pdf')
        pypl.close()
    return effamp

def makeratio(nclns, opnum):
    """Construct the ratio of correlators"""
    ratiou = stats.feynhellratio(nclns[1][0], nclns[0][1], nclns[0][0], nclns[1][1])
    ratiod = stats.feynhellratio(nclns[2][0], nclns[0][1], nclns[0][0], nclns[2][1])
    # ratiou = stats.feynhellratioshort(nclns[1][0], nclns[0][0])
    # ratiod = stats.feynhellratioshort(nclns[2][0], nclns[0][0])
    ratios = [ratiou, ratiod]
    # Take the average of the (pos. parity, trev=0) and (neg parity, trev=1) two-point functions to get an energy value
    unpert_2p =[]
    for i in range(len(nclns[0][0])): #64
        unpert_2p.append(BootStrap(nclns[0][0][i].nboot, 68))
        unpert_2p[-1] = (nclns[0][0][i] + nclns[0][1][i])*0.5
        unpert_2p[-1].Stats()
    return ratios, unpert_2p


if __name__ == "__main__":
    if len(sys.argv)>3:
        print("Starting Oneexp")
        start = tm.time()

        pypl.rc('font', size=18, **{'family': 'sans-serif','serif': ['Computer Modern']})
        pypl.rc('text', usetex=True)
        rcParams.update({'figure.autolayout': True})
        
        momentum     = int(sys.argv[1])
        kappas       = int(sys.argv[2])
        sinktype=0
        pars = pr.params(kappas, sinktype, momentum)
        pars.fit = 'Aexp'
        pars.makeresultdir(plots=True)

        # Set the range of tmin values to use
        # pars.tminmin = 10 #5
        # pars.tminmax = 12 #19

        print("------------------------------",pars.momfold[pars.momentum][:7],"------------------------------")
        opnum=0
        fitfunction  = ff.initffncs(pars.fit) #Initialise the fitting function

        # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # for lmbstring in [pars.lmblist[-1]]:
        for lmbstring in pars.lmblist:
            pars.lmbstring = lmbstring
            print(pars.lmbstring)
            nucleon_data = read_data(pars)
            ratios, unpert_2p = makeratio(nucleon_data,opnum)
            dictenergy, dictenergyshift = oneexpfitter(ratios, unpert_2p, fitfunction, pars, opnum, plots=0)
        
        # pars.lmbstring = 'lp001'
        # print(pars.lmbstring)
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp001 = read_data(pars)
        # ratioslp001, unpert_2plp001 = makeratio(nucleon_datalp001,opnum)
        # dictenergylp001, dictenergyshiftlp001 = oneexpfitter(ratioslp001, unpert_2plp001, fitfunction, pars, opnum, plots=0)

        # pars.lmbstring = 'lp01'
        # print(pars.lmbstring)
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp01 = read_data(pars)
        # ratioslp01, unpert_2plp01 = makeratio(nucleon_datalp01,opnum)
        # dictenergylp01, dictenergyshiftlp01 = oneexpfitter(ratioslp01, unpert_2plp01, fitfunction, pars, opnum, plots=0)

        # pars.lmbstring = 'lp02'
        # print(pars.lmbstring)
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp02 = read_data(pars)
        # ratioslp02, unpert_2plp02 = makeratio(nucleon_datalp02,opnum)
        # dictenergylp02, dictenergyshiftlp02 = oneexpfitter(ratioslp02, unpert_2plp02, fitfunction, pars, opnum, plots=0)
        
        # pars.lmbstring = 'lp04'
        # print(pars.lmbstring)
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp04 = read_data(pars)
        # ratioslp04, unpert_2plp04 = makeratio(nucleon_datalp04,opnum)
        # dictenergylp04, dictenergyshiftlp04 = oneexpfitter(ratioslp04, unpert_2plp04, fitfunction, pars, opnum, plots=0)
        
        # pars.lmbstring = 'lp08'
        # print(pars.lmbstring)
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp08 = read_data(pars)
        # ratioslp08, unpert_2plp08 = makeratio(nucleon_datalp08,opnum)
        # dictenergylp08, dictenergyshiftlp08 = oneexpfitter(ratioslp08, unpert_2plp08, fitfunction, pars, opnum, plots=0)
        
        # Function which plots results from both lambdas on the same plot, for
        # quarknum = 0 #up quark
        # plotratios(ratioslp025, ratioslp01, dictenergyshiftlp025[quarknum], dictenergyshiftlp01[quarknum], fitfunction, pars, opnum, quarknum)
        
        print('script time: \t', tm.time()-start)
    else:
        print("arg[1] = momentum")
        print("arg[2] = kappas")
        print("arg[3] = sinktype")
