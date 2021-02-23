# -*- eval: (comment-tags-mode) -*-
#INFO: Plot the fits of the ratio of two-exponential functions to the ratio of FH correlators. The data is read from the output of ratiofit.py and the plots are saved under Twoexp/mom/plots/
import sys
from scipy.special import gamma, gammainc, gammaincc
import numpy as np
import pandas as pd
import matplotlib.pyplot as pypl
import csv
import copy
import time as tm
from matplotlib import rcParams

from BootStrap3 import BootStrap
from evxptreaders import ReadEvxptdump
import fitfunc as ff
import stats as stats
import params as pr
from formatting import err_brackets

def readfitdata(folder,error):
    try:
        with open(folder) as csvfile:
            dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
            rows = [x for x in dataread][1:]
            tlist = rows[0]
            nboot = len(rows[1])
            dA0 = BootStrap(nboot,68)
            dA0.values = np.array([float(i) for i in rows[1]])
            dA0.Stats()
            dE0 = BootStrap(nboot,68)
            dE0.values = np.array([float(i) for i in rows[2]])
            dE0.Stats()
            dA1 = BootStrap(nboot,68)
            dA1.values = np.array([float(i) for i in rows[3]])
            dA1.Stats()
            dE1 = BootStrap(nboot,68)
            dE1.values = np.array([float(i) for i in rows[4]])
            dE1.Stats()
            chisq = rows[5]
    except IOError:
        error=True
    return dA0,dE0,dA1,dE1,tlist,chisq,error

def readcsv(folder,E,E1,dEu,dEd,error,j):
    try:
        with open(folder) as csvfile:
            dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
            rows = [x for x in dataread][1:]
            nboot = len(rows)
            E[j].append(BootStrap(nboot,68))
            E[j][-1].values = np.array([float(row[0]) for row in rows])
            E[j][-1].Stats()
            dEu[j].append(BootStrap(nboot,68))
            dEu[j][-1].values = np.array([float(row[1]) for row in rows])
            dEu[j][-1].Stats()
            dEd[j].append(BootStrap(nboot,68))
            dEd[j][-1].values = np.array([float(row[2]) for row in rows])
            dEd[j][-1].Stats()
            if len(rows[0])==4: # Check for Excited state energy
                E1[j].append(BootStrap(nboot,68))
                E1[j][-1].values = np.array([float(row[3]) for row in rows])
                E1[j][-1].Stats()
    except IOError:
        error=True
        #print("file not found")
    return E, E1, dEu, dEd, error

def readcsv_chi(folder,error):
    try:
        with open(folder) as csvfile:
            dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
            rows = [x for x in dataread]
            #nboot = len(rows)
            chisq_E0 = float(rows[1][0])
            chisq_dEq1 = float(rows[1][1])
            chisq_dEq2 = float(rows[1][2])
            dof = float(rows[1][3])
    except IOError:
        error=True
        #print("file not found")
    return chisq_E0, chisq_dEq1, chisq_dEq2, dof, error


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
    return nucleons


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

def weights(dof, chisq, derrors):
    """Take a list of degrees of freedom and of chi-squared values and errors of the fit and return the weights for each fit"""
    pf =[]
    for d, chi in zip(dof, chisq):
        #pf.append(gammaincc(d/2,chi/2)/gamma(d/2))
        pf.append(gammaincc(d/2,chi/2))
    denominator = sum(np.array(pf)*np.array([d**(-2) for d in derrors]))
    weights=[]
    for p, de in zip(pf, derrors):
        weights.append(p*(de**(-2))/denominator)
    return weights


def effmassplotter(ydata, fitrange, fitdata, fold, name, fitfnc, redchisq):
    """Plot the fit to the FH ratio and the data itself"""
    time    = np.arange(0,len(ydata))
    efftime = time[:-1]+0.5
    yavg    = np.array([y.Avg for y in ydata])
    yerr    = np.array([y.Std for y in ydata])
    yavgeff = np.array([y.Avg for y in stats.effectivemass(ydata)])
    yerreff = np.array([y.Std for y in stats.effectivemass(ydata)])

    fitBS1       = np.array([ stats.effmass(fitfnc.eval(time[:pars.xlim+10], [fd.values[nbo] for fd in fitdata])) for nbo in range(fitdata[0].nboot) ])
    fitBSavg1    = np.average(fitBS1,axis=0)
    fitBSstd1    = fitBS1.std(axis=0, ddof=1)
    fitBSlower1  = fitBSavg1 - fitBSstd1
    fitBShigher1 = fitBSavg1 + fitBSstd1
    paramnames   = ['$A_0$', '$E_0$', '$A_1$', '$E_1$', '$A_2$', '$E_2$', '$A_3$', '$E_3$'][:len(fitdata)]
    ratioavg     = np.array([fd.Avg for fd in fitdata])
    ratiostd     = np.array([fd.Std for fd in fitdata])
    textstr      = '\n'.join( [ paramnames[i]+' = '+err_brackets(float(ratioavg[i]), float(ratiostd[i]), form='Sci', texify=True) for i in range(len(ratioavg)) ] )
    props        = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='0.8')

    pypl.figure("effmassplot", figsize=(7,5))
    pypl.errorbar(efftime[:pars.xlim+10], yavgeff[:pars.xlim+10], yerreff[:pars.xlim+10], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])

    pypl.plot(fitrange[:-1]+.5, fitBSavg1[fitrange[:-1]], linestyle='-', color=pars.colors[1], label=r'correlator fit $\chi^2/{{\mathrm{{dof}}}}=${:0.2f}'.format(redchisq))
    pypl.fill_between(efftime[:pars.xlim+10-1], fitBSlower1, fitBShigher1, color=pars.colors[1], alpha=0.3, linewidth=0)
    pypl.legend(fontsize='x-small',loc='upper left')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(r'$E_{\mathrm{eff}}$',labelpad=5,fontsize=18)
    pypl.title(r'Effective energy '+pars.momfold[pars.momentum][:7])
    pypl.ylim(bottom=0,top=2.0)
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'Eff_energy_' + pars.momfold[pars.momentum][:7] +  '_' + pars.fit + '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + name + '.pdf')
    pypl.close()

def effmassplotter2(ydata, fitrange, fitdata, fold, name, fitfnc, redchisq):
    """Plot the fit to the FH ratio and the data itself also plot the fit & data with the excited state subtracted"""
    time    = np.arange(0,len(ydata))
    efftime = time[:-1]+0.5
    yavg    = np.array([y.Avg for y in ydata])
    yerr    = np.array([y.Std for y in ydata])
    yavgeff = np.array([y.Avg for y in stats.effectivemass(ydata)])
    yerreff = np.array([y.Std for y in stats.effectivemass(ydata)])

    fitfnc1exp  = ff.initffncs("Aexp")
    yvalues = np.array([y.values for y in ydata])
    yground = yvalues - np.array([ fitfnc1exp.eval(time, [fd.values[nbo] for fd in fitdata[2:]]) for nbo in range(fitdata[0].nboot) ]).T
    ygroundeff = np.array([ stats.effmass(yground[:,nbo]) for nbo in range(len(yground[0,:]))  ])
    ygroundeffavg = np.average(ygroundeff, axis=0)
    ygroundefferr = np.std(ygroundeff, axis=0, ddof=1)
    
    fitBS1       = np.array([ stats.effmass(fitfnc.eval(time[:pars.xlim+10], [fd.values[nbo] for fd in fitdata])) for nbo in range(fitdata[0].nboot) ])
    fitBSavg1    = np.average(fitBS1,axis=0)
    fitBSstd1    = np.std(fitBS1,axis=0, ddof=1)
    fitBSlower1  = fitBSavg1 - fitBSstd1
    fitBShigher1 = fitBSavg1 + fitBSstd1

    groundBSavg    = fitdata[1].Avg
    groundBSstd    = fitdata[1].Std
    groundBSlower  = groundBSavg - groundBSstd
    groundBShigher = groundBSavg + groundBSstd
    
    paramnames  = ['$A_0$', '$E_0$', '$A_1$', '$E_1$', '$A_2$', '$E_2$', '$A_3$', '$E_3$'][:len(fitdata)]
    ratioavg    = np.array([fd.Avg for fd in fitdata])
    ratiostd    = np.array([fd.Std for fd in fitdata])

    textstr = '\n'.join( [ paramnames[i]+' = '+err_brackets(float(ratioavg[i]), float(ratiostd[i]), form='Sci', texify=True) for i in range(len(ratioavg)) ] )
    pypl.figure("effmassplot", figsize=(9,6))
    pypl.errorbar(efftime[:pars.xlim+10], yavgeff[:pars.xlim+10], yerreff[:pars.xlim+10], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink], alpha=0.2)
    pypl.errorbar(efftime[:pars.xlim+10], ygroundeffavg[:pars.xlim+10], ygroundefferr[:pars.xlim+10], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink] +' ground state', alpha=1.0)
    pypl.plot(fitrange[:-1]+.5, fitBSavg1[fitrange[:-1]], linestyle='-', linewidth=0.5,  color=pars.colors[1], alpha=0.8)
    pypl.fill_between(efftime[:pars.xlim+10-1], fitBSlower1, fitBShigher1, color=pars.colors[1], alpha=0.2, linewidth=0, label=r'fit $\chi^2_{{\mathrm{{red}}}}=${:0.2f}'.format(redchisq))
    pypl.fill_between(efftime[:pars.xlim+10-1], groundBSlower, groundBShigher, color=pars.colors[2], alpha=0.4, linewidth=0, label=r'ground state energy $\chi^2=${:0.2f}'.format(redchisq))
    pypl.legend(fontsize='x-small',loc='upper left', framealpha=1.0)
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(r'$E_{\mathrm{eff}}$',labelpad=5,fontsize=18)
    pypl.title(r'Effective energy '+pars.momfold[pars.momentum][:7])
    #pypl.ylim(*[i for i in pars.ylimde[op][quark][pars.momentum] ])
    pypl.ylim(bottom=0,top=2.0)
    pypl.grid(True, alpha=0.4)
    props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='0.8')
    ax = pypl.gca()
    pypl.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'Eff_energy_' + pars.momfold[pars.momentum][:7] +  '_' + pars.fit + '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + name + '_2.pdf')
    pypl.close()

    
def dEplotter(pars, ydata, fitrange, fitdata, fold, fitfnc, redchisq, filename='', title=r'Energy shift', ylabel=r'dE', ylim=None, ratiofit=[]):
    """Plotting the the ratio with the fit and error ranges."""
    time       = np.arange(0,len(ydata))
    #fitBS2 = np.array([ fitfncratio.eval2(rangesq[quark][i], fitbootsratioq[quark][i][:,nbo], fitbootsratiop[quark][i][:,nbo]) for nbo in range(len(fitbootsratioq[quark][i][0][:])) ])**0.5
    if ratiofit:
        fitBS = np.array([ fitfnc.eval2(time[:pars.xlim], [q.values[nbo] for q in ratiofit], [p.values[nbo] for p in fitdata]) for nbo in range(fitdata[0].nboot) ])**0.5
        #fullcovfit = fitfncratio.eval2(fitrange,twoptfitpars,fitparamfullcov)**0.5
    else:
        fitBS = np.array([ fitfnc.eval(time[:pars.xlim], [fd.values[nbo] for fd in fitdata]) for nbo in range(fitdata[0].nboot) ])
        #fullcovfit = fitfnc.eval(fitrange,fitparamfullcov)**0.5
    
    fitBSavg   = np.average(fitBS,axis=0)
    fitBSstd   = fitBS.std(axis=0, ddof=1)
    fitBSlower = fitBSavg - fitBSstd
    fitBSupper = fitBSavg + fitBSstd
    
    yavg       = np.array([y.Avg for y in ydata])
    yerr       = np.array([y.Std for y in ydata])

    paramnames = ['$dA_0$', '$dE_0$', '$dA_1$', '$dE_1$', '$dA_2$', '$dE_2$', '$dA_3$', '$dE_3$'][:len(fitdata)]
    ratioavg   = np.array([fd.Avg for fd in fitdata])
    ratiostd   = np.array([fd.Std for fd in fitdata])
    textstr    = '\n'.join( [ paramnames[i]+' = '+err_brackets(float(ratioavg[i]), float(ratiostd[i]), form='Sci', texify=True) for i in range(len(ratioavg)) ] )
    props      = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='0.8')
    
    pypl.figure(figsize=(9,6))
    pypl.errorbar(time[:pars.xlim], yavg[:pars.xlim], yerr[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])
    pypl.plot(fitrange,fitBSavg[fitrange], linestyle='-', linewidth=0.3, color=pars.colors[1], label="Ratio fit $\chi^2=${:0.2f}".format(redchisq))
    pypl.fill_between(fitrange, fitBSlower[fitrange], fitBSupper[fitrange], color=pars.colors[1], alpha=0.3, linewidth=0)
    # pypl.plot(fitrange,fullcovfit, linestyle='-', linewidth=0.3, color=pars.colors[3], label="$\chi^2=${:0.2f}".format(redchisq))
    
    pypl.legend(fontsize='x-small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(ylabel,labelpad=5,fontsize=18)
    pypl.title(title)
    pypl.ylim(ylim)
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'dEshift_' + pars.momfold[pars.momentum][:7] +  '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + filename + '.pdf')
    pypl.close()
    #---------------------------------------------

def effdEplotter(pars, ydata, fitrange, fitdata, fold, fitfnc, redchisq, filename='', title=r'Energy shift', ylabel=r'dE', ylim=None, ratiofit=[]):
    """Plotting the effective energy of the ratio with the fit and error ranges."""
    # xlim = pars.xlim
    start = tm.time()
    xlim = fitrange[-1]+4
    time = np.arange(0,len(ydata))
    efftime = time[:-1]+0.5
    if ratiofit:
        fitBS = np.array([ stats.effmass(fitfnc.eval2(time[:xlim+1], [q.values[nbo] for q in ratiofit], [p.values[nbo] for p in fitdata]))/2 for nbo in range(fitdata[0].nboot) ])
    else:
        fitBS = np.array([ stats.effmass(fitfnc.eval(time[:xlim+1], [fd.values[nbo] for fd in fitdata])) for nbo in range(fitdata[0].nboot) ])
    fitBSavg   = np.average(fitBS,axis=0)
    fitBSstd   = fitBS.std(axis=0, ddof=1)
    fitBSlower = fitBSavg - fitBSstd
    fitBSupper = fitBSavg + fitBSstd

    yavgeff = np.array([y.Avg for y in stats.effectivemass(ydata)])
    yerreff = np.array([y.Std for y in stats.effectivemass(ydata)])
    # ylim = (yavgeff[fitrange[-3]]-2*yerreff[fitrange[-3]], yavgeff[fitrange[-3]]+2*yerreff[fitrange[-3]])
    # ylim = (fitBSavg[fitrange[-1]]-1.5*fitBSstd[fitrange[-1]], fitBSavg[fitrange[-1]]+1.5*fitBSstd[fitrange[-1]])
    
    paramnames = ['$dA_0$', '$dE_0$', '$dA_1$', '$dE_1$', '$dA_2$', '$dE_2$', '$dA_3$', '$dE_3$'][:len(fitdata)]
    ratioavg   = np.array([fd.Avg for fd in fitdata])
    ratiostd   = np.array([fd.Std for fd in fitdata])
    textstr    = '\n'.join( [ paramnames[i]+' = '+err_brackets(float(ratioavg[i]), float(ratiostd[i]), form='Sci', texify=True) for i in range(len(ratioavg)) ] )
    props      = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='0.8')

    start1 = tm.time()
    pypl.figure(figsize=(7,5))
    pypl.errorbar(efftime[:xlim], yavgeff[:xlim], yerreff[:xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])
    pypl.plot(fitrange[:-1]+0.5,fitBSavg[fitrange[:-1]], linestyle='-', color=pars.colors[1], label="Ratio fit $\chi^2=${:0.2f}".format(redchisq))
    pypl.fill_between(efftime[:xlim], fitBSlower, fitBSupper, color=pars.colors[1], alpha=0.3, linewidth=0)
    print('plotting time: \t', tm.time()-start1)

    start2 = tm.time()
    pypl.legend(fontsize='x-small', loc='upper left')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(ylabel,labelpad=5,fontsize=18)
    pypl.title(title)
    pypl.ylim(ylim)
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    print('settings time: \t', tm.time()-start2)

    start3 = tm.time()
    pypl.savefig(fold + 'Eff_dEshift_' + pars.momfold[pars.momentum][:7] +  '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + filename + '.pdf')
    pypl.close()
    print('saving time: \t', tm.time()-start3)
    
    print('effdEplotter time: \t', tm.time()-start)
    #---------------------------------------------

def effdEplotter_cheap(pars, ydata, fitrange, fitdata, fold, fitfnc, redchisq, filename='', title=r'Energy shift', ylabel=r'dE', ylim=None, ratiofit=[]):
    """Plotting the effective energy of the ratio with the fit and error ranges."""
    # xlim = pars.xlim
    start = tm.time()
    xlim = fitrange[-1]+4
    time = np.arange(0,len(ydata))
    efftime = time[:-1]+0.5
    if ratiofit:
        fitBS = np.array([ stats.effmass(fitfnc.eval2(time[:xlim+1], [q.values[nbo] for q in ratiofit], [p.values[nbo] for p in fitdata]))/2 for nbo in range(fitdata[0].nboot) ])
    else:
        fitBS = np.array([ stats.effmass(fitfnc.eval(time[:xlim+1], [fd.values[nbo] for fd in fitdata])) for nbo in range(fitdata[0].nboot) ])
    fitBSavg   = np.average(fitBS,axis=0)
    fitBSstd   = fitBS.std(axis=0, ddof=1)
    fitBSlower = fitBSavg - fitBSstd
    fitBSupper = fitBSavg + fitBSstd

    yavgeff = np.array([y.Avg for y in stats.effectivemass(ydata)])
    yerreff = np.array([y.Std for y in stats.effectivemass(ydata)])
    
    # paramnames = ['$dA_0$', '$dE_0$', '$dA_1$', '$dE_1$', '$dA_2$', '$dE_2$', '$dA_3$', '$dE_3$'][:len(fitdata)]
    ratioavg   = np.array([fd.Avg for fd in fitdata])
    ratiostd   = np.array([fd.Std for fd in fitdata])
    # textstr    = '\n'.join( [ paramnames[i]+' = '+err_brackets(float(ratioavg[i]), float(ratiostd[i]), form='Sci', texify=True) for i in range(len(ratioavg)) ] )
    # props      = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='0.8')

    pypl.figure(figsize=(7,6))
    # pypl.errorbar(efftime[:xlim], yavgeff[:xlim], yerreff[:xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=pars.snkfold[pars.sink])
    pypl.errorbar(efftime[:xlim], yavgeff[:xlim], yerreff[:xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none'
                  , label=r'$Q^2='+str(np.dot(pars.qval[pars.momentum],pars.qval[pars.momentum]))+'$')
    pypl.plot(fitrange[:-1]+0.5,fitBSavg[fitrange[:-1]], linestyle='-', color=pars.colors[1], label=r"$\chi^2_{{ \textrm{{dof}} }} ={:0.2f}$".format(redchisq))
    pypl.fill_between(efftime[:xlim], fitBSlower, fitBSupper, color=pars.colors[1], alpha=0.2, linewidth=0)

    # pypl.legend(fontsize='x-small', loc='upper left')
    pypl.legend(fontsize='small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(ylabel,labelpad=5,fontsize=18)
    pypl.title(title)
    pypl.ylim(ylim)
    pypl.axhline(0, color='k')

    start3 = tm.time()
    pypl.savefig(fold + 'Eff_dEshift_q'+np.array2string(pars.qval[pars.momentum],separator='')[1:-1] + '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + filename + '.pdf')
    # pypl.savefig(fold + 'Eff_dEshift_' + pars.momfold[pars.momentum][:7] +  '_t=' +  str(fitrange[0]) + '-' +  str(fitrange[-1]) + pars.snkfold[pars.sink] + filename + '.pdf')
    pypl.close()
    pypl.cla()
    print('saving time: \t', tm.time()-start3)
    
    print('effdEplotter time: \t', tm.time()-start)
    #---------------------------------------------

    
def ratioplotter(ratios, unpert_2p, fitfnc, fitfncratio, pars, opnum, best_tmin_ratio, best_tmin_oneexp, plots=True):
    """
    Loop over the the tmin values and plot the fits
    nclns: [nucleon, nucleon_q1, nucleon_q2] (unperturbed 2pt. fn. and perturbed 2pt. fn. for both quarks
    opnum: 0 or 1 for the operator ('g2' or 'g4')
    pars:  param class object
    plots: 0=No plots; 1=plot only the fit with the chi-squared values closest to 1; 2=plot every fit 
    """
    # Plot setup
    pypl.rc('font', size=18, **{'family': 'sans-serif','serif': ['Computer Modern']})
    pypl.rc('text', usetex=True)
    rcParams.update({'figure.autolayout': True})

    print('\n',pars.operators[opnum])
    pars.fit  = 'Twoexp'
    pars.fit2 = 'TwoexpRatio4'
    fitfunction = ff.initffncs(pars.fit)
    fitfncratio = ff.initffncs(pars.fit2)
    fitfunction_oneexp = ff.initffncs('Aexp')
    #-----------------------------------------------------------------------------
    # Read the fit data from the files with the best fit parameters
    mom = pars.momfold[pars.momentum]
    tmin = best_tmin_ratio[1][pars.momentum]
    folderenergy = pars.basedir0+mom[:8]+'fit_data_'+str(pars.nboot)+'/energyfit_'+pars.operators[opnum]+mom[:7]+'_Twoexp' + pars.snkfold[pars.sink]+'_tmin'+str(tmin)+'.csv'
    error = False
    A0,E0,A1,E1,tlist,chisq,error = readfitdata(folderenergy,error)
    energyfit = [ A0, E0, A1, E1]
    tlist = np.array([int(i) for i in tlist])
    chisq = float(chisq[0])

    # Make the ratio data line up assymptotically with the energy shift by taking the square root
    for quark in range(2):
        for ti in range(64):
            ratios[quark][ti] = ratios[quark][ti]**0.5
            ratios[quark][ti].Stats()
    
    for quark in np.arange(1,3):
        folderenergyshift=pars.basedir0+mom[:8]+'fit_data_'+str(pars.nboot)+'/energyshiftfit_'+pars.operators[opnum]+'_q'+str(quark)+mom[:7]+'_Twoexp' + pars.snkfold[pars.sink]+'_tmin'+str(tmin)+'.csv'
        error = False
        dA0,dE0,dA1,dE1,tlist,chisq,error = readfitdata(folderenergyshift,error)
        energyshiftfit = [dA0,dE0,dA1,dE1]
        tlist = np.array([int(i) for i in tlist])
        chisq = float(chisq[0])

        #-----------------------------------------------------------------------------
        # Plot the effective energy for the energy fit
        if plots==1:
            # Plots for the energy shift
            # dEplotter(pars,ratios[quark],trangelist[quark][indices[quark]],ratiofitlist[quark][indices[quark]],pars.plot_dir[opnum][quark],fitfncratio,ratiofitchisq[quark][indices[quark]]
            #           , filename = '_gamma'+op[-1]+'_quark'+str(quark+1)
            #           , title    = r'Energy shift '+pars.momfold[pars.momentum][:7]+', $\gamma_{'+op[-1]+'}$, quark '+str(quark+1)
            #           , ylabel   = r'$dE_{q_'+str(quark+1)+', \gamma_{'+op[-1]+'}}$'
            #           , ylim     = pars.ylims[opnum][quark][pars.momentum]
            #           , ratiofit = energyfit[index]
            # )
            op = pars.operators[opnum]
            # effdEplotter(pars, ratios[quark-1], tlist, energyshiftfit, pars.plot_dir[opnum][0], fitfncratio, chisq
            #              , filename = '_gamma'+op+'_quark'+str(quark)
            #              , title    = r'$\vec{q}='+np.array2string(pars.qval[pars.momentum], separator=',')+'$, $\gamma_{'+op[-1]+'}$, quark '+str(quark)
            #              , ylabel   = r'$dE_{q_'+str(quark)+', \gamma_{'+op[-1]+'}}$'
            #              , ylim     = pars.ylimde[opnum][quark-1][pars.momentum]
            #              , ratiofit = energyfit
            #              )
            effdEplotter_cheap(pars, ratios[quark-1], tlist, energyshiftfit, pars.plot_dir[opnum][0], fitfncratio, chisq
                         , filename = '_'+op+'_quark'+str(quark)
                         , title    = ''
                         , ylabel   = r'$dE_{q_'+str(quark)+', \gamma_{'+op[-1]+'}}$'
                         , ylim     = pars.ylimde[opnum][quark-1][pars.momentum]
                         , ratiofit = energyfit
                         )
            # Add another function here which plots all momenta on the same plot and shows the effective form factor.
#---------------------------------------------


def weighted_avg(pars, op=1,exclude0=False):
    if exclude0: minmom=1
    else: minmom=0
    #INFO: For the data from the ratiofit first
    tminlist   = []
    E          = []
    E1         = []
    dEq1       = []
    dEq2       = []
    chisq_E0   = []
    chisq_dEq1 = []
    chisq_dEq2 = []
    dof        = []
    #INFO: For the data from the one-exp fit
    tminlist_1   = []
    E_1          = []
    E1_1         = []
    dEq1_1       = []
    dEq2_1       = []
    chisq_E0_1   = []
    chisq_dEq1_1 = []
    chisq_dEq2_1 = []
    dof_1        = []
    #list of highest-weighted tmin values
    best_tmin_ratio = [[],[],[]]
    best_tmin_oneexp = [[],[],[]]
    # Loop over all the momenta
    for j,mom in enumerate(pars.momfold):
        try:
            # This excludes Q^2=0 for the magnetic form factor
            if exclude0 and j==0:
                pass
            else:
                #ratiofit
                tminlist.append([])
                chisq_E0.append([])
                chisq_dEq1.append([])
                chisq_dEq2.append([])
                dof.append([])
                E.append([])
                E1.append([])
                dEq1.append([])
                dEq2.append([])
                #onexp
                tminlist_1.append([])
                chisq_E0_1.append([])
                chisq_dEq1_1.append([])
                chisq_dEq2_1.append([])
                dof_1.append([])
                E_1.append([])
                E1_1.append([])
                dEq1_1.append([])
                dEq2_1.append([])

                print('\n',mom)
                # Read the fit data for any fits with tmin between 0 and 20
                for tmin in range(0,20):
                    folders  = [pars.basedir0+mom[:8]+'data_'+str(pars.nboot)+'/energyBS_'+pars.operators[op]+mom[:7]+'_'+pars.fitratio[:] + pars.snkfold[pars.sink]+'_tmin'+str(tmin)+'.csv',
                                pars.basedir1+mom[:8]+'data_'+str(pars.nboot)+'/energyBS_'+pars.operators[op]+mom[:7]+'_'+pars.fitoneexp[:]  + pars.snkfold[pars.sink]+'_tmin'+str(tmin)+'.csv']
                    folders_chi  = [pars.basedir0+mom[:8]+'data_'+str(pars.nboot)+'/fitchisq_'+pars.operators[op]+mom[:7]+'_'+pars.fitratio[:] + pars.snkfold[pars.sink]+'_tmin'+str(tmin)+'.csv',
                                    pars.basedir1+mom[:8]+'data_'+str(pars.nboot)+'/fitchisq_'+pars.operators[op]+mom[:7]+'_'+pars.fitoneexp[:]  + pars.snkfold[pars.sink]+'_tmin'+str(tmin)+'.csv']

                    errora=False # This will be set to true if the following reader function fails
                    E,E1,dEq1,dEq2,errora = readcsv(folders[0],E,E1,dEq1,dEq2,errora,-1) # Attempt to read the fit results
                    if not errora:
                        # tmin was found, so we save the fit data to variables
                        tminlist[-1].append(tmin)
                        errorb=False
                        # Also read the chi squared values and dof's from the file
                        csq_E0, csq_dEq1, csq_dEq2, degreesof, errorb = readcsv_chi(folders_chi[0],errorb)
                        chisq_E0[-1].append(csq_E0)
                        chisq_dEq1[-1].append(csq_dEq1)
                        chisq_dEq2[-1].append(csq_dEq2)
                        dof[-1].append(degreesof)
                    # else:
                    #     print('tmin=',tmin, 'not found')

                    # Same for the one-exponential fit
                    errorc=False
                    E_1,E1_1,dEq1_1,dEq2_1,errorc = readcsv(folders[1],E_1,E1_1,dEq1_1,dEq2_1,errorc,-1)
                    if not errorc:
                        tminlist_1[-1].append(tmin)
                        errord=False
                        csq_E0, csq_dEq1, csq_dEq2, degreesof, errord = readcsv_chi(folders_chi[1],errord)
                        chisq_E0_1[-1].append(csq_E0)
                        chisq_dEq1_1[-1].append(csq_dEq1)
                        chisq_dEq2_1[-1].append(csq_dEq2)
                        dof_1[-1].append(degreesof)

                #INFO: Make copies of the tminlists because we could use different tmin ranges for the quarks 1&2
                tminlist_dEq1 = copy.copy(tminlist)
                tminlist_dEq2 = copy.copy(tminlist)
                dof_dEq1 = copy.copy(dof)
                dof_dEq2 = copy.copy(dof)
                tminlist_dEq1_1 = copy.copy(tminlist_1)
                tminlist_dEq2_1 = copy.copy(tminlist_1)
                dof_dEq1_1 = copy.copy(dof_1)
                dof_dEq2_1 = copy.copy(dof_1)

                # This will pick out the range of tmin values to use in the weighted averaging (if choosemin=True in param.py)
                # two-exponential ratio fit
                if len(tminlist[-1])>0 and pars.choosetmins:
                    mini = tminlist[-1][0]
                    lnt = len(tminlist[-1])
                    E[-1] = E[-1][max(0,pars.tmin_energy[j][0]-mini):min(lnt,pars.tmin_energy[j][1]-mini)]
                    chisq_E0[-1] = chisq_E0[-1][max(0,pars.tmin_energy[j][0]-mini):min(lnt,pars.tmin_energy[j][1]-mini)]
                    tminlist[-1] = tminlist[-1][max(0,pars.tmin_energy[j][0]-mini):min(lnt,pars.tmin_energy[j][1]-mini)]
                    dof[-1] = dof[-1][max(0,pars.tmin_energy[j][0]-mini):min(lnt,pars.tmin_energy[j][1]-mini)]

                    dEq1[-1] = dEq1[-1][max(0,pars.tmin_quarks[op][0][j][0]-mini):min(lnt,pars.tmin_quarks[op][0][j][1]-mini)]
                    chisq_dEq1[-1] = chisq_dEq1[-1][max(0,pars.tmin_quarks[op][0][j][0]-mini):min(lnt,pars.tmin_quarks[op][0][j][1]-mini)]
                    tminlist_dEq1[-1] = tminlist_dEq1[-1][max(0,pars.tmin_quarks[op][0][j][0]-mini):min(lnt,pars.tmin_quarks[op][0][j][1]-mini)]
                    dof_dEq1[-1] = dof_dEq1[-1][max(0,pars.tmin_quarks[op][0][j][0]-mini):min(lnt,pars.tmin_quarks[op][0][j][1]-mini)]

                    dEq2[-1] = dEq2[-1][max(0,pars.tmin_quarks[op][1][j][0]-mini):min(lnt,pars.tmin_quarks[op][1][j][1]-mini)]
                    chisq_dEq2[-1] = chisq_dEq2[-1][max(0,pars.tmin_quarks[op][1][j][0]-mini):min(lnt,pars.tmin_quarks[op][1][j][1]-mini)]
                    tminlist_dEq2[-1] = tminlist_dEq2[-1][max(0,pars.tmin_quarks[op][1][j][0]-mini):min(lnt,pars.tmin_quarks[op][1][j][1]-mini)]
                    dof_dEq2[-1] = dof_dEq2[-1][max(0,pars.tmin_quarks[op][1][j][0]-mini):min(lnt,pars.tmin_quarks[op][1][j][1]-mini)]

                #oneexp fit 
                if len(tminlist_1[-1])>0 and pars.choosetmins_1:
                    mini = tminlist_1[-1][0]
                    lnt = len(tminlist_1[-1])
                    E_1[-1] = E_1[-1][max(0,pars.tmin_energy_1[j][0]-mini):min(lnt,pars.tmin_energy_1[j][1]-mini)]
                    chisq_E0_1[-1] = chisq_E0_1[-1][max(0,pars.tmin_energy_1[j][0]-mini):min(lnt,pars.tmin_energy_1[j][1]-mini)]
                    tminlist_1[-1] = tminlist_1[-1][max(0,pars.tmin_energy_1[j][0]-mini):min(lnt,pars.tmin_energy_1[j][1]-mini)]
                    dof_1[-1] = dof_1[-1][max(0,pars.tmin_energy_1[j][0]-mini):min(lnt,pars.tmin_energy_1[j][1]-mini)]

                    dEq1_1[-1] = dEq1_1[-1][max(0,pars.tmin_quarks_1[op][0][j][0]-mini):min(lnt,pars.tmin_quarks_1[op][0][j][1]-mini)]
                    chisq_dEq1_1[-1] = chisq_dEq1_1[-1][max(0,pars.tmin_quarks_1[op][0][j][0]-mini):min(lnt,pars.tmin_quarks_1[op][0][j][1]-mini)]
                    tminlist_dEq1_1[-1] = tminlist_dEq1_1[-1][max(0,pars.tmin_quarks_1[op][0][j][0]-mini):min(lnt,pars.tmin_quarks_1[op][0][j][1]-mini)]
                    dof_dEq1_1[-1] = dof_dEq1_1[-1][max(0,pars.tmin_quarks_1[op][0][j][0]-mini):min(lnt,pars.tmin_quarks_1[op][0][j][1]-mini)]

                    dEq2_1[-1] = dEq2_1[-1][max(0,pars.tmin_quarks_1[op][1][j][0]-mini):min(lnt,pars.tmin_quarks_1[op][1][j][1]-mini)]
                    chisq_dEq2_1[-1] = chisq_dEq2_1[-1][max(0,pars.tmin_quarks_1[op][1][j][0]-mini):min(lnt,pars.tmin_quarks_1[op][1][j][1]-mini)]
                    tminlist_dEq2_1[-1] = tminlist_dEq2_1[-1][max(0,pars.tmin_quarks_1[op][1][j][0]-mini):min(lnt,pars.tmin_quarks_1[op][1][j][1]-mini)]
                    dof_dEq2_1[-1] = dof_dEq2_1[-1][max(0,pars.tmin_quarks_1[op][1][j][0]-mini):min(lnt,pars.tmin_quarks_1[op][1][j][1]-mini)]

                #INFO: Calculate weights for each fit here.(in functions) ratiofit
                weights_E0 = weights(dof[-1], chisq_E0[-1], [i.Std for i in E[-1]])
                weights_dEq1 = weights(dof_dEq1[-1], chisq_dEq1[-1], [i.Std for i in dEq1[-1]])
                weights_dEq2 = weights(dof_dEq2[-1], chisq_dEq2[-1], [i.Std for i in dEq2[-1]])
                #INFO: Calculate weights for each fit here.(in functions) one-exponential fit
                weights_E0_1 = weights(dof_1[-1], chisq_E0_1[-1], [i.Std for i in E_1[-1]])
                weights_dEq1_1 = weights(dof_dEq1_1[-1], chisq_dEq1_1[-1], [i.Std for i in dEq1_1[-1]])
                weights_dEq2_1 = weights(dof_dEq2_1[-1], chisq_dEq2_1[-1], [i.Std for i in dEq2_1[-1]])

                # take the index of the highest weighting and add the corresponding tmin value to the right list:
                best_tmin_ratio[0].append( tminlist[-1][np.argmax(weights_E0)])
                best_tmin_oneexp[0].append( tminlist_1[-1][np.argmax(weights_E0_1)])
                best_tmin_ratio[1].append( tminlist_dEq1[-1][np.argmax(weights_dEq1)])
                best_tmin_oneexp[1].append( tminlist_dEq1_1[-1][np.argmax(weights_dEq1_1)])
                best_tmin_ratio[2].append( tminlist_dEq2[-1][np.argmax(weights_dEq2)])
                best_tmin_oneexp[2].append( tminlist_dEq2_1[-1][np.argmax(weights_dEq2_1)])
            print('didit')
        except:
            pass
    return best_tmin_ratio, best_tmin_oneexp
 
def effmass(pars, ydata, plot=False):
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
        pypl.ylim(pars.ylimde[opnum][0][pars.momentum])
        pypl.grid(True, alpha=0.4)
        ax = pypl.gca()
        pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
        pypl.savefig(pars.plot_dir[0][0] + '../Eff_energy_' + pars.momfold[pars.momentum][:7] +  '_'+ pars.snkfold[pars.sink] + '.pdf')
        pypl.close()
    return effmass

def effamp(pars, ydata, plot=False,timeslice=10):
    """
    Return the effective amplitude and plot it if plot==True
    Idea comes from Hoerz2020 paper
    """
    effmass0 = stats.effectivemass(ydata)
    effamp =[]
    for i, meff in enumerate(effmass0):
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
        #pypl.ylim(bottom=pars.ylimeffmass0[0],top=pars.ylimeffmass0[1])
        # pypl.ylim(-3000,-250)
        pypl.ylim(yavg[timeslice]-20*yerr[timeslice], yavg[timeslice]+10*yerr[timeslice])
        pypl.grid(True, alpha=0.4)
        ax = pypl.gca()
        pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
        pypl.savefig(pars.plot_dir[0][0] + '../Eff_amplitude_' + pars.momfold[pars.momentum][:7] +  '_'+ pars.snkfold[pars.sink] + '.pdf')
        pypl.close()
    return effamp

def makeratio(nclns, opnum):
    """Construct the ratio of correlators"""
    # The ratio of perturbed and unperturbed two-point functions is different for the two operators
    if opnum==0: #g2
        ratiou = stats.feynhellratio5(nclns[0], nclns[1])
        ratiod = stats.feynhellratio5(nclns[0], nclns[2])
    elif opnum==1: #g4
        ratiou = stats.feynhellratio2(nclns[1][0], nclns[0][1], nclns[0][0], nclns[1][1])
        ratiod = stats.feynhellratio2(nclns[2][0], nclns[0][1], nclns[0][0], nclns[2][1])
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
        for i in range(len(nclns[quark][0])):
            # Take average over trev=0 and trev=1
            # Take sum or difference of unpolarized and polarized projections (spin up, spin down)
            spinupposm = 0.25*(nclns[quark][0][i]+nclns[quark][1][i]+nclns[quark][2][i]+nclns[quark][3][i])
            spindnposm = 0.25*(nclns[quark][0][i]-nclns[quark][1][i]+nclns[quark][2][i]-nclns[quark][3][i])
            spinupnegm = 0.25*(nclns[quark][4][i]+nclns[quark][5][i]+nclns[quark][6][i]+nclns[quark][7][i])
            spindnnegm = 0.25*(nclns[quark][4][i]-nclns[quark][5][i]+nclns[quark][6][i]-nclns[quark][7][i])

            # Take the sam combinations for the correlators with FH perturbations
            spinupposmlmb = 0.25*(nclns[quark][8][i]+nclns[quark][9][i]+nclns[quark][10][i]+nclns[quark][11][i])
            spindnposmlmb = 0.25*(nclns[quark][8][i]-nclns[quark][9][i]+nclns[quark][10][i]-nclns[quark][11][i])
            spinupnegmlmb = 0.25*(nclns[quark][12][i]+nclns[quark][13][i]+nclns[quark][14][i]+nclns[quark][15][i])
            spindnnegmlmb = 0.25*(nclns[quark][12][i]-nclns[quark][13][i]+nclns[quark][14][i]-nclns[quark][15][i])

            # Construct the ratio of the various correlators defined above
            ratio1 = ((spinupposmlmb*spindnposm)*(spinupposm*spindnposmlmb)**(-1))**(1/2)
            ratio2 = ((spinupnegm*spindnnegmlmb)*(spinupnegmlmb*spindnnegm)**(-1))**(1/2)
            fullratio[quark].append( ratio1*ratio2 )
            fullratio[quark][-1].Stats()
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
    colors    = ['r', 'g', 'b', 'k', 'y', 'm','k','k']
    markers   = ['s','o','^','*', 'v', '>', '<','s','s']
    labels    = ['FH Ratio fit', 'FH One-exp fit', 'FH corr-fit', 'FH combined corr-fit']
    datafiles = ['Ratio_fit', 'One-exp_fit', 'corr_fit', 'combined_corr_fit']

    if len(sys.argv)>2:
        print("Starting ffplot")
        kappas     = int(sys.argv[1]) 
        sinktype   = int(sys.argv[2]) # ptsnk smsnk30 smsnk60
        pars = pr.params(kappas, sinktype)
        pars.fit = 'FFplot'
        pars.makeresultdir(plots=True)
        pypl.rc('font', size=18, **{'family': 'DejaVu Sans'})
        pypl.rc('text', usetex=True)

        best_tmin_ratio, best_tmin_oneexp = weighted_avg(pars, op=1,exclude0=False)
        best_tmin_ratioM, best_tmin_oneexpM = weighted_avg(pars, op=0,exclude0=True)

        for j,mom in enumerate(pars.momfold):
            #parsratio = pr.params(kappas, sinktype, j)
            pars.momentum = j
            #Set the operator choice correctly for the given momentum
            if j==0: pars.numbers=2; pars.opchoice=[1] #For Q=0, only calculate the electric FF.
            else: pars.numbers=10; pars.opchoice=[0,1] #For Q=0, only calculate the electric FF.
            #Initialise the fitting function
            fitfnconeexp = ff.initffncs('Aexp')
            fitfnctwoexp = ff.initffncs('Twoexp')
            fitfncratio  = ff.initffncs('TwoexpRatio4')

            #Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
            nucleon_data = read_data(pars)
            # Loop over the two operator types
            for opnum in pars.opchoice:
                ratios, unpert_2p = makeratio(nucleon_data,opnum)
                # If kappa=kp119930kp119930 then also read the gamma3 files
                if pars.kappas==14 and pars.momentum!=0:
                    nucleon_datag3 = read_datag3_ratio(pars,sgn=-1.0)
                    ratiosg3, unpert_2pg3 = makeratiog3(nucleon_datag3)
                    ratios, unpert_2p = combine_ratios(ratios, ratiosg3, unpert_2p, unpert_2pg3)
                print('yas-s')

                ratioplotter(ratios, unpert_2p, fitfnctwoexp, fitfncratio, pars, opnum, best_tmin_ratio, best_tmin_oneexp, plots=True)
            # print('yass')
            # except:
            #     print('ehhh')
            #     pass
        
    else:
        print("arg[1] = kappa")
        print("arg[2] = sinktype")
