# -*- eval: (comment-tags-mode) -*-
#INFO: This script will calculate the form factors of the baryons by using multiple fits to the data and taking a weighted average of them based on the uncertainty, chi-squared value and degrees of freedom.
import numpy as np
from scipy.special import gamma, gammainc, gammaincc
import csv
import copy
import matplotlib.pyplot as pypl
from matplotlib import rcParams
import math
import sys

from BootStrap3 import BootStrap
import params as pr
from formatting import err_brackets

def writeBS(filename, headernames, bootstraps):
    """
    filename to be written to
    headernames to print in the first row
    list of bootstrap elements of the same length as headernames.
    """
    with open(filename,'w') as csvfile:
        datawrite = csv.writer(csvfile,delimiter=',',quotechar='|')
        datawrite.writerow(headernames)
        for i in range(bootstraps[-1].nboot):
            datawrite.writerow([bs.values[i] for bs in bootstraps])
    return
            
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


def threeptread(threept, error):
    try:
        with open(threept) as csvfile:
            dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
            rows = [x for x in dataread]
            x3  = np.array([float(row[0]) for row in rows])
            GE3 = np.array([float(row[1]) for row in rows])
            GEerr3 = np.array([float(row[2]) for row in rows])
    except IOError:
        error=True
        print("Three-pt. file not found")
    return x3, GE3, GEerr3, error

def plotenergyshift(pars, Qsqrd, dE1, dE1err, gamma=4,quark=1, ylim=None):
    quarknum = ['dbl\ quark', 'sgl\ quark']
    pypl.figure('dE', figsize=(16,9))
    dE_wa = [d/pars.almb for d in dE1]
    dEerr_wa = [d/pars.almb for d in dE1err]
    if (pars.lattice=='Feyn-Hell_kp120900kp120900/' or pars.lattice=='Feyn-Hell_kp121095kp120512/' or pars.lattice=='Feyn-Hell_kp122005kp122005/' or pars.lattice=='Feyn-Hell_kp122130kp121756/' or pars.lattice=='Feyn-Hell_kp122078kp121859/') and gamma==4:
        dE_wa[0] = dE_wa[0]/2
        dEerr_wa[0] = dEerr_wa[0]/2
    pypl.errorbar(Qsqrd+1/10, dE_wa, dEerr_wa, fmt='.', capsize=4, marker='s', color=pars.colors[1], label='weighted avg')
    pypl.legend(fontsize='small', framealpha=1.0)
    pypl.xlabel(r'$Q^2[\mathrm{GeV}^2]$',labelpad=14,fontsize=18)
    pypl.ylabel(r'$dE_{\gamma_'+str(gamma)+'}/\lambda$',labelpad=5,fontsize=18)
    pypl.title(r'dE$_{\mathrm{\gamma_'+str(gamma)+','+quarknum[quark-1]+'}}$')
    pypl.ylim(ylim)
    pypl.xlim(0,10)
    pypl.grid(True, alpha=0.4)
    #pypl.savefig(pars.savedir+'dEg'+str(gamma)+'q'+str(quark)+'.pdf')
    pypl.savefig(pars.plot_dir[0][0]+'../dEg'+str(gamma)+'q'+str(quark)+'.pdf')
    pypl.close()
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

def plotweights(tminlist, weights, redchisqlist, filename, switch):
    fig,ax1 = pypl.subplots(figsize=(9,6))

    # Plot the reduced chi-squared line
    ax1.plot(tminlist[:switch], redchisqlist[:switch], color='b', marker='.',label='ratio of 2 exponentials')
    ax1.plot(tminlist[switch:], redchisqlist[switch:], color='k', marker='.',label='1 exponential')
    #ax1.plot(tminlist, redchisqlist, color='k')
    ax1.set_ylim(0,3)
    ax1.set_ylabel(r'$\chi^2_{\textrm{dof}}$')
    ax1.set_xlabel(r't$_{\textrm{min}}/a$')
    ax1.legend(fontsize='x-small', framealpha=0.8,facecolor='white', frameon=True, shadow=False, fancybox=True, loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(tminlist[:switch], weights[:switch], color='b',alpha=0.3)
    ax2.bar(tminlist[switch:], weights[switch:], color='k',alpha=0.3)
    ax2.set_ylabel('weights')

    pypl.savefig(filename)
    #pypl.show()
    pypl.close()
    return
    
def plotenergies(tminlist, energies, energieserr, lower, upper, filename, weights=None, switch=None, lowersyst=None, uppersyst=None, ylabel=r'$a E_{\textrm{p}}$'):
    rcParams.update({'figure.autolayout': True})
    fig,ax1 = pypl.subplots(figsize=(9,6))
    if weights:
        ax2 = ax1.twinx()
        if switch:
            ax2.bar(tminlist[:switch], weights[:switch], color='b',alpha=0.3)
            ax2.bar(tminlist[switch:], weights[switch:], color='k',alpha=0.3)
        else:
            ax2.bar(tminlist, weights, color='b',alpha=0.3)
        ax2.set_ylabel(r'weights')
    if switch:
        ax1.errorbar(tminlist[:switch], energies[:switch], energieserr[:switch], fmt=markers[0], capsize=4, color='b',label='ratio of 2 exponentials')
        ax1.errorbar(tminlist[switch:], energies[switch:], energieserr[switch:], fmt=markers[0], capsize=4, color='k',label='1 exponential')
        ax1.legend(fontsize='x-small', framealpha=0.8,facecolor='white', frameon=True, shadow=False, fancybox=True, loc='upper left')
    else:
        ax1.errorbar(tminlist, energies, energieserr, fmt=markers[0], capsize=4, color=pars.colors[0])
    ax1.fill_between(tminlist, [lower]*len(tminlist), [upper]*len(tminlist), color=pars.colors[0], alpha=0.4, linewidth=0)
    if lowersyst and uppersyst:
        ax1.fill_between(tminlist, [lowersyst]*len(tminlist), [uppersyst]*len(tminlist), color=pars.colors[0], alpha=0.3, linewidth=0)
    ax1.set_ylabel(ylabel)
    #ax1.set_ylabel(r'$a E_{\textrm{p}}$')
    #ax1.set_ylabel(r'Energy')
    ax1.set_xlabel(r't$_{\textrm{min}}/a$')
    ax1.set_ylim((lower+upper)/2-3*(upper-lower), (lower+upper)/2+3*(upper-lower))
    pypl.savefig(filename)
    pypl.close()
    return

def plotenergies2(tminlist, energies, energieserr, lower, upper, filename, weights=None, switch=None, lowersyst=None, uppersyst=None, ylabel=r'$a E_{\textrm{p}}$', title=''):
    """ Only plot the energies agains tmin, no weights and no average"""
    rcParams.update({'figure.autolayout': True})
    fig,ax1 = pypl.subplots(figsize=(8,6))
    # if weights:
    #     ax2 = ax1.twinx()
    #     if switch:
    #         ax2.bar(tminlist[:switch], weights[:switch], color='b',alpha=0.3)
    #         ax2.bar(tminlist[switch:], weights[switch:], color='k',alpha=0.3)
    #     else:
    #         ax2.bar(tminlist, weights, color='b',alpha=0.3)
    #     ax2.set_ylabel(r'weights')
    if switch:
        ax1.errorbar(tminlist[:switch], energies[:switch], energieserr[:switch], fmt=markers[0], capsize=4, color='b',label='ratio of 2 exponentials')
        ax1.errorbar(tminlist[switch:], energies[switch:], energieserr[switch:], fmt=markers[0], capsize=4, color='k',label='1 exponential')
        ax1.legend(fontsize='x-small', framealpha=0.8,facecolor='white', frameon=True, shadow=False, fancybox=True, loc='upper left')
    else:
        ax1.errorbar(tminlist, energies, energieserr, fmt=markers[0], capsize=4, color=pars.colors[0])
    # ax1.fill_between(tminlist, [lower]*len(tminlist), [upper]*len(tminlist), color=pars.colors[0], alpha=0.4, linewidth=0)
    # if lowersyst and uppersyst:
    #     ax1.fill_between(tminlist, [lowersyst]*len(tminlist), [uppersyst]*len(tminlist), color=pars.colors[0], alpha=0.3, linewidth=0)
    ax1.set_ylabel(ylabel)
    #ax1.set_ylabel(r'$a E_{\textrm{p}}$')
    #ax1.set_ylabel(r'Energy')
    ax1.grid(True, alpha=0.4)
    ax1.set_xlabel(r't$_{\textrm{min}}/a$')
    ax1.set_ylim((lower+upper)/2-3*(upper-lower), (lower+upper)/2+3*(upper-lower))
    pypl.title(title)
    pypl.savefig(filename)
    pypl.close()
    return



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
    E1_1          = []
    dEq1_1       = []
    dEq2_1       = []
    chisq_E0_1   = []
    chisq_dEq1_1 = []
    chisq_dEq2_1 = []
    dof_1        = []
    means      = []
    mean_errs  = []
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
                    errora=False
                    # print(f"{folders[0]=}")
                    # print(f"{folders[1]=}")

                    E,E1,dEq1,dEq2,errora = readcsv(folders[0],E,E1,dEq1,dEq2,errora,-1)
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


                #INFO: Choose a set of tmin values to include
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
                    # print(f"{mini=}")
                    # print(f"{lnt=}")
                    # print(f"{pars.tmin_energy_1[j][1]-mini=}")
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

                # combine ratiofit and oneexp fit data into one list
                E[-1] = E[-1] + E_1[-1]
                chisq_E0[-1] = chisq_E0[-1] + chisq_E0_1[-1]
                tminlist[-1] = tminlist[-1] + tminlist_1[-1]
                dof[-1] = dof[-1] + dof_1[-1]

                dEq1[-1] = dEq1[-1] + dEq1_1[-1]
                chisq_dEq1[-1] = chisq_dEq1[-1] + chisq_dEq1_1[-1]
                tminlist_dEq1[-1] = tminlist_dEq1[-1] + tminlist_dEq1_1[-1]
                dof_dEq1[-1] = dof_dEq1[-1] + dof_dEq1_1[-1]

                dEq2[-1] = dEq2[-1] + dEq2_1[-1]
                chisq_dEq2[-1] = chisq_dEq2[-1] + chisq_dEq2_1[-1]
                tminlist_dEq2[-1] = tminlist_dEq2[-1] + tminlist_dEq2_1[-1]
                dof_dEq2[-1] = dof_dEq2[-1] + dof_dEq2_1[-1]

                #INFO: Calculate weights for each fit here.(in functions)
                weights_E0 = weights(dof[-1], chisq_E0[-1], [i.Std for i in E[-1]])
                # print(f"{weights_E0=}")
                weights_dEq1 = weights(dof_dEq1[-1], chisq_dEq1[-1], [i.Std for i in dEq1[-1]])
                weights_dEq2 = weights(dof_dEq2[-1], chisq_dEq2[-1], [i.Std for i in dEq2[-1]])

                #TODO: Calculate the errors on the bootstrap level
                # if op==1 or j!=0:
                #     avgE0 = sum( np.array(weights_E0)*np.array(E[-1]) )
                #     avgE0.Stats()
                #     systE0sq = sum( weights_E0*(np.array(E[-1]) - avgE0)**2 )
                #     systE0 = systE0sq**(0.5)
                #     systE0.Stats()
                #     combE0 = np.sqrt(avgE0.Std**2 + systE0.Avg**2)
                #     print('--------------------')
                #     print(f"{avgE0=}")
                #     print(f"{avgE0.Avg=}")
                #     print(f"{avgE0.Std=}")
                #     print(f"{systE0.Avg=}")
                #     print(f"{systE0.Std=}")
                #     print(f"{combE0=}") 
                #     print('--------------------')

                avgE0 = sum( np.array(weights_E0)*np.array(E[-1]) )
                avgdEq1 = sum( np.array(weights_dEq1)*np.array(dEq1[-1]) )
                avgdEq2 = sum( np.array(weights_dEq2)*np.array(dEq2[-1]) )

                mean_E0 = sum( weights_E0*np.array([i.Avg for i in E[-1]]) )
                mean_dEq1 = sum( weights_dEq1*np.array([i.Avg for i in dEq1[-1]]) )
                mean_dEq2 = sum( weights_dEq2*np.array([i.Avg for i in dEq2[-1]]) )

                # # Calculate the systematic errors of each weighted average
                # # This would calculate a systematic error on each boostrap (but then would you average them?)
                # systerrsq_E0BS = sum( weights_E0*np.array([(i-avgE0)**2 for i in E[-1]]) )
                # systerrsq_dEq1BS = sum( weights_dEq1*np.array([(i-avgdEq1)**2 for i in dEq1[-1]]) )
                # systerrsq_dEq2BS = sum( weights_dEq2*np.array([(i-avgdEq2)**2 for i in dEq2[-1]]) )
                # systerrsq_E0 = systerrsq_E0BS.Avg
                # systerrsq_dEq1 = systerrsq_dEq1BS.Avg
                # systerrsq_dEq2 = systerrsq_dEq2BS.Avg

                # Calculate the systematic errors of each weighted average
                systerrsq_E0 = sum( weights_E0*np.array([(i.Avg-mean_E0)**2 for i in E[-1]]) )
                systerrsq_dEq1 = sum( weights_dEq1*np.array([(i.Avg-mean_dEq1)**2 for i in dEq1[-1]]) )
                systerrsq_dEq2 = sum( weights_dEq2*np.array([(i.Avg-mean_dEq2)**2 for i in dEq2[-1]]) )

                # staterr_E0 = sum( weights_E0*np.array([i.Std**2 for i in E[-1]]) )
                # staterr_dEq1 = sum( weights_dEq1*np.array([i.Std**2 for i in dEq1[-1]]) )
                # staterr_dEq2 = sum( weights_dEq2*np.array([i.Std**2 for i in dEq2[-1]]) )

                avgE0.Stats()
                avgdEq1.Stats()
                avgdEq2.Stats()
                staterr_E0 = avgE0.Std**2
                staterr_dEq1 = avgdEq1.Std**2
                staterr_dEq2 = avgdEq2.Std**2

                comberr_E0 = np.sqrt(staterr_E0+systerrsq_E0)
                comberr_dEq1 = np.sqrt(staterr_dEq1+systerrsq_dEq1)
                comberr_dEq2 = np.sqrt(staterr_dEq2+systerrsq_dEq2)

                # print(f"{pars.savedir + pars.operators[op]+ '/weights_chi_'+mom[:7]+'_E0.pdf'=}")
                # print(f"{pars.plot_dir[0]=}")

                # plotweights(tminlist[-1],weights_E0, np.array(chisq_E0[-1])/np.array(dof[-1]), pars.savedir + pars.operators[op]+ '/weights_chi_'+mom[:7]+'_E0.pdf',len(E[-1])-len(E_1[-1]) )
                # plotweights(tminlist_dEq1[-1],weights_dEq1, np.array(chisq_dEq1[-1])/np.array(dof_dEq1[-1]), pars.savedir + pars.operators[op]+ '/weights_chi_'+mom[:7]+'_dEq1.pdf',len(dEq1[-1])-len(dEq1_1[-1]) )
                # plotweights(tminlist_dEq2[-1],weights_dEq2, np.array(chisq_dEq2[-1])/np.array(dof_dEq2[-1]), pars.savedir + pars.operators[op]+ '/weights_chi_'+mom[:7]+'_dEq2.pdf',len(dEq2[-1])-len(dEq2_1[-1]) )
                plotweights(tminlist[-1],weights_E0, np.array(chisq_E0[-1])/np.array(dof[-1]), pars.plot_dir[op][0] + '/weights_chi_q'+np.array2string(pars.qval[j],separator='')[1:-1] + '_E0.pdf',len(E[-1])-len(E_1[-1]) )
                plotweights(tminlist_dEq1[-1],weights_dEq1, np.array(chisq_dEq1[-1])/np.array(dof_dEq1[-1]), pars.plot_dir[op][0] + '/weights_chi_q'+np.array2string(pars.qval[j],separator='')[1:-1] + '_dEq1.pdf',len(dEq1[-1])-len(dEq1_1[-1]) )
                plotweights(tminlist_dEq2[-1],weights_dEq2, np.array(chisq_dEq2[-1])/np.array(dof_dEq2[-1]), pars.plot_dir[op][0] + '/weights_chi_q'+np.array2string(pars.qval[j],separator='')[1:-1] + '_dEq2.pdf',len(dEq2[-1])-len(dEq2_1[-1]) )


                # Plot the weighted average with the just the stat. error.
                # plotenergies(tminlist[-1],np.array([i.Avg for i in E[-1]]),np.array([i.Std for i in E[-1]]), avgE0.Avg-avgE0.Std, avgE0.Avg+avgE0.Std, pars.savedir + pars.operators[op]+ '/energies_'+mom[:7]+'_E0.pdf', weights=weights_E0,switch=len(E[-1])-len(E_1[-1]))
                # plotenergies(tminlist_dEq1[-1],np.array([i.Avg for i in dEq1[-1]]),np.array([i.Std for i in dEq1[-1]]), avgdEq1.Avg-avgdEq1.Std, avgdEq1.Avg+avgdEq1.Std, pars.savedir + pars.operators[op]+ '/energies_'+mom[:7]+'_dEq1.pdf', weights=weights_dEq1,switch=len(dEq1[-1])-len(dEq1_1[-1]))
                # plotenergies(tminlist_dEq2[-1],np.array([i.Avg for i in dEq2[-1]]),np.array([i.Std for i in dEq2[-1]]), avgdEq2.Avg-avgdEq2.Std, avgdEq2.Avg+avgdEq2.Std, pars.savedir + pars.operators[op]+ '/energies_'+mom[:7]+'_dEq2.pdf', weights=weights_dEq2,switch=len(dEq2[-1])-len(dEq2_1[-1]))

                # Plot the weighted average with the stat. error and the combined error shown
                plotenergies(tminlist[-1],np.array([i.Avg for i in E[-1]]),np.array([i.Std for i in E[-1]]), avgE0.Avg-avgE0.Std, avgE0.Avg+avgE0.Std
                             , pars.plot_dir[op][0] + '/energies_q'+np.array2string(pars.qval[j],separator='')[1:-1] + '_E0.pdf'
                             , weights=weights_E0
                             , switch=len(E[-1])-len(E_1[-1])
                             , lowersyst=mean_E0-comberr_E0
                             , uppersyst=mean_E0+comberr_E0
                             , ylabel=r'$a E_{\textrm{p}}$')
                plotenergies(tminlist_dEq1[-1],np.array([i.Avg for i in dEq1[-1]]),np.array([i.Std for i in dEq1[-1]]), avgdEq1.Avg-avgdEq1.Std, avgdEq1.Avg+avgdEq1.Std
                             , pars.plot_dir[op][0]+ '/energies_q'+np.array2string(pars.qval[j],separator='')[1:-1] + '_dEq1.pdf'
                             , weights=weights_dEq1
                             ,switch=len(dEq1[-1])-len(dEq1_1[-1])
                             , lowersyst=mean_dEq1-comberr_dEq1
                             , uppersyst=mean_dEq1+comberr_dEq1
                             , ylabel=r'$a dE_{u,\gamma_{'+ pars.operators[op][-1]+'}}$')
                plotenergies(tminlist_dEq2[-1],np.array([i.Avg for i in dEq2[-1]]),np.array([i.Std for i in dEq2[-1]]), avgdEq2.Avg-avgdEq2.Std, avgdEq2.Avg+avgdEq2.Std
                             , pars.plot_dir[op][0]+ '/energies_q'+np.array2string(pars.qval[j],separator='')[1:-1] + '_dEq2.pdf'
                             , weights=weights_dEq2
                             , switch=len(dEq2[-1])-len(dEq2_1[-1])
                             , lowersyst=mean_dEq2-comberr_dEq2
                             , uppersyst=mean_dEq2+comberr_dEq2
                             , ylabel=r'$a dE_{d,\gamma_{'+ pars.operators[op][-1]+'}}$')

                #Plot just the energies without average or weighting
                plotenergies2(tminlist[-1],np.array([i.Avg for i in E[-1]]),np.array([i.Std for i in E[-1]]), avgE0.Avg-avgE0.Std, avgE0.Avg+avgE0.Std
                              , pars.plot_dir[op][0]+ '/noavg/energies_q'+np.array2string(pars.qval[j],separator='')[1:-1] + '_E0.pdf'
                              , weights=weights_E0
                              , switch=len(E[-1])-len(E_1[-1])
                              , lowersyst=mean_E0-comberr_E0
                              , uppersyst=mean_E0+comberr_E0
                              , ylabel=r'$a E_{\textrm{p}}$'
                              , title=r'$\mathbf{q}/a=$'+str(pars.qval[j]))
                plotenergies2(tminlist_dEq1[-1],np.array([i.Avg for i in dEq1[-1]]),np.array([i.Std for i in dEq1[-1]]), avgdEq1.Avg-avgdEq1.Std, avgdEq1.Avg+avgdEq1.Std
                              , pars.plot_dir[op][0]+ '/noavg/energies_q'+np.array2string(pars.qval[j],separator='')[1:-1]+'_dEq1.pdf'
                              , weights=weights_dEq1
                              , switch=len(dEq1[-1])-len(dEq1_1[-1])
                              , lowersyst=mean_dEq1-comberr_dEq1
                              , uppersyst=mean_dEq1+comberr_dEq1
                              , ylabel=r'$a dE_{u,\gamma_{'+ pars.operators[op][-1]+'}}$'
                              , title=r'$\mathbf{q}/a=$'+str(pars.qval[j]))
                plotenergies2(tminlist_dEq2[-1],np.array([i.Avg for i in dEq2[-1]]),np.array([i.Std for i in dEq2[-1]]), avgdEq2.Avg-avgdEq2.Std, avgdEq2.Avg+avgdEq2.Std
                              , pars.plot_dir[op][0]+ '/noavg/energies_q'+np.array2string(pars.qval[j],separator='')[1:-1]+'_dEq2.pdf'
                              , weights=weights_dEq2
                              , switch=len(dEq2[-1])-len(dEq2_1[-1])
                              , lowersyst=mean_dEq2-comberr_dEq2
                              , uppersyst=mean_dEq2+comberr_dEq2
                              , ylabel=r'$a dE_{d,\gamma_{'+ pars.operators[op][-1]+'}}$'
                              , title=r'$\mathbf{q}/a=$'+str(pars.qval[j]))

                # Plot the weighted average with the combined error shown.
                # plotenergies(tminlist[-1],np.array([i.Avg for i in E[-1]]),np.array([i.Std for i in E[-1]]), mean_E0-comberr_E0, mean_E0+comberr_E0, pars.savedir + pars.operators[op]+ '/energies_'+mom[:7]+'_E0.pdf', weights=weights_E0,switch=len(E[-1])-len(E_1[-1]))
                # plotenergies(tminlist_dEq1[-1],np.array([i.Avg for i in dEq1[-1]]),np.array([i.Std for i in dEq1[-1]]), mean_dEq1-comberr_dEq1, mean_dEq1+comberr_dEq1, pars.savedir + pars.operators[op]+ '/energies_'+mom[:7]+'_dEq1.pdf', weights=weights_dEq1,switch=len(dEq1[-1])-len(dEq1_1[-1]))
                # plotenergies(tminlist_dEq2[-1],np.array([i.Avg for i in dEq2[-1]]),np.array([i.Std for i in dEq2[-1]]), mean_dEq2-comberr_dEq2, mean_dEq2+comberr_dEq2, pars.savedir + pars.operators[op]+ '/energies_'+mom[:7]+'_dEq2.pdf', weights=weights_dEq2,switch=len(dEq2[-1])-len(dEq2_1[-1]))

                # Rescale the bootstrap errors to include the systematic errors
                avgE0.Stats()
                avgdEq1.Stats()
                avgdEq2.Stats()
                print(f"{comberr_dEq1/avgdEq1.Std=}")
                print(f"{avgdEq1.Std=}")

                for k in range(avgE0.nboot):
                    avgE0.values[k] = avgE0.Avg+(avgE0.values[k]-avgE0.Avg)*comberr_E0/avgE0.Std
                    avgdEq1.values[k] = avgdEq1.Avg+(avgdEq1.values[k]-avgdEq1.Avg)*comberr_dEq1/avgdEq1.Std
                    avgdEq2.values[k] = avgdEq2.Avg+(avgdEq2.values[k]-avgdEq2.Avg)*comberr_dEq2/avgdEq2.Std

                avgE0.Stats()
                avgdEq1.Stats()
                avgdEq2.Stats()
                print(f"{avgdEq1.Std=}")

                means.append([avgE0, avgdEq1, avgdEq2])
                mean_errs.append([systerrsq_E0, systerrsq_dEq1, systerrsq_dEq2])
                #mean_errs.append([avgE0, avgdEq1, avgdEq2])
        except:
            pass
    return np.array(means), np.array(mean_errs)

def Electric(pars, means,mean_errssq, meansM, mean_errssqM):
    GEdbl_wa = [] # doubly represented quark
    GEsgl_wa = [] # singly represented quark
    GEdbl_wa_err = []
    GEsgl_wa_err = []
    #------Multiply kinetic factors------
    for i, mom in enumerate(pars.momfold):
        GEdbl_wa.append(means[i,1]*means[i,0]*2*(means[0,0]*pars.almb)**(-1))
        GEsgl_wa.append(means[i,2]*means[i,0]*2*(means[0,0]*pars.almb)**(-1))
        # systematic error propagation
        GEdbl_wa_err.append(GEdbl_wa[-1].Avg*np.sqrt( mean_errssq[i,1]/(means[i,1].Avg)**2+mean_errssq[i,0]/(means[i,0].Avg)**2+mean_errssq[0,0]/(means[0,0].Avg)**2 ))
        GEsgl_wa_err.append(GEsgl_wa[-1].Avg*np.sqrt( mean_errssq[i,2]/(means[i,2].Avg)**2+mean_errssq[i,0]/(means[i,0].Avg)**2+mean_errssq[0,0]/(means[0,0].Avg)**2 ))
    # Important: This is because there are two FH insertions with q=(0,0,0) for these lattices.
    if pars.lattice=='Feyn-Hell_kp120900kp120900/' or pars.lattice=='Feyn-Hell_kp121095kp120512/' or pars.lattice=='Feyn-Hell_kp122005kp122005/' or pars.lattice=='Feyn-Hell_kp122130kp121756/' or pars.lattice=='Feyn-Hell_kp122078kp121859/':
        GEdbl_wa[0] = 0.5*GEdbl_wa[0] #Necessary as the 0-mom case has two FH insertions coupling to it for kp120900kp120900 (effectively doubling lambda)
        GEsgl_wa[0] = 0.5*GEsgl_wa[0]
        GEdbl_wa_err[0] = 0.5*GEdbl_wa_err[0]
        GEsgl_wa_err[0] = 0.5*GEsgl_wa_err[0]

    # #INFO: Plot of the energies against momentum squared
    Qsqrd = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2*(0.1973**2)/(pars.a**2) for q in pars.qval ])
    # pypl.figure("E", figsize=(9,6))
    # for j, error in enumerate(errors):
    #     if (not error):
    #         energies = np.array([E[j][i].Avg for i in range(len(E[j]))])
    #         energieserr = np.array([E[j][i].Std for i in range(len(E[j]))])
    #         pypl.errorbar(Qsqrd+j/7, energies, energieserr, fmt=markers[j], capsize=4, color=colors[j], label=labels[j])
    #         pypl.errorbar(Qsqrd+0.1, [m.Avg for m in means[:,0]], [m.Std for m in means[:,0]], fmt=markers[j+1], capsize=4, color=colors[j+1], label='weighted avg')
    #         # Compute total the error by adding stat error and syst error in quadrature, then plot with this error
    #         totalerr = np.array([ np.sqrt(means[i,0].Std**2 + mean_errssq[i,0] ) for i in range(len(means[:,0]))])
    #         pypl.errorbar(Qsqrd+0.1, [m.Avg for m in means[:,0]], totalerr, fmt=markers[j+1], capsize=4, color=colors[j+1])
    #         if len(E1[j])>1:
    #             energies = np.array([E1[j][i].Avg for i in range(len(E1[j]))])
    #             energieserr = np.array([E1[j][i].Std for i in range(len(E1[j]))])
    #             pypl.errorbar(Qsqrd+j/7, energies, energieserr, fmt='^', capsize=4, color=colors[j], label=labels[j][:-1]+'1')

    # pypl.legend(fontsize='x-small')
    # pypl.xlabel(r'$Q^2[\mathrm{GeV}^2]$',labelpad=14,fontsize=18)
    # #pypl.ylabel(r'$\textrm{a E_{p}}$',labelpad=5,fontsize=18)
    # pypl.ylabel(r'Energy',labelpad=5,fontsize=18)
    # pypl.title(r'Ground state and first excited state energy')
    # pypl.grid(True, alpha=0.4)
    # pypl.subplots_adjust(bottom=0.17, top=.91, left=0.14, right=0.93)
    # pypl.savefig(pars.savedir + 'Energies.pdf')
    # pypl.close()

    #INFO: Plot the energy shifts of the doubly and singly represented quark
    # plot with combined error (stat+syst)
    energysyst1 = np.array([ np.sqrt(means[i,1].Std**2 + mean_errssq[i,1] ) for i in range(len(means[:,1]))])
    energysyst2 = np.array([ np.sqrt(means[i,2].Std**2 + mean_errssq[i,2] ) for i in range(len(means[:,2]))])
    plotenergyshift(pars, Qsqrd,[m.Avg for m in means[:,1]],energysyst1,gamma=4,quark=1,ylim=(-0.1,1.5))
    plotenergyshift(pars, Qsqrd,[m.Avg for m in means[:,2]],energysyst2,gamma=4,quark=2,ylim=(-0.1,1.5))
    return GEdbl_wa, GEdbl_wa_err, GEsgl_wa, GEsgl_wa_err

def Magnetic(pars, means,mean_errssq, meansM, mean_errssqM):
    #-------------------------------------------------------------------------------------------------
    # gamma_2 operator
    #-------------------------------------------------------------------------------------------------
    GMdbl_wa = []
    GMsgl_wa = []
    GMdbl_wa_err = []
    GMsgl_wa_err = []
    #------Multiply kinetic factors------
    for i, mom in enumerate(pars.momfold[1:]):
        GMdbl_wa.append(meansM[i,1]*2*meansM[i,0]*2*(pars.qval[1:][i][0]*(2*np.pi/pars.L)*pars.almb)**(-1))
        GMsgl_wa.append(meansM[i,2]*2*meansM[i,0]*2*(pars.qval[1:][i][0]*(2*np.pi/pars.L)*pars.almb)**(-1))
        GMdbl_wa_err.append( abs(GMdbl_wa[-1].Avg)*np.sqrt( mean_errssqM[i,1]/(meansM[i,1].Avg)**2+mean_errssqM[i,0]/(meansM[i,0].Avg)**2 ))
        GMsgl_wa_err.append( abs(GMsgl_wa[-1].Avg)*np.sqrt( mean_errssqM[i,2]/(meansM[i,2].Avg)**2+mean_errssqM[i,0]/(meansM[i,0].Avg)**2 ))

    #INFO: the energy shift at zero momentum is ignored for gamma_2
    Qsqrd = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2*(0.1973**2)/(pars.a**2) for q in pars.qval ])
    xM=Qsqrd[1:]
    # plot with combined error (stat+syst)
    energysyst1 = np.array([ np.sqrt(meansM[i,1].Std**2 + mean_errssqM[i,1] ) for i in range(len(meansM[:,1]))])
    energysyst2 = np.array([ np.sqrt(meansM[i,2].Std**2 + mean_errssqM[i,2] ) for i in range(len(meansM[:,2]))])
    plotenergyshift(pars, Qsqrd[1:], [m.Avg for m in meansM[:,1]],energysyst1,gamma=2,quark=1,ylim=(-0.1,0.3))
    plotenergyshift(pars, Qsqrd[1:], [m.Avg for m in meansM[:,2]],energysyst2,gamma=2,quark=2,ylim=(-0.08,0.01))
    return GMdbl_wa, GMdbl_wa_err, GMsgl_wa, GMsgl_wa_err


def paulidirac(pars, means,mean_errssq, meansM, mean_errssqM, GEdbl_wa, GEsgl_wa, GMdbl_wa, GMsgl_wa):
    #------Construct Pauli & Dirac FF for quarks------
    Qsqrdlatt = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2 for q in pars.qval[1:] ])
    #Qsqrdlatt = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2*(0.1973**2)/(pars.a**2) for q in pars.qval[1:] ])
    F1dbl_wa = []
    F1sgl_wa = []
    F1dbl_wa_err = []
    F1sgl_wa_err = []

    # Pauli form factor F1
    F1dbl_wa.append(GEdbl_wa[0])
    F1sgl_wa.append(GEsgl_wa[0])
    F1dbl_wa[-1].Stats()
    F1sgl_wa[-1].Stats()
    F1dbl_wa_err.append(GEdbl_wa_err[0])
    F1sgl_wa_err.append(GEsgl_wa_err[0])
    for i, mom in enumerate(pars.momfold[1:]):
        F1dbl_wa.append( (GMdbl_wa[i]*Qsqrdlatt[i]*(4*means[0,0]**2)**(-1) + GEdbl_wa[i+1])*(1+ Qsqrdlatt[i]*(4*means[0,0]**2)**(-1))**(-1) )
        F1sgl_wa.append( (GMsgl_wa[i]*Qsqrdlatt[i]*(4*means[0,0]**2)**(-1) + GEsgl_wa[i+1])*(1+ Qsqrdlatt[i]*(4*means[0,0]**2)**(-1))**(-1) )
        F1dbl_wa[-1].Stats()
        F1sgl_wa[-1].Stats()

    # Dirac form factor F2
    F2dbl_wa = []
    F2sgl_wa = []
    for i, mom in enumerate(pars.momfold[1:]):
        F2dbl_wa.append( (GMdbl_wa[i] - GEdbl_wa[i+1])*(1+ Qsqrdlatt[i]*(4*means[0,0]**2)**(-1))**(-1) )
        F2sgl_wa.append( (GMsgl_wa[i] - GEsgl_wa[i+1])*(1+ Qsqrdlatt[i]*(4*means[0,0]**2)**(-1))**(-1) )
        F2dbl_wa[-1].Stats()
        F2sgl_wa[-1].Stats()
    return F1dbl_wa, F1sgl_wa, F2dbl_wa, F2sgl_wa
    
    
def savefiles(pars, Qsqrd, FFs):
    #------Save FF quark contributions to files------
    Qvalues = [ Qsqrd, Qsqrd[1:], Qsqrd, Qsqrd[1:], Qsqrd, Qsqrd[1:], Qsqrd, Qsqrd[1:] ]
    # FFs_wa = [ GEdbl_wa, GMdbl_wa, F1dbl_wa, F2dbl_wa, GEsgl_wa, GMsgl_wa, F1sgl_wa, F2sgl_wa ]
    FFnames_wa = [ 'GEdbl_wa', 'GMdbl_wa', 'F1dbl_wa', 'F2dbl_wa', 'GEsgl_wa', 'GMsgl_wa', 'F1sgl_wa', 'F2sgl_wa' ]
    for k,FF in enumerate(FFs):
        #btstrps = [b for b in FF]
        writeBS(pars.data_dir+FFnames_wa[k]+'.csv', Qvalues[k], FF)

def savefiles2(pars, Qsqrd, FF, FFname):
    #------Save FF quark contributions to files------
    # Qvalues = [ Qsqrd, Qsqrd[1:], Qsqrd, Qsqrd[1:], Qsqrd, Qsqrd[1:], Qsqrd, Qsqrd[1:] ]
    # FFnames_wa = [ 'GE_wa', 'GM_wa', 'F1_wa', 'F2_wa']
    writeBS(pars.data_dir+FFname+'.csv', Qsqrd, FF)
    # for k,FF in enumerate(FFs):
    #     #btstrps = [b for b in FF]
    #     writeBS(pars.data_dir+FFnames_wa[k]+'.csv', Qvalues[k], FF)

def normalization(pars, GEdbl_wa, GEsgl_wa, GMdbl_wa, GMsgl_wa):
    #--------Normalization--------
    Zdbl_wa = []
    Zsgl_wa = []
    Zdbl_wa = ( 2*(GEdbl_wa[0])**(-1) ) # Normalise doubly represented quark to 2
    Zsgl_wa = ( (GEsgl_wa[0])**(-1) )   # Normalise singly represented quark to 1
    Zdbl_wa_err = ( 2*(GEdbl_wa[0].Avg)**(-2)*GEdbl_wa_err[0] ) # Normalise doubly represented quark to 2
    Zsgl_wa_err = ( (GEsgl_wa[0].Avg)**(-2)*GEsgl_wa_err[0] )   # Normalise singly represented quark to 1

    for i, mom in enumerate(pars.momfold):
        GEdbl_wa[i] = GEdbl_wa[i]*Zdbl_wa
        GEsgl_wa[i] = GEsgl_wa[i]*Zsgl_wa
        GEdbl_wa[i].Stats()
        GEsgl_wa[i].Stats()
        GEdbl_wa_err[i] = abs(GEdbl_wa[i].Avg)*np.sqrt( (GEdbl_wa_err[i]/GEdbl_wa[i].Avg)**2 + ( Zdbl_wa_err/Zdbl_wa.Avg)**2 )
        GEsgl_wa_err[i] = abs(GEsgl_wa[i].Avg)*np.sqrt( (GEsgl_wa_err[i]/GEsgl_wa[i].Avg)**2 + ( Zsgl_wa_err/Zsgl_wa.Avg)**2 )

    #---Normalization---
    print(f"{GMdbl_wa=}")
    print(f"{GMsgl_wa=}")
    for i, mom in enumerate(pars.momfold[1:]):
        GMdbl_wa[i] = GMdbl_wa[i]*Zdbl_wa
        GMsgl_wa[i] = GMsgl_wa[i]*Zsgl_wa
        GMdbl_wa[i].Stats()
        GMsgl_wa[i].Stats()
        GMdbl_wa_err[i] = GMdbl_wa[i].Avg*np.sqrt( (GMdbl_wa_err[i]/GMdbl_wa[i].Avg)**2 + ( Zdbl_wa_err/Zdbl_wa.Avg)**2 )
        GMsgl_wa_err[i] = GMsgl_wa[i].Avg*np.sqrt( (GMsgl_wa_err[i]/GMsgl_wa[i].Avg)**2 + ( Zsgl_wa_err/Zsgl_wa.Avg)**2 )
    return GEdbl_wa, GEsgl_wa, GMdbl_wa, GMsgl_wa
        

def formfactors(pars, GEdbl_wa, GEsgl_wa, GMdbl_wa, GMsgl_wa, erroralex=False, error3=False, erroralexM=False, error3M=False):
    #----------------------------------------------------------------------
    # Form factors
    #----------------------------------------------------------------------
    Qsqrd = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2*(0.1973**2)/(pars.a**2) for q in pars.qval ])
    Qsq   = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2 for q in pars.qval[1:] ])
    xM=Qsqrd[1:]
    for i,bar in enumerate(pars.baryons): # Loop over both quark combinations that make baryons
        if pars.lattice=='Feyn-Hell_kp120900kp120900/':
            alex = '/home/mischa/Documents/PhD/analysis_results/Alex/vector-nucleon-'+bar+'--GE_0_Feynman-Hellmann.csv'
            alexGM = '/home/mischa/Documents/PhD/analysis_results/Alex/vector-nucleon-'+bar+'--GM_0_Feynman-Hellmann.csv'
            alexGEGM = '/home/mischa/Documents/PhD/analysis_results/Alex/vector-nucleon-'+bar+'--GEGM_0_Feynman-Hellmann.csv'
            alexF1 = '/home/mischa/Documents/PhD/analysis_results/Alex/vector-nucleon-'+bar+'--ffpauli_0_Feynman-Hellmann.csv'
            alexF2 = '/home/mischa/Documents/PhD/analysis_results/Alex/vector-nucleon-'+bar+'--ffdirac_0_Feynman-Hellmann.csv'
            try:
                with open(alex) as csvfile:
                    dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
                    rows = [x for x in dataread]
                    xA  = np.array([float(row[0]) for row in rows])
                    GEA = np.array([float(row[1]) for row in rows])
                    GEerrA = np.array([float(row[2]) for row in rows])
                    erroralex=False
            except IOError:
                erroralex=True
                print("Alex file not found")
            try:
                with open(alexGM) as csvfile:
                    dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
                    rows = [x for x in dataread]
                    xAM  = np.array([float(row[0]) for row in rows])
                    GMA = np.array([float(row[1]) for row in rows])
                    GMerrA = np.array([float(row[2]) for row in rows])
            except IOError:
                erroralexM=True
                print("Alex file not found")
            try:
                with open(alexGEGM) as csvfile:
                    dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
                    rows = [x for x in dataread]
                    xRA  = np.array([float(row[0]) for row in rows])
                    GRA = np.array([float(row[1]) for row in rows])
                    GRerrA = np.array([float(row[2]) for row in rows])
            except IOError:
                print("Alex ratio file not found")
            try:
                with open(alexF1) as csvfile:
                    dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
                    rows = [x for x in dataread]
                    x1A  = np.array([float(row[0]) for row in rows])
                    F1A = np.array([float(row[1]) for row in rows])
                    F1errA = np.array([float(row[2]) for row in rows])
            except IOError:
                print("Alex F1 file not found")
            try:
                with open(alexF2) as csvfile:
                    dataread = csv.reader(csvfile, delimiter=',', quotechar='|')
                    rows = [x for x in dataread]
                    x2A  = np.array([float(row[0]) for row in rows])
                    F2A = np.array([float(row[1]) for row in rows])
                    F2errA = np.array([float(row[2]) for row in rows])
            except IOError:
                print("Alex F2 file not found")
                
            if i==0:
                x3, GE3, GEerr3, error3 = threeptread('/home/mischa/Documents/PhD/analysis_results/3ptFF/vector-nucleon-ffGE_threept_2exp_dt2.csv', error3)
                x3M, GM3, GMerr3, error3M = threeptread('/home/mischa/Documents/PhD/analysis_results/3ptFF/vector-nucleon-ffGM_threept_2exp_dt2.csv', error3M)
            else:
                error3=True
                error3M=True
        else:
            erroralex=True
            error3=True
            erroralexM=True
            error3M=True

        #-----------------------------------------------------------------------------------------------
        # Electric (GE) Form Factor
        #-----------------------------------------------------------------------------------------------
        GEp_wa = []
        #GEp_wa_err = []
        for i, mom in enumerate(pars.momfold):
            GEp_wa.append(pars.baryons[bar][1][0]*GEdbl_wa[i] + pars.baryons[bar][1][1]*GEsgl_wa[i])
            GEp_wa[-1].Stats()
            #GEp_wa_err.append(np.sqrt( (pars.baryons[bar][1][0]*GEdbl_wa_err[i])**2 + (pars.baryons[bar][1][1]*GEsgl_wa_err[i])**2 ) )
        #INFO: Plot of the full electric form factor
        pypl.figure("GE", figsize=(9,6))
        pypl.errorbar(Qsqrd, [g.Avg for g in GEp_wa], [g.Std for g in GEp_wa], fmt=markers[-1], capsize=5, color=colors[-1], label='Weighted average')
        if (not erroralex):
            pypl.errorbar(xA+0.15, GEA, GEerrA, fmt='x', capsize=5, color=colors[2], label='FH 1 exp [Chambers 2017]')
        if (not error3):
            pypl.errorbar(x3+0.3, GE3, GEerr3, fmt='^', capsize=5, color=colors[4], label='3-pt. functions (2 exp fit)')
        pypl.legend(fontsize='small')
        pypl.xlabel(r'$Q^2[\mathrm{GeV}^2]$',labelpad=14,fontsize=18)
        pypl.ylabel(r'$G_{\mathrm{E,'+pars.baryons[bar][0]+'}}$',labelpad=5,fontsize=18)
        pypl.title(r'Form factor $G_{\mathrm{E}}$ for the '+bar)
        #pypl.ylim(-0.2,1.2)
        pypl.ylim(pars.baryons[bar][2][0])
        pypl.xticks(range(0,11))
        pypl.xlim(0,10)
        #pypl.xticks(range(0,math.ceil(Qsqrd[-1])+1))
        #pypl.xlim(0,math.ceil(Qsqrd[-1]))
        pypl.axhline(y=0,alpha=0.4,color='k')
        pypl.subplots_adjust(bottom=0.17, left=0.15)
        pypl.grid(True, alpha=0.4)
        #pypl.savefig(pars.savedir + 'GE_'+bar+'.pdf')
        pypl.savefig(pars.plot_dir[0][0]+'../GE_'+bar+'.pdf')
        pypl.close()
        #INFO: Save the data of the form factor
        savefiles2(pars, Qsqrd, GEp_wa, 'GE_'+bar)

        #-----------------------------------------------------------------------------------------------
        # Magnetic (GM) Form Factor
        #-----------------------------------------------------------------------------------------------
        GMp_wa = []
        GMp_wa_err = []
        for i, mom in enumerate(pars.momfold[1:]):
            GMp_wa.append(pars.baryons[bar][1][0]*GMdbl_wa[i] + pars.baryons[bar][1][1]*GMsgl_wa[i])
            GMp_wa[-1].Stats()

        #INFO: Plot of the full magnetic form factor
        pypl.figure("GM", figsize=(9,6))
        pypl.errorbar(xM, [g.Avg for g in GMp_wa], [g.Std for g in GMp_wa], fmt=markers[-1], capsize=5, color=colors[-1], label='Weighted average')
        if (not erroralexM):
            pypl.errorbar(xAM+0.15, GMA, GMerrA, fmt='x', capsize=5, color=colors[2], label='FH 1 exp [Chambers 2017]')
        if (not error3M):
            pypl.errorbar(x3M+0.3, GM3, GMerr3, fmt='^', capsize=5, color=colors[4], label='3-pt. functions (2 exp fit)')

        pypl.legend(fontsize='small')
        pypl.xlabel(r'$Q^2[\mathrm{GeV}^2]$',labelpad=14,fontsize=18)
        pypl.ylabel(r'$G_{\mathrm{M,'+pars.baryons[bar][0]+'}}$',labelpad=5,fontsize=18)
        pypl.title(r'Form factor $G_{\mathrm{M}}$ for the '+bar)
        #pypl.ylim(-0.1,1.1)
        pypl.ylim(pars.baryons[bar][2][1])
        #pypl.xticks(range(0,math.ceil(Qsqrd[-1])+1))
        #pypl.xlim(0,math.ceil(Qsqrd[-1]))
        pypl.xticks(range(0,11))
        pypl.xlim(0,10)
        pypl.axhline(y=0,alpha=0.4,color='k')
        pypl.subplots_adjust(bottom=0.17)
        pypl.grid(True, alpha=0.4)
        #pypl.savefig(pars.savedir + 'GM_'+bar+'.pdf')
        pypl.savefig(pars.plot_dir[0][0]+'../GM_'+bar+'.pdf')
        pypl.close()
        #INFO: Save the data of the form factor
        savefiles2(pars, Qsqrd[1:], GMp_wa, 'GM_'+bar)

        #-----------------------------------------------------------------------------------------------
        # Ratio of Electric Form Factor over Magnetic Form Factor
        #-----------------------------------------------------------------------------------------------
        if (not error3) and (not error3M):
            GEGM3 = GE3[1:]/GM3
            GEGMerr3 = np.zeros(len(GEGM3))
            for r, ffratio in enumerate(GEGM3):
                GEGMerr3[r] = abs(ffratio)*np.sqrt( (GEerr3[r+1]/GE3[r+1])**2 + (GMerr3[r]/GM3[r])**2 )
        GEGM_wa = []
        for i, mom in enumerate(pars.momfold[1:]):
            GEGM_wa.append( (pars.baryons[bar][1][0]*GEdbl_wa[i+1] + pars.baryons[bar][1][1]*GEsgl_wa[i+1])*(pars.baryons[bar][1][0]*GMdbl_wa[i] + pars.baryons[bar][1][1]*GMsgl_wa[i])**(-1) )
            GEGM_wa[-1].OldStats()

        # #TEST__Dealing with outliers__
        # n_bins = 100
        # for i, mom in enumerate(pars.momfold[1:]):
        #     for j, (error,errorM) in enumerate(zip(errors,errorsM)):
        #         if (not error and not errorM):
        #             fig, axs = pypl.subplots(1, 3, sharey=True, tight_layout=True, figsize=(12,6))
        #             axs[0].hist(GEp[j][i+1].values, bins=n_bins)
        #             axs[1].hist(GMp[j][i].values, bins=n_bins)
        #             axs[2].hist(GEGM[j][i].values, bins=n_bins)
        #             pypl.savefig(pars.savedir + 'test'+str(i)+str(j)+bar+'.pdf')
        #             pypl.close()
        # #_____________________________

        #INFO: Plot of the full proton electric over magnetic FF
        xM=Qsqrd[1:]
        pypl.figure("GEGM", figsize=(9,6))
        pypl.errorbar(xM, [g.Avg for g in GEGM_wa], [g.Std for g in GEGM_wa], fmt=markers[-1], capsize=5, color=colors[-1], label='FH Weighted average')
        if (not erroralexM):
            pypl.errorbar(xRA+0.15, GRA, GRerrA, fmt='x', capsize=5, color=colors[2], label='FH 1 exp [Chambers 2017]')
        if (not error3M):
            pypl.errorbar(x3M+0.3, GEGM3, GEGMerr3, fmt='^', capsize=5, color=colors[4], label='3-pt. functions (2 exp fit)')
        pypl.legend(fontsize='small')
        pypl.xlabel(r'$Q^2[\mathrm{GeV}^2]$',labelpad=14,fontsize=18)
        pypl.ylabel(r'$G_{\mathrm{E,'+pars.baryons[bar][0]+'}}/G_{\mathrm{M,'+pars.baryons[bar][0]+'}}$',labelpad=5,fontsize=18)
        pypl.title(r'Ratio of form factors $G_{\mathrm{E}}$/$G_{\mathrm{M}}$ for the '+bar)
        pypl.ylim(pars.baryons[bar][2][2])
        #pypl.xticks(range(0,math.ceil(Qsqrd[-1])+1))
        #pypl.xlim(0,math.ceil(Qsqrd[-1]))
        pypl.xticks(range(0,11))
        pypl.xlim(0,10)
        pypl.axhline(y=0,alpha=0.4,color='k')
        pypl.subplots_adjust(bottom=0.17)
        pypl.grid(True, alpha=0.4)
        pypl.savefig(pars.plot_dir[0][0]+'../GEGM_'+bar+'.pdf')
        #pypl.savefig(pars.savedir + 'GEGM'+bar+'.pdf')
        pypl.close()
        #INFO: Save the data of the form factor
        savefiles2(pars, Qsqrd[1:], GEGM_wa, 'GEGM_'+bar)

        #-----------------------------------------------------------------------------------------------
        # Pauli (F1) Form Factor
        #-----------------------------------------------------------------------------------------------
        #---Create Pauli form factor---
        F1p_wa = []
        F1p_wa.append(GEp_wa[0])
        F1p_wa[-1].Stats()
        for i, mom in enumerate(pars.momfold[1:]):
            F1p_wa.append( (GMp_wa[i]*Qsq[i]*(4*means[0,0]**2)**(-1) + GEp_wa[i+1])*(1+ Qsq[i]*(4*means[0,0]**2)**(-1))**(-1) )
            F1p_wa[-1].Stats()
        #---plot the Pauli form factor---
        xF1=Qsqrd
        pypl.figure("F1p", figsize=(9,6))
        pypl.errorbar(xF1+1/10, [f.Avg for f in F1p_wa], [f.Std for f in F1p_wa], fmt=markers[-1], capsize=5, color=colors[-1], label='FH weighted average')
        if (not erroralexM):
            pypl.errorbar(x1A+0.2, F1A, F1errA, fmt='x', capsize=5, color=colors[2], label='FH 1 exp [Chambers 2017]')
        pypl.legend(fontsize='small')
        pypl.xlabel(r'$Q^2[\mathrm{GeV}^2]$',labelpad=14,fontsize=18)
        pypl.ylabel(r'$F_{\mathrm{1,'+pars.baryons[bar][0]+'}}$',labelpad=5,fontsize=18)
        pypl.title(r'Form factor $F_{\mathrm{1}}$ for the '+bar)
        pypl.ylim(pars.baryons[bar][2][3])
        #pypl.xticks(range(0,math.ceil(Qsqrd[-1])+1))
        #pypl.xlim(0,math.ceil(Qsqrd[-1]))
        pypl.xticks(range(0,11))
        pypl.xlim(0,10)
        pypl.axhline(y=0,alpha=0.4,color='k')
        pypl.subplots_adjust(bottom=0.17)
        pypl.grid(True, alpha=0.4)
        pypl.savefig(pars.plot_dir[0][0]+'../PauliF1_'+bar+'.pdf')
        #pypl.savefig(pars.savedir + 'PauliF1'+bar+'.pdf')
        pypl.close()
        #INFO: Save the data of the form factor
        savefiles2(pars, Qsqrd, F1p_wa, 'F1_'+bar)

        #-------------------------------------------------------------------------------------------------
        # Dirac (F2) Form Factor
        #-------------------------------------------------------------------------------------------------
        #---Create Dirac form factor---
        F2p_wa=[]
        for i, mom in enumerate(pars.momfold[1:]):
            F2p_wa.append( (GMp_wa[i] - GEp_wa[i+1])*(1+ Qsq[i]*(4*means[0,0]**2)**(-1))**(-1) )
            F2p_wa[-1].Stats()

        xF2=Qsqrd[1:]
        pypl.figure("F2p", figsize=(9,6))
        pypl.errorbar(xF2+1/10, [f.Avg for f in F2p_wa], [f.Std for f in F2p_wa], fmt=markers[-1], capsize=5, color=colors[-1], label='FH weighted average')
        if (not erroralexM):
            pypl.errorbar(x2A+0.2, F2A, F2errA, fmt='x', capsize=5, color=colors[2], label='FH 1 exp [Chambers 2017]')

        pypl.legend(fontsize='small')
        pypl.xlabel(r'$Q^2[\mathrm{GeV}^2]$',labelpad=14,fontsize=18)
        pypl.ylabel(r'$F_{\mathrm{2,'+pars.baryons[bar][0]+'}}$',labelpad=5,fontsize=18)
        pypl.title(r'Form factor $F_{\mathrm{2}}$ for the '+bar)
        pypl.ylim(pars.baryons[bar][2][4])
        #pypl.xticks(range(0,math.ceil(Qsqrd[-1])+1))
        #pypl.xlim(0,math.ceil(Qsqrd[-1]))
        pypl.xticks(range(0,11))
        pypl.xlim(0,10)
        pypl.axhline(y=0,alpha=0.4,color='k')
        pypl.subplots_adjust(bottom=0.17)
        pypl.grid(True, alpha=0.4)
        #pypl.savefig(pars.savedir + 'DiracF2'+bar+'.pdf')
        pypl.savefig(pars.plot_dir[0][0]+'../DiracF2_'+bar+'.pdf')
        pypl.close()
        #INFO: Save the data of the form factor
        savefiles2(pars, Qsqrd[1:], F2p_wa, 'F2_'+bar)

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
        # pars=pr.params()
        # pars.makedirs()
        pypl.rc('font', size=18, **{'family': 'DejaVu Sans'})
        pypl.rc('text', usetex=True)
        for bar in pars.baryons: print(bar)

        #Read in the different fit results and make the weighted average of them.
        means, mean_errssq = weighted_avg(pars, op=1, exclude0=False)
        meansM, mean_errssqM = weighted_avg(pars, op=0, exclude0=True)

        GEdbl_wa, GEdbl_wa_err, GEsgl_wa, GEsgl_wa_err = Electric(pars, means,mean_errssq, meansM, mean_errssqM)
        GMdbl_wa, GMdbl_wa_err, GMsgl_wa, GMsgl_wa_err = Magnetic(pars, means,mean_errssq, meansM, mean_errssqM)
        F1dbl_wa, F1sgl_wa, F2dbl_wa, F2sgl_wa = paulidirac(pars, means,mean_errssq, meansM, mean_errssqM, GEdbl_wa, GEsgl_wa, GMdbl_wa, GMsgl_wa)
        Qsqrd = np.array([ np.dot(q,q)*(2*np.pi/pars.L)**2*(0.1973**2)/(pars.a**2) for q in pars.qval ])
        FFs = [ GEdbl_wa, GMdbl_wa, F1dbl_wa, F2dbl_wa, GEsgl_wa, GMsgl_wa, F1sgl_wa, F2sgl_wa ]
        savefiles(pars, Qsqrd, FFs)
        GEdbl_wa, GEsgl_wa, GMdbl_wa, GMsgl_wa = normalization(pars, GEdbl_wa, GEsgl_wa, GMdbl_wa, GMsgl_wa)
        formfactors(pars, GEdbl_wa, GEsgl_wa, GMdbl_wa, GMsgl_wa)
    else:
        print("arg[1] = kappa")
        print("arg[2] = sinktype")
