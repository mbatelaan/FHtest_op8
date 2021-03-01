# Read the fit files from the oneexp fit script and Plot the effective energy shift for multiple lambda values
# also plot value of the weighted average of the fits with the fitrange of the fit which got the highest weighting.
import numpy as np
from scipy.special import gamma, gammainc, gammaincc
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
    print(f"{np.shape(nucleon)=}")
    print(f"{np.shape(nucleons_raw)=}")
    print(f"{np.shape(nucleon[0])=}")
    print(f"{np.shape(nucleon[1])=}")

    nucleons = [[],[],[]]
    for num in range(len(nucleons_raw[0])):
        for type,nu in enumerate(nucleons):
            nu.append([])
            for t in range(len(nucleons_raw[type][num])):
                nu[num].append(BootStrap(len(nucleons_raw[type][num][t]), pars.confidence))
                nu[num][-1].values=nucleons_raw[type][num][t]
                # nu[num][-1].values=rdata.loc[t,:]
                nu[num][-1].Stats()

    print(f"{np.shape(nucleons)=}")
    return nucleons

def weights(dof, chisq, derrors):
    """Take a list of degrees of freedom and of chi-squared values and errors of the fit and return the weights for each fit"""
    pf =[]
    for d, chi in zip(dof, chisq):
        pf.append(gammaincc(d/2,chi/2))
    denominator = sum(np.array(pf)*np.array([d**(-2) for d in derrors]))
    weights=[]
    for p, de in zip(pf, derrors):
        weights.append(p*(de**(-2))/denominator)
    return weights


def oneexpreader(pars,opnum):
    op = pars.operators[opnum]
    # read the dictionary of the energy fits from the pickle file
    energyfitfile = pars.data_dir + 'energyBS_' + op +'_'+ pars.momfold[pars.momentum][:7]+'_'+pars.fit + pars.snkfold[pars.sink] + '_' + pars.lmbstring + '.pkl'
    # energyfitfile = pars.data_dir + 'energyBS_' + op +'_'+ pars.momfold[pars.momentum][:7]+'_'+pars.fit + pars.snkfold[pars.sink]  + '.pkl'
    with open(energyfitfile, 'rb') as fileout:
        dictenergy = pickle.load(fileout)

    # read the dictionary of the energy shift fits from the pickle file
    dictenergyshift =[]
    energyshiftfiles = [ pars.data_dir + 'energyshiftfit_'+op+'_q'+str(quark+1) + pars.momfold[pars.momentum][:7]+'_'+pars.fit+pars.snkfold[pars.sink]+'_'+pars.lmbstring+'.pkl' for quark in range(2) ]
    # energyshiftfiles = [ pars.data_dir + 'energyshiftfit_' + op +'_q'+str(quark+1) + pars.momfold[pars.momentum][:7]+'_'+pars.fit + pars.snkfold[pars.sink]+'.pkl' for quark in range(2) ]
    for i, outfile in enumerate(energyshiftfiles):
        with open(outfile, 'rb') as fileout:
            dictenergyshift.append(pickle.load(fileout))
    return dictenergy, dictenergyshift

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
    for index0, tmax in enumerate(np.arange(pars.tmaxmin, pars.tmaxmax)):
        for index, tmin in enumerate(np.arange(pars.tminmin, pars.tminmax)):
            print("\n------------------------------")
            print("\nquark1 range = ",tmin,"-",tmax)
            print("quark2 range = ",tmin,"-",tmax)
            if (tmax-tmin)>fitfnc.npar+1: #Check whether there are more data points than fit parameters
                #---------------------------------------------
                #INFO: Fit to the 2pt. function to get a value for the energy
                tdata = np.arange(tmin,tmax)
                fitparam,BSen0,redchisq = stats.BootFitclass3(fitfnc.eval, pars.initpar0[pars.momentum], tdata, np.array(unpert_2p)[tdata], bounds=pars.bounds2pt[:fitfnc.npar], fullcov=True)

                # Save the fit results in a dictionary
                dictenergy[index] = {}
                dictenergy[index]['tmin'] = tmin
                dictenergy[index]['tmax'] = tmax
                dictenergy[index]['A0'] = BSen0[0].values
                dictenergy[index]['E0'] = BSen0[1].values
                dictenergy[index]['chisq'] = redchisq
                dictenergy[index]['dof'] = len(tdata)-len(pars.initpar0[pars.momentum])

                energyfit.append(BSen0)
                #---------------------------------------------
                #INFO: Fit to the ratio of correlators to get the energy shift
                for quark in np.arange(2):
                    #INFO: Fit over all bootstraps for 1-exp
                    fitboot,BSen1,redchisq1=stats.BootFitclass3(fitfnc.eval, pars.initpar1[pars.momentum], tdata, np.array(ratios[quark])[tdata],bounds=fitfnc.bounds,time=True, fullcov=True)
                    dictenergyshift[quark][index] = {}
                    dictenergyshift[quark][index]['tmin'] = tmin
                    dictenergyshift[quark][index]['tmax'] = pars.tmaxratio[opnum][0][pars.momentum]
                    dictenergyshift[quark][index]['A0'] = BSen1[0].values
                    dictenergyshift[quark][index]['E0'] = BSen1[1].values
                    dictenergyshift[quark][index]['chisq'] = redchisq1.copy()
                    dictenergyshift[quark][index]['dof'] = len(tdata)-len(pars.initpar1[pars.momentum])
                    # Save bootstrap objects of parameters for the plot
                    fitlist1[quark].append(BSen1)

    # Save the energy fit to the pickle file 
    energyfitfile = pars.data_dir + 'energyBS_' + op +'_'+ pars.momfold[pars.momentum][:7]+'_'+pars.fit + pars.snkfold[pars.sink]  + '.pkl'
    with open(energyfitfile, 'wb') as fileout:
        pickle.dump(dictenergy, fileout)

    # Save the energy shift fit to the pickle file
    energyshiftfiles = [ pars.data_dir + 'energyshiftfit_' + op +'_q'+str(quark+1) + pars.momfold[pars.momentum][:7]+'_'+pars.fit + pars.snkfold[pars.sink]+'.pkl' for quark in range(2) ]
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
    op      = pars.operators[opnum]
    fold    = pars.plot_dir[opnum][0]+'../'
    time    = np.arange(0,len(ratios1[quarknum]))
    efftime = time[:-1]+0.5

    #INFO: Calculate weights for each fit here.(in functions)
    #lp025
    weights_dE_lp025 = weights([dictenergyshift1[i]['dof'] for i in dictenergyshift1]
                            , [dictenergyshift1[i]['chisq'] for i in dictenergyshift1]
                            , [np.std(dictenergyshift1[i]['E0'], ddof=1) for i in dictenergyshift1] )
    print(f"{weights_dE_lp025=}")
    print(f"{np.shape(np.array([dictenergyshift1[i]['E0'] for i in dictenergyshift1] ))=}")
    weighted_dE_lp025 = (np.array(weights_dE_lp025)*np.array([dictenergyshift1[i]['E0'] for i in dictenergyshift1]).T).T
    print(f"{np.shape(weighted_dE_lp025)=}")
    avgdE_lp025 = np.sum(weighted_dE_lp025, axis=0)
    meanavgdE_lp025 = np.average(avgdE_lp025)
    staterrsq_dE = np.std(avgdE_lp025,ddof=1)**2
    print(f"{np.shape(avgdE_lp025)=}")
    print(f"{np.average(avgdE_lp025)=}")
    print(f"{np.std(avgdE_lp025, ddof=1)=}")
    # The square of the differences of each fit mean to the weighted average mean
    mean_diff = np.array([(np.average(dictenergyshift1[i]['E0'])-meanavgdE_lp025)**2 for i in dictenergyshift1])
    # The systematic error due to taking multiple fit results in the average
    systerrsq_dE = np.sum(weights_dE_lp025*mean_diff)
    # The combined error includes the statistical and systematic uncertainty
    comberr_dE = np.sqrt(staterrsq_dE+systerrsq_dE)
    # Rescale the bootstrap values such that the standard deviation of the bootstrap values is equal to the combined error, without changing the mean value of the bootstraps
    for nboot in range(len(avgdE_lp025)):
        avgdE_lp025[nboot] = meanavgdE_lp025+(avgdE_lp025[nboot]-meanavgdE_lp025)*comberr_dE/(staterrsq_dE**(0.5))
    print(f"{np.average(avgdE_lp025)=}")
    print(f"{np.std(avgdE_lp025, ddof=1)=}")

    #lp01
    weights_dE_lp01 = weights([dictenergyshift2[i]['dof'] for i in dictenergyshift2]
                            , [dictenergyshift2[i]['chisq'] for i in dictenergyshift2]
                            , [np.std(dictenergyshift2[i]['E0'], ddof=1) for i in dictenergyshift2] )
    weighted_dE_lp01 = (np.array(weights_dE_lp01)*np.array([dictenergyshift2[i]['E0'] for i in dictenergyshift2]).T).T
    avgdE_lp01 = np.sum(weighted_dE_lp01, axis=0)
    meanavgdE_lp01 = np.average(avgdE_lp01)
    staterrsq_dE = np.std(avgdE_lp01,ddof=1)**2
    # The square of the differences of each fit mean to the weighted average mean
    mean_diff = np.array([(np.average(dictenergyshift2[i]['E0'])-meanavgdE_lp01)**2 for i in dictenergyshift2])
    # The systematic error due to taking multiple fit results in the average
    systerrsq_dE = np.sum(weights_dE_lp01*mean_diff)
    # The combined error includes the statistical and systematic uncertainty
    comberr_dE = np.sqrt(staterrsq_dE+systerrsq_dE)
    # Rescale the bootstrap values such that the standard deviation of the bootstrap values is equal to the combined error, without changing the mean value of the bootstraps
    for nboot in range(len(avgdE_lp01)):
        avgdE_lp01[nboot] = meanavgdE_lp01+(avgdE_lp01[nboot]-meanavgdE_lp01)*comberr_dE/(staterrsq_dE**(0.5))
    print(f"{np.average(avgdE_lp01)=}")
    print(f"{np.std(avgdE_lp01, ddof=1)=}")

    
    #lp025
    # Get the index of the fit with the chi-squared value closest to 1
    # index = min(dictenergyshift1, key=(lambda x : abs(dictenergyshift1[x]['chisq']-1) ) )
    bestweight1 = np.argsort(weights_dE_lp025)[-1]
    # print(f"{bestweight[0]=}")
    # print(f"{weights_dE_lp025[bestweight[0]]=}")
    # print(f"{bestweight[-1]=}")
    # print(f"{weights_dE_lp025[bestweight[-1]]=}")
    # print(f"{dictenergyshift1[bestweight[-1]]['chisq']=}")
    # print(f"{dictenergyshift1[bestweight[-1]]['tmin']=}")
    # print(f"{dictenergyshift1[bestweight[-1]]['tmax']=}") 
    index = bestweight1
    fitrange   = np.arange(dictenergyshift1[index]['tmin'], dictenergyshift1[index]['tmax'])
    redchisq   = dictenergyshift1[index]['chisq']
    # fitBS  = np.array([ stats.effmass(fitfnc.eval(time[:pars.xlim], [dictenergyshift1[index]['A0'][nbo], dictenergyshift1[index]['E0'][nbo]] )) for nbo in range(len(dictenergyshift1[index]['A0'])) ])
    fitBS=np.array([avgdE_lp025 for  i in time[:pars.xlim]]).T
    print(f"{np.shape(fitBS)=}")
    fitBSavg   = np.average(fitBS,axis=0)
    fitBSstd   = fitBS.std(axis=0, ddof=1)
    fitBSlower = fitBSavg - fitBSstd
    fitBSupper = fitBSavg + fitBSstd

    #lp01
    # Get the index of the fit with the chi-squared value closest to 1
    # index2 = min(dictenergyshift2, key=(lambda x : abs(dictenergyshift2[x]['chisq']-1) ) )
    bestweight2 = np.argsort(weights_dE_lp01)[-1]
    print(f"{bestweight2=}")
    print(f"{weights_dE_lp01[bestweight2]=}")
    print(f"{dictenergyshift2[bestweight2]['chisq']=}")
    print(f"{dictenergyshift2[bestweight2]['dof']=}")
    print(f"{dictenergyshift2[bestweight2]['tmin']=}")
    print(f"{dictenergyshift2[bestweight2]['tmax']=}") 
    index2 = bestweight2
    fitrange2   = np.arange(dictenergyshift2[index2]['tmin'], dictenergyshift2[index2]['tmax'])
    redchisq2   = dictenergyshift2[index2]['chisq']
    # fitBS2=np.array([stats.effmass(fitfnc.eval(time[:pars.xlim], [dictenergyshift2[index2]['A0'][nbo], dictenergyshift2[index2]['E0'][nbo]] )) for nbo in range(len(dictenergyshift2[index2]['A0'])) ])
    # fitBS2=np.array([avgdE_lp01*time[:pars.xlim] for nbo in range(len(avgdE_lp01)) ])
    fitBS2=np.array([avgdE_lp01 for  i in time[:pars.xlim]]).T
    print(f"{np.shape(fitBS2)=}")
    fitBSavg2   = np.average(fitBS2,axis=0)
    fitBSstd2   = fitBS2.std(axis=0, ddof=1)
    fitBSlower2 = fitBSavg2 - fitBSstd2
    fitBSupper2 = fitBSavg2 + fitBSstd2

    #lp025
    yavgeff1 = np.array([y.Avg for y in stats.effectivemass(ratios1[quarknum])])
    yerreff1 = np.array([y.Std for y in stats.effectivemass(ratios1[quarknum])])
    #lp01
    yavgeff2 = np.array([y.Avg for y in stats.effectivemass(ratios2[quarknum])])
    yerreff2 = np.array([y.Std for y in stats.effectivemass(ratios2[quarknum])])
    
    pypl.figure(figsize=(9,6))
    #lp025
    pypl.errorbar(efftime[:pars.xlim], yavgeff1[:pars.xlim], yerreff1[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='k', marker=pars.markers[pars.sink], markerfacecolor='none', label=r'$\lambda=0.025$')
    fitrangelp025 = np.arange(pars.tminmin,pars.tmaxmax)
    pypl.plot(fitrangelp025+0.5,fitBSavg[fitrangelp025], linestyle='-', color='k')#, label="OneExp fit $\chi^2=${:0.2f}".format(redchisq))
    pypl.fill_between(fitrangelp025+0.5, fitBSlower[fitrangelp025], fitBSupper[fitrangelp025], color='k', alpha=0.3, linewidth=0)
    #lp01
    pypl.errorbar(efftime[:pars.xlim], yavgeff2[:pars.xlim], yerreff2[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color='b', marker=pars.markers[pars.sink], markerfacecolor='none', label=r'$\lambda=0.01$')
    # pypl.plot(fitrange2[:-1]+0.5,fitBSavg2[fitrange2[:-1]], linestyle='-', color=pars.colors[2], label="OneExp fit $\chi^2=${:0.2f}".format(redchisq2))
    # pypl.fill_between(fitrange2[:-1]+0.5, fitBSlower2[fitrange2[:-1]], fitBSupper2[fitrange2[:-1]], color=pars.colors[2], alpha=0.3, linewidth=0)
    fitrangelp01 = np.arange(pars.tminmin,pars.tmaxmax)
    pypl.plot(fitrangelp01+0.5,fitBSavg2[fitrangelp01], linestyle='-', color='b')#, label="OneExp fit $\chi^2=${:0.2f}".format(redchisq2))
    pypl.fill_between(fitrangelp01+0.5, fitBSlower2[fitrangelp01], fitBSupper2[fitrangelp01], color='b', alpha=0.3, linewidth=0)

    # indicate timeranges
    pypl.fill_between(np.arange(0,pars.tminmin+1), -100, 100, color='k', alpha=0.2, linewidth=0)
    pypl.fill_between(np.arange(pars.tmaxmax,32), -100, 100, color='k', alpha=0.2, linewidth=0)
    # pypl.fill_between(np.arange(-1,pars.tminmax), -100, 100, color='k', alpha=0.2, linewidth=0)
    # pypl.fill_between(np.arange(pars.tmaxmin,32), -100, 100, color='k', alpha=0.2, linewidth=0)
    
    pypl.legend(fontsize='x-small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(r'$dE_{'+['u','d'][quarknum]+', \gamma_{'+op[1:]+'}}$',labelpad=5,fontsize=18)
    pypl.title(r'Energy shift '+pars.momfold[pars.momentum][:7]+r', $\gamma_{'+op[1:]+r'}$')
    pypl.ylim([[-0.02,0.18],[-0.05,0.02]][quarknum])
    # pypl.ylim([[-0.0002,0.0018],[-0.0005,0.0002]][quarknum])
    pypl.xlim(0,28)
    pypl.legend(fontsize='small')
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'Eff_dEshift_' + pars.momfold[pars.momentum][:7] + pars.snkfold[pars.sink] +'_'+op+'q'+str(quarknum+1)+'.pdf')
    pypl.show()
    pypl.close()
    return 

def plotratiosloop(ratiolist, dictenergyshift, fitfnc, pars, opnum, quarknum):
    """ Plot the effective mass of the ratio of correlators for both lambdas and plot their fits """
    op      = pars.operators[opnum]
    fold    = pars.plot_dir[opnum][0]+'../'
    time    = np.arange(0,len(ratiolist[-1][quarknum]))
    efftime = time[:-1]+0.5

    pypl.figure(figsize=(9,6))

    for lmbnum, lmb in enumerate(pars.lmblist):
        print(f"{lmb=}")
        #INFO: Calculate weights for each fit here.(in functions)
        weights_dE = weights([dictenergyshift[lmbnum][i]['dof'] for i in dictenergyshift[lmbnum]]
                                , [dictenergyshift[lmbnum][i]['chisq'] for i in dictenergyshift[lmbnum]]
                                , [np.std(dictenergyshift[lmbnum][i]['E0'], ddof=1) for i in dictenergyshift[lmbnum]] )
        weighted_dE = (np.array(weights_dE)*np.array([dictenergyshift[lmbnum][i]['E0'] for i in dictenergyshift[lmbnum]]).T).T
        avgdE = np.sum(weighted_dE, axis=0)
        meanavgdE = np.average(avgdE)
        # print(f"{np.array([dictenergyshift[lmbnum][i]['E0'] for i in dictenergyshift[lmbnum]])=}")
        # print(f"{meanavgdE=}")
        staterrsq_dE = np.std(avgdE,ddof=1)**2
        # The square of the differences of each fit mean to the weighted average mean
        mean_diff = np.array([(np.average(dictenergyshift[lmbnum][i]['E0'])-meanavgdE)**2 for i in dictenergyshift[lmbnum]])
        # The systematic error due to taking multiple fit results in the average
        systerrsq_dE = np.sum(weights_dE*mean_diff)
        # The combined error includes the statistical and systematic uncertainty
        comberr_dE = np.sqrt(staterrsq_dE+systerrsq_dE)
        # Rescale the bootstrap values such that the standard deviation of the bootstrap values is equal to the combined error, without changing the mean value of the bootstraps
        for nboot in range(len(avgdE)):
            avgdE[nboot] = meanavgdE+(avgdE[nboot]-meanavgdE)*comberr_dE/(staterrsq_dE**(0.5))

        # Get the index of the fit with the chi-squared value closest to 1
        bestweight1 = np.argsort(weights_dE)[-1]
        index = bestweight1
        fitrange   = np.arange(dictenergyshift[lmbnum][index]['tmin'], dictenergyshift[lmbnum][index]['tmax'])
        redchisq   = dictenergyshift[lmbnum][index]['chisq']
        fitBS=np.array([avgdE for  i in time[:pars.xlim]]).T
        fitBSavg   = np.average(fitBS,axis=0)
        fitBSstd   = fitBS.std(axis=0, ddof=1)
        fitBSlower = fitBSavg - fitBSstd
        fitBSupper = fitBSavg + fitBSstd

        yavgeff1 = np.array([y.Avg for y in stats.effectivemass(ratiolist[lmbnum][quarknum])])
        yerreff1 = np.array([y.Std for y in stats.effectivemass(ratiolist[lmbnum][quarknum])])

        pypl.errorbar(efftime[:pars.xlim], yavgeff1[:pars.xlim]/pars.lmbvals[lmbnum], yerreff1[:pars.xlim]/pars.lmbvals[lmbnum], fmt='.', capsize=4, elinewidth=1, color=pars.colors[lmbnum], marker=pars.markers[pars.sink], markerfacecolor='none', label=r'$\lambda='+str(pars.lmbvals[lmbnum])+'$')
        # fitrange = np.arange(pars.tminmin,pars.tmaxmax)
        pypl.plot(fitrange+0.5,fitBSavg[fitrange]/pars.lmbvals[lmbnum], linestyle='-', color=pars.colors[lmbnum])#, label="OneExp fit $\chi^2=${:0.2f}".format(redchisq))
        pypl.fill_between(fitrange+0.5, fitBSlower[fitrange]/pars.lmbvals[lmbnum], fitBSupper[fitrange]/pars.lmbvals[lmbnum], color=pars.colors[lmbnum], alpha=0.3, linewidth=0)

    # indicate timeranges
    pypl.fill_between(np.arange(0,pars.tminmin+1), -100, 100, color='k', alpha=0.2, linewidth=0)
    pypl.fill_between(np.arange(pars.tmaxmax,32), -100, 100, color='k', alpha=0.2, linewidth=0)
    # pypl.fill_between(np.arange(-1,pars.tminmax), -100, 100, color='k', alpha=0.2, linewidth=0)
    # pypl.fill_between(np.arange(pars.tmaxmin,32), -100, 100, color='k', alpha=0.2, linewidth=0)

    pypl.legend(fontsize='x-small')
    pypl.xlabel(r'$\textrm{t/a}$',labelpad=14,fontsize=18)
    pypl.ylabel(r'$\Delta E_{'+['u','d'][quarknum]+', \gamma_{'+op[1:]+'}}/\lambda$',labelpad=5,fontsize=18)
    pypl.title(r'Energy shift '+pars.momfold[pars.momentum][:7]+r', $\gamma_{'+op[1:]+r'}$')
    pypl.ylim([[-5,-1.9],[-3.3,-0.9]][quarknum])
    # pypl.ylim([[-2.5,0.01],[-1.3,0.01]][quarknum])
    # pypl.ylim([[-0.12,0.01],[-0.12,0.01]][quarknum])
    # pypl.ylim([[-0.02,0.18],[-0.05,0.02]][quarknum])
    # pypl.ylim([[-0.0002,0.0018],[-0.0005,0.0002]][quarknum])
    pypl.xlim(0,28)
    pypl.legend(fontsize='small')
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'Eff_dEshift_' + pars.momfold[pars.momentum][:7] + pars.snkfold[pars.sink] +'_'+op+'q'+str(quarknum+1)+'.pdf')
    pypl.close()
    
    return 


def plotlambdadep(dictenergyshift, pars, opnum, quarknum):
    """ Plot the effective mass of the ratio of correlators for both lambdas and plot their fits """
    op      = pars.operators[opnum]
    fold    = pars.plot_dir[opnum][0]+'../'

    pypl.figure(figsize=(9,6))
    energies = []
    energieserr = []
    for lmbnum, lmb in enumerate(pars.lmblist):
        print(f"{lmb=}")
        #INFO: Calculate weights for each fit here.(in functions)
        weights_dE = weights([dictenergyshift[lmbnum][i]['dof'] for i in dictenergyshift[lmbnum]]
                                , [dictenergyshift[lmbnum][i]['chisq'] for i in dictenergyshift[lmbnum]]
                                , [np.std(dictenergyshift[lmbnum][i]['E0'], ddof=1) for i in dictenergyshift[lmbnum]] )
        weighted_dE = (np.array(weights_dE)*np.array([dictenergyshift[lmbnum][i]['E0'] for i in dictenergyshift[lmbnum]]).T).T
        avgdE = np.sum(weighted_dE, axis=0)
        meanavgdE = np.average(avgdE)
        staterrsq_dE = np.std(avgdE,ddof=1)**2
        # The square of the differences of each fit mean to the weighted average mean
        mean_diff = np.array([(np.average(dictenergyshift[lmbnum][i]['E0'])-meanavgdE)**2 for i in dictenergyshift[lmbnum]])
        # The systematic error due to taking multiple fit results in the average
        systerrsq_dE = np.sum(weights_dE*mean_diff)
        # The combined error includes the statistical and systematic uncertainty
        comberr_dE = np.sqrt(staterrsq_dE+systerrsq_dE)
        # Rescale the bootstrap values such that the standard deviation of the bootstrap values is equal to the combined error, without changing the mean value of the bootstraps
        for nboot in range(len(avgdE)):
            avgdE[nboot] = meanavgdE+(avgdE[nboot]-meanavgdE)*comberr_dE/(staterrsq_dE**(0.5))
        stdavgdE = np.std(avgdE, ddof=1)
        energies.append(meanavgdE)
        energieserr.append(stdavgdE)

    # energies = np.array(energies)/np.array(pars.lmbvals)
    # energieserr = np.array(energieserr)/np.array(pars.lmbvals)
    pypl.errorbar(pars.lmbvals,energies,energieserr, fmt='.', capsize=4, elinewidth=1, color=pars.colors[0], marker=pars.markers[1], markerfacecolor='none')
    # pypl.legend(fontsize='x-small')
    pypl.xlabel(r'$\lambda$',labelpad=14,fontsize=18)
    pypl.ylabel(r'$\Delta E_{'+['u','d'][quarknum]+', \gamma_{'+op[1:]+'}}$',labelpad=5,fontsize=18)
    pypl.title(r'Energy shift '+pars.momfold[pars.momentum][:7]+r', $\gamma_{'+op[1:]+r'}$')
    # pypl.ylim([[-5,-1.9],[-3.3,-0.9]][quarknum])
    # pypl.xlim(0,1e-1)
    # pypl.xscale("log")
    pypl.legend(fontsize='small')
    pypl.grid(True, alpha=0.4)
    ax = pypl.gca()
    pypl.subplots_adjust(bottom=0.17, top=.91, left=0.16, right=0.93)
    pypl.savefig(fold + 'Eff_dEshift_lambda_' + pars.momfold[pars.momentum][:7] + pars.snkfold[pars.sink] +'_'+op+'q'+str(quarknum+1)+'.pdf')
    # pypl.show()
    pypl.close()
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
    # ratiou = stats.feynhellratio(nclns[1][0], nclns[0][1], nclns[0][0], nclns[1][1])
    # ratiod = stats.feynhellratio(nclns[2][0], nclns[0][1], nclns[0][0], nclns[2][1])
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
        quarknum     = int(sys.argv[3])
        sinktype=0
        pars = pr.params(kappas, sinktype, momentum)
        pars.fit = 'Aexp'
        pars.makeresultdir(plots=True)

        opnum=0
        fitfunction  = ff.initffncs(pars.fit) #Initialise the fitting function
        
        # pars.lmbstring = 'lp001'
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp001 = read_data(pars)
        # ratioslp001, unpert_2plp001 = makeratio(nucleon_datalp001,opnum)
        # dictenergylp001, dictenergyshiftlp001 = oneexpreader(pars,opnum)

        # pars.lmbstring = 'lp01'
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp01 = read_data(pars)
        # ratioslp01, unpert_2plp01 = makeratio(nucleon_datalp01,opnum)
        # dictenergylp01, dictenergyshiftlp01 = oneexpreader(pars,opnum)

        # pars.lmbstring = 'lp02'
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp02 = read_data(pars)
        # ratioslp02, unpert_2plp02 = makeratio(nucleon_datalp02,opnum)
        # dictenergylp02, dictenergyshiftlp02 = oneexpreader(pars,opnum)
        
        # pars.lmbstring = 'lp04'
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp04 = read_data(pars)
        # ratioslp04, unpert_2plp04 = makeratio(nucleon_datalp04,opnum)
        # dictenergylp04, dictenergyshiftlp04 = oneexpreader(pars,opnum)

        # pars.lmbstring = 'lp08'
        # # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
        # nucleon_datalp08 = read_data(pars)
        # ratioslp08, unpert_2plp08 = makeratio(nucleon_datalp08,opnum)
        # dictenergylp08, dictenergyshiftlp08 = oneexpreader(pars,opnum)

        # # Function which plots results from both lambdas on the same plot, for
        # plotratios(ratioslp02, ratioslp01, dictenergyshiftlp02[quarknum], dictenergyshiftlp01[quarknum], fitfunction, pars, opnum, quarknum)
        # # plotratios(ratioslp001, ratioslp0001, dictenergyshiftlp001[quarknum], dictenergyshiftlp0001[quarknum], fitfunction, pars, opnum, quarknum)

        ratiolist = []
        dictenergyshiftlist = []
        for lmbstring in pars.lmblist:
            pars.lmbstring = lmbstring
            # Read the Bootstrap objects from the files. (Make sure self.nboot is set to the desired value in params.py)
            nucleon_data = read_data(pars)
            ratioslp, unpert_2plp = makeratio(nucleon_data,opnum)
            ratiolist.append(ratioslp.copy())
            dictenergylp, dictenergyshiftlp = oneexpreader(pars,opnum)
            dictenergyshiftlist.append(dictenergyshiftlp[quarknum])

        # print(f"{np.shape(ratiolist)=}")
        # print(f"{np.shape(dictenergyshiftlist)=}")
        # print(f"{dictenergyshiftlist=}")
        plotratiosloop(ratiolist, dictenergyshiftlist, fitfunction, pars, opnum, quarknum)
        plotlambdadep(dictenergyshiftlist, pars, opnum, quarknum)
        print('script time: \t', tm.time()-start)
    else:
        print("arg[1] = momentum")
        print("arg[2] = kappas")
        print("arg[3] = quarknum")
