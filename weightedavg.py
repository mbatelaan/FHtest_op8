# Read the fit files from the oneexp fit script and Plot the effective energy shift for multiple lambda values
# also plot value of the weighted average of the fits with the fitrange of the fit which got the highest weighting.
import numpy as np
from scipy.special import gammaincc
import dill as pickle
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

    # open the pickle files and extract the fit parameters
    with open(bsfile, 'rb') as fileout:
        nucleon = pickle.load(fileout)
    with open(bsfile1, 'rb') as fileout:
        nucleon_q1  = pickle.load(fileout)
    with open(bsfile2, 'rb') as fileout:
        nucleon_q2 = pickle.load(fileout)
    nucleons_raw = [nucleon,nucleon_q1,nucleon_q2]

    # Save the data in Bootstrap objects
    nucleons = [[],[],[]]
    for num in range(len(nucleons_raw[0])):
        for type,nu in enumerate(nucleons):
            nu.append([])
            for t in range(len(nucleons_raw[type][num])):
                nu[num].append(BootStrap(len(nucleons_raw[type][num][t]), pars.confidence))
                nu[num][-1].values=nucleons_raw[type][num][t]
                nu[num][-1].Stats()
    return nucleons

def weights(dof, chisq, derrors):
    """
    Take a list of degrees of freedom and of chi-squared values and errors of the fit and return the weights for each fit
    chisq is NOT the reduced chi-squared value
    don't forget to import gammaincc from scipy.special
    """
    pf =[]
    for d, chi in zip(dof, chisq):
        pf.append(gammaincc(d/2,chi/2))
    denominator = sum(np.array(pf)*np.array([d**(-2) for d in derrors]))
    weights=[]
    for p, de in zip(pf, derrors):
        weights.append(p*(de**(-2))/denominator)
    return weights

def oneexpreader(pars,opnum):
    """
    Read the fit parameters from pickle files for the energy fit and the energy shift fit. Return the data as a dictionary
    """
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
    # print(f"{weights_dE_lp025=}")
    # print(f"{np.shape(np.array([dictenergyshift1[i]['E0'] for i in dictenergyshift1] ))=}")
    weighted_dE_lp025 = (np.array(weights_dE_lp025)*np.array([dictenergyshift1[i]['E0'] for i in dictenergyshift1]).T).T
    # print(f"{np.shape(weighted_dE_lp025)=}")
    avgdE_lp025 = np.sum(weighted_dE_lp025, axis=0)
    meanavgdE_lp025 = np.average(avgdE_lp025)
    staterrsq_dE = np.std(avgdE_lp025,ddof=1)**2
    # print(f"{np.shape(avgdE_lp025)=}")
    # print(f"{np.average(avgdE_lp025)=}")
    # print(f"{np.std(avgdE_lp025, ddof=1)=}")
    # The square of the differences of each fit mean to the weighted average mean
    mean_diff = np.array([(np.average(dictenergyshift1[i]['E0'])-meanavgdE_lp025)**2 for i in dictenergyshift1])
    # The systematic error due to taking multiple fit results in the average
    systerrsq_dE = np.sum(weights_dE_lp025*mean_diff)
    # The combined error includes the statistical and systematic uncertainty
    comberr_dE = np.sqrt(staterrsq_dE+systerrsq_dE)
    # Rescale the bootstrap values such that the standard deviation of the bootstrap values is equal to the combined error, without changing the mean value of the bootstraps
    for nboot in range(len(avgdE_lp025)):
        avgdE_lp025[nboot] = meanavgdE_lp025+(avgdE_lp025[nboot]-meanavgdE_lp025)*comberr_dE/(staterrsq_dE**(0.5))
    # print(f"{np.average(avgdE_lp025)=}")
    # print(f"{np.std(avgdE_lp025, ddof=1)=}")

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
    # print(f"{np.average(avgdE_lp01)=}")
    # print(f"{np.std(avgdE_lp01, ddof=1)=}")

    
    #lp025
    # Get the index of the fit with the chi-squared value closest to 1
    bestweight1 = np.argsort(weights_dE_lp025)[-1]
    index = bestweight1
    fitrange   = np.arange(dictenergyshift1[index]['tmin'], dictenergyshift1[index]['tmax'])
    redchisq   = dictenergyshift1[index]['chisq']
    fitBS=np.array([avgdE_lp025 for  i in time[:pars.xlim]]).T
    print(f"{np.shape(fitBS)=}")
    fitBSavg   = np.average(fitBS,axis=0)
    fitBSstd   = fitBS.std(axis=0, ddof=1)
    fitBSlower = fitBSavg - fitBSstd
    fitBSupper = fitBSavg + fitBSstd

    #lp01
    # Get the index of the fit with the chi-squared value closest to 1
    bestweight2 = np.argsort(weights_dE_lp01)[-1]
    index2 = bestweight2
    fitrange2   = np.arange(dictenergyshift2[index2]['tmin'], dictenergyshift2[index2]['tmax'])
    redchisq2   = dictenergyshift2[index2]['chisq']
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
    fitrangelp01 = np.arange(pars.tminmin,pars.tmaxmax)
    pypl.plot(fitrangelp01+0.5,fitBSavg2[fitrangelp01], linestyle='-', color='b')#, label="OneExp fit $\chi^2=${:0.2f}".format(redchisq2))
    pypl.fill_between(fitrangelp01+0.5, fitBSlower2[fitrangelp01], fitBSupper2[fitrangelp01], color='b', alpha=0.3, linewidth=0)

    # indicate timeranges
    pypl.fill_between(np.arange(0,pars.tminmin+1), -100, 100, color='k', alpha=0.2, linewidth=0)
    pypl.fill_between(np.arange(pars.tmaxmax,32), -100, 100, color='k', alpha=0.2, linewidth=0)
    
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

        plotratiosloop(ratiolist, dictenergyshiftlist, fitfunction, pars, opnum, quarknum)
        plotlambdadep(dictenergyshiftlist, pars, opnum, quarknum)
        print('script time: \t', tm.time()-start)
    else:
        print("arg[1] = momentum")
        print("arg[2] = kappas")
        print("arg[3] = quarknum")
