# -*- eval: (comment-tags-mode) -*-
import numpy as np
import math
import time as tm
import scipy.optimize as syopt
from scipy.special import gamma, gammainc, gammaincc
import fitfunc as ff
from BootStrap3 import BootStrap
import matplotlib.pyplot as pypl
#import multiprocessing
from multiprocessing import Pool

def bstd(boot):
    retstd = boot.std(axis=1, ddof=1)
    return retstd

def Covmat(data):
    covmat=np.cov(data)
    return covmat
    
def Invcovmat(values,fullcov=True,pinv=False):
    if fullcov:
        cv=Covmat(values)
        if pinv:
            cvinv=np.linalg.pinv(cv)
        else:
            try:
                cvinv=np.linalg.inv(cv)
            except:
                print("cov. matrix did not invert")
                cvinv=np.linalg.pinv(cv)
    else:
        yerr=bstd(values)
        cvinv=np.linalg.inv(np.diag(yerr**2))
    return cvinv

def normcorr(corr, norm):
    ydata = corr
    for i in range(len(ydata)):
        ydata[i]=ydata[i]*(norm**(-1))
        ydata[i].Stats()
    return ydata

def effectivemass(G, a=0.074,factor=1.):
    effmass =[]
    for time in range(len(G)-1):
        effmass.append(BootStrap(G[0].nboot, 68))
        effmass[-1] = G[time]*(G[time+1]**(-1))
        effmass[-1].Avg = factor*np.log(np.abs(effmass[-1].Avg))#/a
        effmass[-1].values = factor*np.log(np.abs(effmass[-1].values))#/a
        effmass[-1].Stats()
    return effmass

def effmass(data, a=0.074):
    effmass = np.zeros(len(data)-1)
    for time in range(len(effmass)):
        effmass[time] = np.log(np.abs(data[time]*(data[time+1]**(-1))))
    return effmass

def minimize_comb(chifunc, func, x1, x2, ydata, p0, cov=True, pinv=False, bounds=None,niter=1000,stepsize=3,T=200,time=False):
    if time: start2 = tm.time()
    yavg = np.array([y.Avg for y in ydata])
    valarray = np.array([y.values for y in ydata])
    cvinv = Invcovmat(valarray, fullcov=cov, pinv=pinv)
    #res31=syopt.minimize(ff.chisqfn3,p0,args=(func,func,func,x1,x2,x3,yavg,cvinv),bounds=[(0,1),(0,2),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)],method='L-BFGS-B',options={'gtol': 1e-1, 'ftol':2.22e-12, 'eps':1e-05, 'maxls':30, 'disp': True, 'maxiter':4000, 'maxcor':20})
    #res31=syopt.basinhopping(ff.chisqfn3,p0,niter=30,stepsize=500,T=4000,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'Nelder-Mead'}, disp=True)
    #res31=syopt.basinhopping(ff.chisqfn3,p0,niter=300,stepsize=500,T=2000,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'L-BFGS-B','bounds':[(0,0.7),(0,2),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)]}, disp=True)
    #res31=syopt.basinhopping(chifunc,p0,niter=1000,stepsize=3,T=200,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'L-BFGS-B','bounds':bounds,'options':{'gtol': 1e-5,'eps':1e-11,'maxcor':40}}, disp=False)
    res31=syopt.basinhopping(chifunc,p0,niter=niter,stepsize=3,T=200,minimizer_kwargs={'args':(func,func,x1,x2,yavg,cvinv),'method':'L-BFGS-B','bounds':bounds,'options':{'gtol': 1e-5,'eps':1e-11,'maxcor':40}}, disp=False)
    #res31=syopt.basinhopping(ff.chisqfn3,p0,niter=15,stepsize=50,T=40,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'BFGS'}, disp=True)
    #res31=syopt.minimize(ff.chisqfn3,p0,args=(func,func,func,x1,x2,x3,yavg,cvinv),method='Nelder-Mead',options={'disp': True, 'fatol': 1e-10, 'maxiter':90000, 'adaptive':True }) #, 'eps':1.49e-13})
    print(res31.message, res31.nit)
    #redchisq=res31.fun/(len(x1)+len(x2)+len(x3)-len(p0))
    redchisq=res31.fun/(len(x1)+len(x2)-len(p0))
    if time: print('minimize_comb time: \t', tm.time()-start2)
    return res31.x, redchisq

def minimize_comb1exp(func, x1, x2, x3, ydata, p0, cov=True, pinv=False, bounds=None,niter=1000,stepsize=3,T=200):
    yavg = np.array([y.Avg for y in ydata])
    valarray = np.array([y.values for y in ydata])
    cvinv = Invcovmat(valarray, fullcov=cov, pinv=pinv)
    #res31=syopt.minimize(ff.combchisq1exp,p0,args=(func,func,func,x1,x2,x3,yavg,cvinv),bounds=bounds,method='L-BFGS-B',options={'gtol': 1e-1, 'ftol':2.22e-12, 'eps':1e-05, 'maxls':30, 'disp': False, 'maxiter':4000, 'maxcor':20})
    #res31=syopt.basinhopping(ff.chisqfn3,p0,niter=30,stepsize=500,T=4000,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'Nelder-Mead'}, disp=True)
    #res31=syopt.basinhopping(ff.chisqfn3,p0,niter=300,stepsize=500,T=2000,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'L-BFGS-B','bounds':[(0,0.7),(0,2),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)]}, disp=True)
    #res31=syopt.basinhopping(ff.combchisq1exp,p0,niter=1000,stepsize=3,T=200,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'L-BFGS-B','bounds':bounds,'options':{'gtol': 1e-5,'eps':1e-11,'maxcor':40}}, disp=False)
    res31=syopt.basinhopping(ff.combchisq1exp,p0,niter=15,stepsize=50,T=40,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'BFGS'}, disp=True)
    #res31=syopt.minimize(ff.chisqfn3,p0,args=(func,func,func,x1,x2,x3,yavg,cvinv),method='Nelder-Mead',options={'disp': True, 'fatol': 1e-10, 'maxiter':90000, 'adaptive':True }) #, 'eps':1.49e-13})
    print(res31.message, res31.nit)
    redchisq=res31.fun/(len(x1)+len(x2)+len(x3)-len(p0))

    # fit to the bootstrap ensembles
    #data = np.array([y.values for y in ydata])
    yerr = bstd(valarray)
    cv = np.diag(yerr**(-2))
    nboot=np.shape(valarray)[1]
    result=np.zeros((4,nboot)) #find a way to get number of parameters as a variable
    for iboot in range(nboot):
        yboot=valarray[:,iboot]
        #initialise fit params
        # try:
        #     fitfnc.initparfnc(yboot)
        # except:
        #     foo=1
        #res31=syopt.minimize(ff.combchisq1exp,p0,args=(func,func,func,x1,x2,x3,yavg,cvinv),bounds=bounds,method='L-BFGS-B',options={'gtol': 1e-1, 'ftol':2.22e-12, 'eps':1e-05, 'maxls':30, 'disp': False, 'maxiter':4000, 'maxcor':20})
        #res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'gtol': 1e-7, 'disp': False})
        res=syopt.minimize(ff.combchisq1exp,p0,args=(func,func,func,x1,x2,x3,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'gtol': 1e-7, 'disp': False})
        #res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='BFGS',options={'gtol': 1e-7, 'disp': False})
        result[:,iboot]=res.x
    return res31.x, redchisq, result

def minimize_comb2(func, x1, x2, x3, ydata, p0, cov=True, pinv=False, bounds=None,niter=400,stepsize=500,T=7000):
    yavg = np.array([y.Avg for y in ydata])
    valarray = np.array([y.values for y in ydata])
    cvinv = Invcovmat(valarray, fullcov=cov, pinv=pinv)
    res31=syopt.basinhopping(ff.chisqfn3,p0,niter=niter,stepsize=stepsize,T=T,minimizer_kwargs={'args':(func,func,func,x1,x2,x3,yavg,cvinv),'method':'L-BFGS-B','bounds':bounds}, disp=False)
    redchisq=res31.fun/(len(x1)+len(x2)+len(x3)-len(p0))
    return res31.x, redchisq

def BootFitt(fitfnc, p0,x,data,cv,bounds=None):
    nboot=np.shape(data)[1]
    result=np.zeros((fitfnc.npar,nboot))
    for iboot in range(nboot):
        # get data for current bootstrap
        yboot=data[:,iboot]
        # initialise fit params
        try:
            fitfnc.initparfnc(yboot)
        except:
            foo=1
        #res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='BFGS',options={'gtol': 1e-6, 'disp': False})
        res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'gtol': 1e-7, 'disp': False})
        #res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='Nelder-Mead',options={'disp': False, 'maxiter': 4000})
        result[:,iboot]=res.x
    return result

def BootFitclass(fitfnc, p0,x,ydata,bounds=None):
    data = np.array([y.values for y in ydata])
    yerr = bstd(data)
    cv = np.diag(yerr**(-2))
    nboot=np.shape(data)[1]
    result=np.zeros((fitfnc.npar,nboot))
    for iboot in range(nboot):
        yboot=data[:,iboot]
        #res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'gtol': 1e-7, 'disp': False})
        res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'disp': False})
        #res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='BFGS',options={'gtol': 1e-7, 'disp': False})
        result[:,iboot]=res.x
    energies = np.array([result[2*i+1][:] for i in range(int(fitfnc.npar/2))])
    if fitfnc.label==r"TwoexpRatio2":
        energies = np.array(result[-2:][:])
    energiessort = np.sort(energies, axis=0)
    if np.any(energies != energiessort):
        print("Energies Sorted")
    BSen=[]
    for i in range(len(energiessort)):
        BSen.append(BootStrap(len(energiessort[0]),68))
        BSen[i].values = energiessort[i]
        BSen[i].Stats()
    return result, BSen

def Bootratiofit(fitfnc,fitfncratio,p0,p1,x,ydata,yratio,bounds=None,bounds2=None):
    """Fitting the correlator first and then the ratio with two energy shifts"""
    data = np.array([y.values for y in ydata])
    yerr = bstd(data)
    cv = np.diag(yerr**(-2))
    nboot=np.shape(data)[1]
    result=np.zeros((fitfnc.npar,nboot))
    exppar=np.zeros((2,nboot))
    datar = np.array([y.values for y in yratio])
    yerrr = bstd(datar)
    cvr = np.diag(yerrr**(-2))
    resultratio=np.zeros((fitfncratio.npar,nboot))
    for iboot in range(nboot):
        #Fit the correlator to get amplitudes and energies, and set the variables for the ratio fit
        yboot=data[:,iboot]
        fitfnc.initparfnc(yboot)
        res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'gtol': 1e-9, 'disp': False})
        #res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='nelder-mead',options={'disp': False})
        #res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='BFGS',bounds=bounds,options={'gtol': 1e-9, 'disp': False})
        
        fitfncratio.q[0] = (res.x[2]/res.x[0])**(np.sign(res.x[3]-res.x[1]))
        fitfncratio.q[1] = np.abs(res.x[3]-res.x[1])
        #print(fitfncratio.q[1])
        result[:,iboot]=res.x
        exppar[:,iboot]=np.array([fitfncratio.q[0],fitfncratio.q[1]])

        ybootr=datar[:,iboot]
        fitfnc.initparfnc(ybootr)
        res2 = syopt.minimize(ff.chisqfn,p1,args=(fitfncratio.eval,x,ybootr,cvr),method='L-BFGS-B',bounds=bounds2,options={'gtol': 1e-7, 'disp': False})
        resultratio[:,iboot]=res2.x

    energies = np.array([result[2*i+1][:] for i in range(int(fitfnc.npar/2))])
    energiessort = np.sort(energies, axis=0)
    energyshift = np.array([resultratio[i][:] for i in range(int(fitfncratio.npar))])
    energyshiftsort = np.sort(energyshift, axis=0)
    if np.any(energies != energiessort):
        print("Energies Sorted")

    BSen  = []
    BSen2 = []
    for i in range(len(energiessort)):
        BSen.append( BootStrap(len(energiessort[0]),68))
        BSen[i].values = energiessort[i]
        BSen[i].Stats()
        BSen2.append(BootStrap(len(energiessort[0]),68))
        BSen2[i].values = energyshiftsort[i]
        BSen2[i].Stats()
    return result, exppar, resultratio, BSen, BSen2

def Bootratiofit2(fitfnc,fitfncratio,p0,p1,x,xr,ydata,yratio,bounds=None,bounds2=None):
    """Fitting the correlator first and then the ratio with two energy shifts"""
    data        = np.array([y.values for y in ydata])
    yerr        = bstd(data)
    cv          = np.diag(yerr**(-2))
    nboot       = np.shape(data)[1]
    result      = np.zeros((fitfnc.npar,nboot))
    exppar      = np.zeros((len(fitfncratio.q),nboot))
    datar       = np.array([y.values for y in yratio])
    yerrr       = bstd(datar)
    cvr         = np.diag(yerrr**(-2))
    resultratio = np.zeros((fitfncratio.npar,nboot))
    for iboot in range(nboot):
        #Fit the correlator to get amplitudes and energies, and set the variables for the ratio fit
        yboot=data[:,iboot]
        #res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'gtol': 1e-9, 'maxcor': 20, 'maxls': 50, 'disp': False})
        #res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='Nelder-Mead',options={'disp': False})
        res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'disp': False})
        #res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='BFGS',options={'disp': False})
        #res = syopt.basinhopping(ff.chisqfn,p0,niter=5,stepsize=3,T=5,minimizer_kwargs={'args':(fitfnc.eval,x,yboot,cv),'method':'L-BFGS-B','bounds':bounds}, disp=False)
        #res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='Nelder-Mead',options={'disp': False})
        #res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='BFGS',options={'gtol': 1e-9, 'disp': False})
        if res.x[3]<res.x[1]: sortres = np.array([*res.x[-2:],*res.x[:2]])
        else: sortres = res.x.copy()
        result[:,iboot]=sortres.copy()

        fitfncratio.q = sortres.copy()
        # fitfncratio.q[0] = sortres[0]
        # fitfncratio.q[1] = sortres[2]
        # fitfncratio.q[2] = sortres[3]-sortres[1]
        # p1[0] = sortres[0]
        # p1[2] = sortres[2]
        #print(f"{p1=}")
        exppar[:,iboot]=np.array([fitfncratio.q[0],fitfncratio.q[1],fitfncratio.q[2], fitfncratio.q[3]])
        ybootr=datar[:,iboot]
        #res2 = syopt.minimize(ff.chisqfn,p1,args=(fitfncratio.eval,xr,ybootr,cvr),method='L-BFGS-B',bounds=bounds2,options={'gtol': 1e-7, 'disp': False})
        #res2 = syopt.minimize(ff.chisqfn,p1,args=(fitfncratio.eval,xr,ybootr,cvr),method='BFGS',options={'gtol': 1e-7, 'disp': False})
        #res2 = syopt.minimize(ff.chisqfn,p1,args=(fitfncratio.eval,xr,ybootr,cvr),method='Nelder-Mead',options={'disp': False})
        res2 = syopt.minimize(ff.chisqfn,p1,args=(fitfncratio.eval,xr,ybootr,cvr),method='L-BFGS-B',bounds=bounds2,options={'disp': False})
        #print(f"{res2.x=}")
        # if res2.x[3]<res2.x[1]: ratiores = np.array([*res2.x[-2:],*res2.x[:2]])
        # else: ratiores = res2.x.copy()
        # resultratio[:,iboot]=ratiores.copy()
        resultratio[:,iboot] = res2.x.copy()

    #print(f"{np.average(resultratio,axis=1)=}")
    energies = np.array([result[2*i+1][:] for i in range(int(fitfnc.npar/2))])
    energyshift = np.array([resultratio[2*i+1][:] for i in range(int(fitfncratio.npar/2))])
    #energyshiftsort = np.sort(energyshift, axis=0)
    BSen  = []
    BSen2 = []
    for i in range(len(energies)):
        BSen.append( BootStrap(len(energies[0]),68))
        BSen[i].values = energies[i]
        BSen[i].Stats()
        BSen2.append(BootStrap(len(energyshift[0]),68))
        BSen2[i].values = energyshift[i]
        BSen2[i].Stats()
    return result, exppar, resultratio, BSen, BSen2

def Bootratiofit3(fitfnc,fitfncratio,p0,p1,x,xr,ydata,yratio,bounds=None,bounds2=None,time=False):
    """Fitting the correlator first and then the ratio with two energy shifts. This function will return all parameters as Bootstrap objects"""
    if time: start1 = tm.time() #Start counting
    data           = np.array([y.values for y in ydata])
    data_err       = bstd(data)
    cv             = np.diag(data_err**(-2))
    nboot          = np.shape(data)[1]
    result         = np.zeros((fitfnc.npar,nboot))
    data_ratio     = np.array([y.values for y in yratio])
    data_err_ratio = bstd(data_ratio)
    cv_ratio       = np.diag(data_err_ratio**(-2))
    resultratio    = np.zeros((fitfncratio.npar,nboot))
    
    for iboot in range(nboot):
        #Fit the unperturbed 2-point function to get amplitudes and energies, and set the variables for the ratio fit
        yboot=data[:,iboot]
        res = syopt.minimize(ff.chisqfn,p0,args=(fitfnc.eval,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'disp': False})
        sortres = sortmompar(res.x)
        result[:,iboot]=sortres.copy()
        fitfncratio.q = sortres.copy()
        
        #Fit the ratio of 2-point functions to get the shift in the amplitudes and energies
        ybootr=data_ratio[:,iboot]
        res2 = syopt.minimize(ff.chisqfn,p1,args=(fitfncratio.eval,xr,ybootr,cv_ratio),method='L-BFGS-B',bounds=bounds2,options={'disp': False})
        resultratio[:,iboot] = res2.x.copy()

    parBS=[] #Save resulting parameters as Bootstrap objects
    parBS_ratio=[]
    for i in range(len(p0)):
        parBS.append(BootStrap(nboot,68))
        parBS[i].values = result[i][:]
        parBS[i].Stats()
    for i in range(len(p1)):
        parBS_ratio.append(BootStrap(nboot,68))
        parBS_ratio[i].values = resultratio[i][:]
        parBS_ratio[i].Stats()
    if time: print('Bootratiofit3 time: \t', tm.time()-start1) #Stop counting and print time
    return parBS, parBS_ratio

def Bootratiofit4(fitfncratio,p0,xr,yratio,energyBS,bounds=None,time=False,fullcov=False):
    """Fitting the correlator first and then the ratio with two energy shifts. This function will return all parameters as Bootstrap objects"""
    if time: start1 = tm.time() #Start counting
    nboot          = yratio[0].nboot
    data_ratio     = np.array([y.values for y in yratio])
    data_err_ratio = bstd(data_ratio)
    # cv_ratio       = np.diag(data_err_ratio**(-2))
    cv_ratio       = Invcovmat(data_ratio,fullcov=fullcov)
    cv_ratio_diag  = Invcovmat(data_ratio,fullcov=False)
    resultratio    = np.zeros((fitfncratio.npar,nboot))

    #INFO: Fit to the average of the ratio data to get a chi-square value
    fitfncratio.q = [i.Avg for i in energyBS] #Set the energyfit parameters
    ydata=np.average(data_ratio,axis=1)
    niter=50
    res = syopt.basinhopping(ff.chisqfn,p0,niter=niter,stepsize=8,T=150,minimizer_kwargs={'args':(fitfncratio.eval,xr,ydata,cv_ratio),'method':'L-BFGS-B','bounds':bounds}, disp=False)
    #res = syopt.minimize(ff.chisqfn,p0,args=(fitfncratio.eval,xr,ydata,cv_ratio),method='L-BFGS-B',bounds=bounds,options={'disp': False})
    redchisq = res.fun/(len(ydata) - len(p0))
    
    for iboot in range(nboot):
        #Set the energyfit parameters
        fitfncratio.q = [i.values[iboot] for i in energyBS]
        #Fit the ratio of 2-point functions to get the shift in the amplitudes and energies
        ybootr=data_ratio[:,iboot]
        # res2 = syopt.minimize(ff.chisqfn,p0,args=(fitfncratio.eval,xr,ybootr,cv_ratio),method='L-BFGS-B',bounds=bounds,options={'disp': False})
        
        # THIS NOW USES THE RESULT OF THE FIT TO THE AVG AS INITAL GUESS.
        res2 = syopt.minimize(ff.chisqfn,res.x,args=(fitfncratio.eval,xr,ybootr,cv_ratio),method='L-BFGS-B',bounds=bounds,options={'disp': False})
        resultratio[:,iboot] = res2.x.copy()

    parBS_ratio=[] #Save resulting parameters as Bootstrap objects
    for i in range(len(p0)):
        parBS_ratio.append(BootStrap(nboot,68))
        parBS_ratio[i].values = resultratio[i][:]
        parBS_ratio[i].Stats()
    if time: print('Bootratiofit4 time: \t', tm.time()-start1) #Stop counting and print time
    return parBS_ratio, redchisq

def BootFittComb(func,x1,x2,x3,ydata,p0,cov=False,bounds=None,chisq=ff.chisqfn3):
    valarray = np.array([y.values for y in ydata])
    cvinv = Invcovmat(valarray, fullcov=cov, pinv=False)
    nboot=ydata[0].nboot
    result=np.zeros((len(p0),nboot))
    for iboot in range(nboot):
        yboot=np.array([y.values[iboot] for y in ydata])
        #res=syopt.minimize(ff.chisqfn3,p0,args=(func,func,func,x1,x2,x3,yboot,cvinv),method='Nelder-Mead',options={'disp': False, 'maxiter':400 }) #, 'eps':1.49e-13})
        #res=syopt.minimize(ff.chisqfn3,p0,args=(func,func,func,x1,x2,x3,yboot,cvinv),method='BFGS',options={'gtol': 1e-6, 'disp': False, 'maxiter':400 }) #, 'eps':1.49e-13})
        res=syopt.minimize(chisq,p0,args=(func,func,func,x1,x2,x3,yboot,cvinv),bounds=bounds,method='L-BFGS-B',options={'gtol': 1e-7, 'disp': False}) #, 'eps':1.49e-13})
        result[:,iboot]=res.x
    return result

def feynhellratio(G1, G2, G3, G4, a=0.074, alambda=0.0001, m=1, Ep=1):
    nboot=G1[0].nboot
    ratio=[]
    for gi, gj, gk, gl in zip(G1, G2, G3, G4):
        ratio.append(BootStrap(nboot, 68))
        ratio[-1] = np.abs((gi*gj)*(gk*gl)**(-1))**(0.5)
        ratio[-1].Stats()
    return ratio

def feynhellratio2(G1, G2, G3, G4, a=0.074, alambda=0.0001, m=1, Ep=1):
    ratio=[]
    for gi, gj, gk, gl in zip(G1, G2, G3, G4):
        ratio.append(BootStrap(G1[0].nboot, 68))
        ratio[-1] = np.abs((gi*gj)*(gk*gl)**(-1))
        ratio[-1].Stats()
    return ratio

def feynhellratio3(G0, Gl, a=0.074, alambda=0.0001, m=1, Ep=1):
    ratio=[]
    for i in range(len(G0[0][0])):
        ratio.append(BootStrap(G0[0][0][0].nboot, 68))
        ratio[-1] = np.abs( ((Gl[2][0][i]+Gl[4][0][i])*(G0[6][0][i]+G0[8][0][i])*(G0[3][0][i]+G0[5][0][i])*(Gl[7][0][i]+Gl[9][0][i])) * ((G0[2][0][i]+G0[4][0][i])*(Gl[6][0][i]+Gl[8][0][i])*(Gl[3][0][i]+Gl[5][0][i])*(G0[7][0][i]+G0[9][0][i]))**(-1) )**(1/4)
        ratio[-1].Stats()
    return ratio

def feynhellratio4(G0, Gl, a=0.074, alambda=0.0001, m=1, Ep=1):
    ratio=[]
    for i in range(len(G0[0])):
        ratio.append(BootStrap(G0[0][0].nboot, 68))
        ratio[-1] = np.abs( ((Gl[2][i]+Gl[4][i])*(G0[6][i]+G0[8][i])*(G0[3][i]+G0[5][i])*(Gl[7][i]+Gl[9][i])) * ((G0[2][i]+G0[4][i])*(Gl[6][i]+Gl[8][i])*(Gl[3][i]+Gl[5][i])*(G0[7][i]+G0[9][i]))**(-1) )**(1/4)
        ratio[-1].Stats()
    return ratio

def feynhellratio5(G0, Gl, a=0.074, alambda=0.0001, m=1, Ep=1):
    ratio=[]
    for i in range(len(G0[0])):
        ratio.append(BootStrap(G0[0][0].nboot, 68))
        ratio[-1] = np.abs( ((Gl[2][i]+Gl[4][i])*(G0[6][i]+G0[8][i])*(G0[3][i]+G0[5][i])*(Gl[7][i]+Gl[9][i])) * ((G0[2][i]+G0[4][i])*(Gl[6][i]+Gl[8][i])*(Gl[3][i]+Gl[5][i])*(G0[7][i]+G0[9][i]))**(-1) )**(1/2)
        ratio[-1].Stats()
    return ratio


def feynhellratioshort(G1, G2, a=0.074, alambda=0.0001, m=1, Ep=1):
    ratio=[]
    for gi, gj in zip(G1, G2):
        ratio.append(BootStrap(G1[0].nboot, 68))
        ratio[-1] = np.abs((gi)*(gj**(-1)))
        ratio[-1].Stats()
    return ratio


def EnergyFit(tdata,nucleon,initialg,bounds,btbounds,evalfnc,pars,plot=True,cov=True,niter=400,stepsize=500,T=7000,name="test"):
    """Fit to the data using the combined sinks fit and plot the effective mass with the energy fit to the bootstraps"""
    normdata = np.array([ normcorr(np.array(nucleon[sink]), nucleon[sink][tdata[sink][0]].Avg) for sink in range(3) ])
    ydata0 = np.array([ normdata[i][tdata[i]] for i in range(3) ])
    ydata = np.array([ *ydata0[0], *ydata0[1], *ydata0[2] ])
    fitparam,redchisq=minimize_comb2(evalfnc, tdata[0], tdata[1], tdata[2], ydata, p0=initialg, cov=cov, pinv=False,bounds=bounds,niter=niter,stepsize=stepsize,T=T)
    print(f"{fitparam=}")
    print(f"{redchisq=}")
    energies=np.sort(np.abs(fitparam[:2]))
    print(f"{energies=}")
    
    fitbt = BootFittComb(evalfnc,tdata[0],tdata[1],tdata[2],ydata,p0=fitparam,cov=cov,bounds=btbounds)
    energiessort = np.sort(np.abs(np.array([fitbt[i][:] for i in range(2)])), axis=1)
    energieserr = np.array([i.std(axis=0, ddof=1) for i in energiessort])
    energiesval = np.array([np.average(i, axis=0) for i in energiessort])
    fitavg = np.array([*energiesval, *[np.average(fitbt[i,:]) for i in range(2,8)] ])
    print(f"{energiesval=}")
    print(f"{fitavg=}")
    #TODO: Import into BS elements
    E0 = BootStrap(len(energiessort[0]),68)
    E0.values = energiessort[0]
    E0.Stats()
    E1 = BootStrap(len(energiessort[1]),68)
    E1.values = energiessort[1]
    E1.Stats()
    
    if plot:
        pypl.figure("energycombfit", figsize=(16,9))
        time = np.arange(0,64)
        efftime = time+0.5
        momfold = ['p+0+0+0/q+0+0+0/', 'p+1+0+0/q+0+0+0/', 'p+1+1+1/q+0+0+0/', 'p+2+1+0/q+0+0+0/', 'p+2+2+0/q+0+0+0/']
        snkfold = ['ptsnk/', 'smsnk30/', 'smsnk60/']
        colors = ['b', 'r', 'y']
        markers=['s','o','^','*', 'v', '>', '<']
        for sink in range(3):
            yeffavg = np.array([y.Avg for y in effectivemass(nucleon[sink])])
            yefferr = np.array([y.Std for y in effectivemass(nucleon[sink])])
            pypl.errorbar(efftime[:pars.xlim]+sink/12, yeffavg[:pars.xlim], yefferr[:pars.xlim], fmt='.', capsize=4, elinewidth=1, color=colors[sink], marker=markers[sink], markerfacecolor='none', label='effective energy of the correlator ' + snkfold[sink][:-1] + ' ' + momfold[pars.momentum][:7])
            fitBS = np.array([ effmass(evalfnc(tdata[sink],[fitbt[sink*2+2,nbo],fitbt[0,nbo],fitbt[sink*2+3,nbo],fitbt[1,nbo]])) for nbo in range(len(fitbt[0][:])) ])
            fitBSavg = np.average(fitBS,axis=0)
            fitBSstd = fitBS.std(axis=0, ddof=1)
            fitBSlower = fitBSavg - fitBSstd
            fitBShigher = fitBSavg + fitBSstd
            pypl.plot(tdata[sink][:-1]+0.5,effmass(evalfnc(tdata[sink], [fitparam[sink*2+2],fitparam[0],fitparam[sink*2+3],fitparam[1]])), linestyle='-', color='k', label=pars.fit+' fit effmass fullcov'+str(sink))
            pypl.plot(tdata[sink][:-1]+0.5,fitBSavg, linestyle='-', color=colors[sink], label=pars.fit+' fit effmass '+str(sink))
            pypl.fill_between(tdata[sink][:-1]+0.5, fitBSlower, fitBShigher, color=colors[sink], alpha=0.3, linewidth=0)

        pypl.legend(fontsize='small')
        pypl.xlabel(r'$t/a$',labelpad=14,fontsize=18)
        pypl.ylabel(r'$G_{eff}$',labelpad=5,fontsize=18)
        pypl.title(r'Effective energy correlator')
        pypl.grid(True, alpha=0.6)
        pypl.ylim(0,1.8)
        pypl.subplots_adjust(bottom=0.17)
        pypl.savefig('energy_effcorr_combfit_'+ name + momfold[pars.momentum][:7] + '_' + pars.fit + '.pdf') #, bbox
        pypl.close()    
    return energies,E0,E1

def str_with_err(x, dx):
    """Returns a string of the value with the error in brackets with 2 digits"""
    err_dig = 2 #number of digits in the error
    print(f"{math.floor(math.log10(abs(x)))=}")
    print(f"{math.floor(math.log10(dx))=}")
    if math.floor(math.log10(dx)) > math.floor(math.log10(abs(x))):
        return r'{:.3e}$\pm${:.3e}'.format(x,dx)
    # Power of dx
    pwr_err = math.log10(dx)
    # Digits of dx in format ab.cde
    n_err   = dx / (10**math.floor(pwr_err+1-err_dig))
    #print(f"{n_err=}")
    # If the first excluded digit in dx is >=5
    # round the first two digits in dx up
    if n_err % 1 >= 0.5:
        # If the first two digits of dx are 9
        # the precision is one digit less
        if int(n_err) == 99:
            err = 10
            # The precision of x is determined by the precision of dx
            prec=int(-math.floor(math.log10(dx)))
        else:
            err = math.ceil(n_err)
            # The precision of x is determined by the precision of dx
            prec=int(-math.floor(math.log10(dx)))+1
    # Otherwise round down
    else:
        err = math.floor(npw_err)
        # The precision of x is determined by the precision of dx
        prec=int(-math.floor(math.log10(dx)))+1

    if dx >= 99.5:
        val=x/(10**(math.floor(math.log10(abs(x)))))
        d1 = math.floor(math.log10(abs(x)))
        d2 = math.floor(math.log10(dx))+1-2
        form = 'e'
        print('{:}'.format(val))
        print('{:.{d}{fm}}'.format(val,d=d1-d2,fm=form))
        # if '{:.{d}{fm}}'.format(val,d=d1-d2,fm=form)[-4]=='e':
        #     return '{:.{d}e}({:.0f})*$10^{{{b}}}$'.format(val, err, b=d1, d=d1-d2 )
        # else:
        #     return '{:.{d}e}({:.0f})*$10^{{{b}}}$'.format(val, err, b=d1, d=d1-d2 )
        return '{:.{d}{fm}}({:.0f})*$10^{{{b}}}$'.format(val, err, b=d1, d=d1-d2, fm=form)
    else:
        return '{:.{prec}f}({:.0f})'.format(x, err, prec = max(0,prec))

    #return '{:.{prec}f}({:.0f})'.format(value, math.floor(error / (10**math.floor(math.log10(error)))), prec=-math.floor(math.log10(error)))
    # if error != 0:
    #     digits = -int(math.floor(math.log10(error)))+1
    #     print(f"{digits=}")
    #     return "{0:.{2}f}({1:.0f})".format(value, error*10**digits, digits)
    # else:
    #     return "{:.3f}".format(value)

def minimize_cov2(func, xdata, ydata, p0=None, cov=True, prev=None, fn2=None,bounds=None,niter=50, time=False):
    if time: start2 = tm.time()
    yavg = np.array([y.Avg for y in ydata])
    valarray = np.array([y.values for y in ydata])
    cvinv = Invcovmat(valarray, fullcov=cov)
    #res31=syopt.minimize(ff.chisqfn,p0,args=(func,xdata,yavg,cvinv),method='BFGS',options={'gtol': 1e-6, 'disp': False})
    #res31=syopt.minimize(ff.chisqfn,p0,args=(func,xdata,yavg,cvinv),method='Nelder-Mead',options={'disp': False, 'maxiter': 9000})
    #res31=syopt.minimize(ff.chisqfn,p0,args=(func,xdata,yavg,cvinv),bounds=bounds,method='L-BFGS-B') #,options={'gtol': 1e-1, 'ftol':2.22e-12, 'eps':1e-05, 'maxls':30, 'disp': True, 'maxiter':4000, 'maxcor':20})
    #res31=syopt.basinhopping(ff.chisqfn,p0,niter=400,stepsize=8,T=150,minimizer_kwargs={'args':(func,xdata,yavg,cvinv),'method':'L-BFGS-B','bounds':bounds}, disp=False)
    #res31=syopt.basinhopping(ff.chisqfn,p0,niter=400,stepsize=800,T=1500,minimizer_kwargs={'args':(func,xdata,yavg,cvinv),'method':'L-BFGS-B','bounds':bounds}, disp=False)
    #res31=syopt.basinhopping(ff.chisqfn,p0,niter=400,stepsize=8,T=150,minimizer_kwargs={'args':(func,xdata,yavg,cvinv),'method':'BFGS'}, disp=True)
    #res31=syopt.basinhopping(ff.chisqfn,p0,niter=400,stepsize=30,T=2000,minimizer_kwargs={'args':(func,xdata,yavg,cvinv),'method':'Nelder-Mead'}, disp=False)
    #res31=syopt.basinhopping(ff.chisqfn,p0,niter=40,stepsize=3,T=5,minimizer_kwargs={'args':(func,xdata,yavg,cvinv),'method':'BFGS'}, disp=False)
    #res31=syopt.basinhopping(ff.chisqfn,p0,niter=50,stepsize=3,T=5,minimizer_kwargs={'args':(func,xdata,yavg,cvinv),'method':'Nelder-Mead'}, disp=False)
    # This worked best for kp120900kp120900:
    res31=syopt.basinhopping(ff.chisqfn,p0,niter=niter,stepsize=8,T=150,minimizer_kwargs={'args':(func,xdata,yavg,cvinv),'method':'L-BFGS-B','bounds':bounds}, disp=False)
    #res31=syopt.basinhopping(ff.chisqfn,p0,niter=50,stepsize=3,T=5,minimizer_kwargs={'args':(func,xdata,yavg,cvinv),'method':'L-BFGS-B','bounds':bounds,'options':{'gtol': 1e-6}}, disp=True)
    #print(res31.message, res31.nit)
    redchisq=res31.fun/(len(xdata)-len(p0))
    if time:
        end2 = tm.time()
        #print('Full covariance fit time: \t', end2-start2)
        print('minimize_cov2 time: \t', end2-start2)
    return res31.x, redchisq


def BootFitclass2(fitfnc,p0,x,ydata,bounds=None,time=False):
    if time: start1 = tm.time()
    data = np.array([y.values for y in ydata]).T
    yerr = np.std(data,axis=0)
    cv = np.diag(yerr**(-2))
    nboot=ydata[0].nboot
    result=np.zeros((len(p0),nboot))
    start2=tm.time()
    sorttime = 0
    for iboot in range(nboot):
        yboot=data[iboot,:]
        res=syopt.minimize(ff.chisqfn,p0,args=(fitfnc,x,yboot,cv),method='L-BFGS-B',bounds=bounds,options={'disp': False})
        amplitudes = res.x[::2]
        energies   = res.x[1::2]
        sortinglist = [x for _,x in sorted(zip(energies, np.arange(len(energies)) ))]
        sortedlist  = np.array([ [amplitudes[i], energies[i]] for i in sortinglist ]).flatten()
        if any(sortedlist!=res.x): print("Sorted!")
        result[:,iboot]=sortedlist
    print('----Loop time: \t', tm.time()-start2)
    parBS=[]
    for i in range(len(p0)):
        parBS.append(BootStrap(nboot,68))
        parBS[i].values = result[i][:]
        parBS[i].Stats()
    if time:
        end1  = tm.time()
        print('BootFitclass2 time: \t', end1-start1)
    return result, parBS

def minimizer0(args):
    func,p0,fitfnc,x,data,cv,bounds = args
    res=syopt.minimize(func,p0,args=(fitfnc,x,data,cv),method='L-BFGS-B',bounds=bounds,options={'disp': False})
    return res.x

def BootFitclass3(fitfnc,p0,x,ydata,bounds=None,time=False,fullcov=False):
    """
    Fit to every bootstrap ensemble and use multiprocessing to split up the task over two processors
    p0: initial guess for the parameters
    x: array of x values to fit over
    ydata: array/list of BootStrap objects
    """
    if time: start1 = tm.time() #Start counting

    nboot=ydata[0].nboot
    data = np.array([y.values for y in ydata]).T
    yerr = np.std(data,axis=0)
    cv = Invcovmat(np.array([y.values for y in ydata]),fullcov=fullcov,pinv=False)

    #Fit to the data average to get a chi-squared value for this
    dataavg = np.array([y.Avg for y in ydata]).T
    # resavg=syopt.minimize(ff.chisqfn,p0,args=(fitfnc,x,dataavg,cv),method='L-BFGS-B',bounds=bounds,options={'disp': False})
    niter=100
    resavg = syopt.basinhopping(ff.chisqfn,p0,niter=niter,stepsize=8,T=150,minimizer_kwargs={'args':(fitfnc,x,dataavg,cv),'method':'L-BFGS-B','bounds':bounds}, disp=False)
    redchisq1=resavg.fun/(len(data[0,:])-len(p0))

    p0 = resavg.x
    
    args = [(ff.chisqfn,p0,fitfnc,x,data[iboot,:],cv,bounds) for iboot in range(nboot)]
    p = Pool(2) #Use two processors for the calculation
    result = np.array(p.map(minimizer0,args)).T #Use the pool to map the arguments to the minimiser function.

    # minimized = p.map(minimizer0,args) #Use the pool to map the arguments to the minimizer function.
    # result = np.array([i.x for i in minimized]).T
    # redchisq=np.array([i.fun for i in minimized])/(len(data[0,:])-len(p0))
    # print(f"{np.average(redchisq)=}")
    # print(f"{np.median(redchisq)=}") 

    # resultvals= p.map(minimizer0,args) #Use the pool to map the arguments to the minimizer function.
    # result = np.array([i[0] for i in resultvals]).T
    # redchisq=np.array([i[1] for i in resultvals])/(len(data[0,:])-len(p0))
    # print(f"{resultvals=}")
    # print(f"{result=}")
    # print(f"{redchisq=}")
    # result = np.array(p.map(minimizer0,args)).T #Use the pool to map the arguments to the minimizer function.
    # redchisq=1.0
    
    # This checks whether the energies are sorted from small to large for multiple exponential functions
    for i, pars in enumerate(result.T):
        if len(pars)>=4:
            if pars[1]>pars[3]: #This really only checks for 2 exponential functions
                #print("Not sorted")
                temp = sortmompar(pars)
                for j in range(len(temp)):
                    result[j][i]=temp[j]
    
    parBS=[] #Save resulting parameters as Bootstrap objects
    for i in range(len(p0)):
        parBS.append(BootStrap(nboot,68))
        parBS[i].values = result[i][:]
        parBS[i].Stats()
        
    if time: print('BootFitclass3 time: \t', tm.time()-start1) #Stop counting and print time
    return result, parBS, redchisq1

def minimizer0comb(args):
    func,p0,fitfnc,x1,x2,ydata,cvinv,bounds = args
    res=syopt.minimize(func,p0,args=(fitfnc,fitfnc,x1,x2,ydata,cvinv),method='L-BFGS-B',bounds=bounds,options={'disp': False})
    return res.x

def BootFitcomb3(chifunc,fitfnc,x1,x2,ydata,p0,bounds=None,time=False):
    """Fit to every bootstrap ensemble and use multiprocessing to split up the task over two processors"""
    if time: start1 = tm.time()

    yval  = np.array([y.values for y in ydata]).T
    yerr  = np.std(yval,axis=0)
    cvinv = np.diag(yerr**(-2))
    nboot = ydata[0].nboot

    args   = [(chifunc,p0,fitfnc,x1,x2,yval[iboot,:],cvinv,bounds) for iboot in range(nboot)]
    p      = Pool(2) #Use two processors for the calculation
    result = np.array(p.map(minimizer0comb,args)).T #Use the pool to map the arguments to the minimiser function.

    if np.average(result[0])>np.average(result[1]):
        result = np.array([ result[1], result[0], result[3], result[2], result[5], result[4]])
        print('sorting')
    
    parBS=[] #Save resulting parameters as Bootstrap objects
    for i in range(len(p0)):
        parBS.append(BootStrap(nboot,68))
        parBS[i].values = result[i][:]
        parBS[i].Stats()
        
    if time:
        print('BootFitcomb3 time: \t', tm.time()-start1)
    return result, parBS

def sortmompar(params):
    amplitudes = params[::2]
    energies   = params[1::2]
    sortinglist = [x for _,x in sorted(zip(energies, np.arange(len(energies)) ))]
    sortedlist = np.array([ [amplitudes[i], energies[i]] for i in sortinglist ]).flatten()
    return sortedlist

def fitweights(dof, chisq, derrors):
    """Take a list of degrees of freedom and of chi-squared (not reduced) values and errors of the fit and return the weights for each fit"""
    pf =[]
    for d, chi in zip(dof, chisq):
        #pf.append(gammaincc(d/2,chi/2)/gamma(d/2))
        pf.append(gammaincc(d/2,chi/2))
    denominator = sum(np.array(pf)*np.array([d**(-2) for d in derrors]))
    weights=[]
    for p, de in zip(pf, derrors):
        weights.append(p*(de**(-2))/denominator)
    return weights
