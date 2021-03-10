# -*- eval: (comment-tags-mode) -*-
#INFO: Resample the data from evxpt dumpresult files into bootstrap ensembles and save them as pickle files.
import numpy as np
import sys
import os
import time
# import pandas as pd
import dill as pickle
import params as pr
from BootStrap3 import BootStrap
from evxptreaders import ReadEvxptdump
import stats as stats

def pickledata(pars, imag=0):
    nucleon                = [[],[],[],[],[],[],[],[],[],[]]
    nucleongu, nucleongd   = [[],[],[],[],[],[],[],[],[],[]], [[],[],[],[],[],[],[],[],[],[]]
    print("------------------------------")
    print(pars.momfold[pars.momentum][:7])
    print(pars.lmbstring)
    print("------------------------------")

    # -----Data Files-----
    qval = np.array2string(pars.qval[pars.momentum],separator='')[1:-1]
    bsfile  = pars.data_dir + 'TwoptBS_q' + qval + '.pkl'
    bsfile1 = pars.data_dir + 'TwoptBS_q' + qval + '_quark1_' + pars.lmbstring + '.pkl'
    bsfile2 = pars.data_dir + 'TwoptBS_q' + qval + '_quark2_' + pars.lmbstring +'.pkl'
    # bsfile  = [ pars.data_dir + 'TwoptBS_q' + qval + '_num'+str(i)+'.pkl' for i in range(pars.numbers) ]
    # bsfile1 = [ pars.data_dir + 'TwoptBS_q' + qval + '_quark1_' + pars.lmbstring + '_num'+str(i)+'.pkl' for i in range(pars.numbers) ]
    # bsfile2 = [ pars.data_dir + 'TwoptBS_q' + qval + '_quark2_' + pars.lmbstring + '_num'+str(i)+'.pkl' for i in range(pars.numbers) ]

    #-----EvxptRead-----
    # for sink in [pars.sinkfold]:
    start = time.time()
    for i in range(pars.numbers):
        # print(pars.evxptdir+pars.momfold[pars.momentum]+'rel/dump/dump.res')
        if pars.kappas==1: 
            nucleon[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+pars.momfold[pars.momentum]+'rel_cosine/dump/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
            nucleongu[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+pars.momfold[pars.momentum]+pars.lmbstring+'/rel_cosine/dump/quark1/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
            nucleongd[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+pars.momfold[pars.momentum]+pars.lmbstring+'/rel_cosine/dump/quark2/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
        else:
            nucleon[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+pars.momfold[pars.momentum]+'rel/dump/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
            nucleongu[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+pars.momfold[pars.momentum]+pars.lmbstring+'/rel/dump/quark1/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
            nucleongd[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+pars.momfold[pars.momentum]+pars.lmbstring+'/rel/dump/quark2/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
        # nucleon[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir + '/rel/dump/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
        # nucleongu[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+pars.lmbstring+'/rel/dump/quark1/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
        # nucleongd[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+pars.lmbstring+'/rel/dump/quark2/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
    end = time.time()
    print('readevxptdump time: \t', end-start)

    # print(f"{np.shape(nucleongu)=}")
    # print(f"{np.shape(nucleongd)=}")
    # print(f"{nucleongd=}")

    # pd.DataFrame( np.array([[ nucleon[i][0][j].values for j in range(len(nucleon[i][0])) ] for i in range(pars.numbers)] ) ).to_pickle(bsfile)
    with open(bsfile, 'wb') as fileout:
        pickle.dump(np.array([[ nucleon[i][0][j].values for j in range(len(nucleon[i][0])) ] for i in range(pars.numbers)] ), fileout)
    with open(bsfile1, 'wb') as fileout:
        pickle.dump(np.array([[ nucleongu[i][0][j].values for j in range(len(nucleongu[i][0])) ] for i in range(pars.numbers)] ), fileout)
    with open(bsfile2, 'wb') as fileout:
        pickle.dump(np.array([[ nucleongd[i][0][j].values for j in range(len(nucleongd[i][0])) ] for i in range(pars.numbers)] ), fileout)

if __name__ == "__main__":
    if len(sys.argv)>3:
        print("Starting Pandapickle")
        momentum   = int(sys.argv[1])
        kappas     = int(sys.argv[2]) 
        lambdachoice   = int(sys.argv[3])
        sinktype=0
        pars = pr.params(kappas, sinktype, momentum)
        pars.fit = 'pandapickle'
        pars.makeresultdir()
        # pars.lmbstring = ['lp001', 'lp01', 'lp02', 'lp04', 'lp08'][lambdachoice]
        pars.lmbstring = pars.lmblist[lambdachoice]
        # Because we're only doing g4 operator insertions
        pars.numbers=2
        pickledata(pars, imag=0)
    else:
        print("arg[1] = momentum")
        print("arg[2] = kappas")
        print("arg[3] = lambdachoice")

