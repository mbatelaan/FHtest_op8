# -*- eval: (comment-tags-mode) -*-
#INFO: Resample the data from evxpt dumpresult files into bootstrap ensembles and save them as pickle files.
import numpy as np
import csv
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
    print("------------------------------")

    # -----Data Files-----
    qval = np.array2string(pars.qval[pars.momentum],separator='')[1:-1]
    # bsfile  = [ pars.data_dir + 'TwoptBS_q' + qval + '_num'+str(i)+'.pkl' for i in range(pars.numbers) ]
    # bsfile1 = [ pars.data_dir + 'TwoptBS_q' + qval + '_quark1_' + pars.lmbstring + '_num'+str(i)+'.pkl' for i in range(pars.numbers) ]
    # bsfile2 = [ pars.data_dir + 'TwoptBS_q' + qval + '_quark2_' + pars.lmbstring + '_num'+str(i)+'.pkl' for i in range(pars.numbers) ]
    bsfile  = pars.data_dir + 'TwoptBS_q' + qval + '.pkl'
    bsfile1 = pars.data_dir + 'TwoptBS_q' + qval + '_quark1_' + pars.lmbstring + '.pkl'
    bsfile2 = pars.data_dir + 'TwoptBS_q' + qval + '_quark2_' + pars.lmbstring +'.pkl'

    #-----EvxptRead-----
    # for sink in [pars.sinkfold]:
    start = time.time()
    for i in range(pars.numbers):
        nucleon[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir + '/dump/'+pars.lmbstring+'/unpert/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
        nucleongu[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+ '/dump/'+pars.lmbstring+'/quark1/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
        nucleongd[i].append(stats.normcorr(ReadEvxptdump(pars.evxptdir+ '/dump/'+pars.lmbstring+'/quark2/dump.res', imag, pars.nboot, number=i, bin=pars.nbin),pars.norma[0]))
    end = time.time()
    print('readevxptdump time: \t', end-start)

    # -----PICKLE-----
    # picklestart = time.time()
    # for i in range(pars.numbers):
    #     pd.DataFrame(np.array([ nucleon[i][0][j].values for j in range(len(nucleon[i][0]))     ]) ).to_pickle(bsfile[i])
    #     pd.DataFrame(np.array([ nucleongu[i][0][j].values for j in range(len(nucleongu[i][0])) ]) ).to_pickle(bsfile1[i])
    #     pd.DataFrame(np.array([ nucleongd[i][0][j].values for j in range(len(nucleongd[i][0])) ]) ).to_pickle(bsfile2[i])
    # pickleend = time.time()
    # print('pickle time: \t\t', pickleend-picklestart)

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
        pars.lmbstring = ['lp025', 'lp05'][lambdachoice]
        if pars.momentum==0: pars.numbers=2
        pickledata(pars, imag=0)
    else:
        print("arg[1] = momentum")
        print("arg[2] = kappas")
        print("arg[3] = lambdachoice")


# #-----UNPICKLE-----
# start0 = time.time()
# nucleonlist=[]
# nucleonlistgu=[]
# nucleonlistgd=[]
# for i in range(pars.numbers):
#     rdata = pd.read_pickle(bsfile[i])
#     rdatau = pd.read_pickle(bsfileu[i])
#     rdatad = pd.read_pickle(bsfiled[i])
#     nucleonlist.append([])
#     nucleonlistgu.append([])
#     nucleonlistgd.append([])
#     for t in range(len(rdata.index)):
#         nucleonlist[i].append(BootStrap(len(rdata.columns), 68))
#         nucleonlist[i][-1].values=rdata.loc[t,:]
#         nucleonlist[i][-1].Stats()
#         nucleonlistgu[i].append(BootStrap(len(rdatau.columns), 68))
#         nucleonlistgu[i][-1].values=rdatau.loc[t,:]
#         nucleonlistgu[i][-1].Stats()
#         nucleonlistgd[i].append(BootStrap(len(rdatad.columns), 68))
#         nucleonlistgd[i][-1].values=rdatad.loc[t,:]
#         nucleonlistgd[i][-1].Stats()
# end0 = time.time()

# print(np.shape(nucleonlist))
# print(nucleonlist[1])
# print(nucleonlist[1][0])

# print('-----------------------------------')
# print('unpickle time: \t\t', end0-start0)
# print('readevxptdump/unpickle time: \t', (end-start)/(end0-start0) )
# print('-----------------------------------')
