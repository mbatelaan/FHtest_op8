from numpy import zeros, size, sort, exp, average,split,append,multiply,sum,sqrt, linspace
from numpy import array as array
#from array import array
#from BootStrap import BootStrap1
from BootStrap3 import BootStrap

def ReadEvxpt(file,par):
    FF=[]
    f=open(file)
    for line in f:
        strpln=line.rstrip()
        if len(strpln) > 0:
            if strpln[0]=="+" and strpln[1]=="F" and strpln[2]=="I" and strpln[5]=="0":
                tmp=f.readline()
                tmp=f.readline()
                tmp=f.readline()
                tmp=f.readline()
                tmp=f.readline()
                tmp=f.readline()
                tmp=f.readline()
                tmp=f.readline().split()
                boots=int(tmp[3])
                tmp=f.readline()
                FF.append(BootStrap(boots,68))
                for iff in range(par+1):
                    tmp=f.readline().split()
                    if par==iff:
                        FF[-1].Avg=float(tmp[2])
                        FF[-1].Std=float(tmp[3])
                    # if iff==(nff-1):
                    #     baddata=append(baddata,float(tmp[2]))
                tmp=f.readline().split()
                while tmp[0]!="+NUmbers="+str(boots):
                    tmp=f.readline().split()
                if tmp[0]=="+NUmbers="+str(boots):
                    for iboot in range(boots):
                        tmp=f.readline().split()
                        FF[-1].values[iboot]=float(tmp[2*par+4])
    f.close()
    return FF

def ReadEvxptdump(file, par, boots, confs=0, times=64, number=0, bin=1):
    # par should be 0 or 1 depending on whether the real or imaginary part is chosen (0=R,1=I)
    # number defines which result you want, if there are multiple in the same file
    f=open(file)
    FF=[]
    for line in f:
        strpln=line.rstrip()
        if len(strpln) > 0:
            if strpln[0]=="+" and strpln[1]=="E" and strpln[2]=="N":
                tmp = f.readline()
                tmp = f.readline()
                tmp = f.readline().split()
                times = int(tmp[5])
            if strpln[0]=="+" and strpln[1]=="R" and strpln[2]=="P" and int(strpln[4:6])==number:
                tmp = f.readline().split()
                while tmp[0] != "nmeas":
                    tmp = f.readline().split()
                confs = int(tmp[2])
                G = zeros(shape=(confs,times,2))
            if strpln[0]=="+" and strpln[1]=="R" and strpln[2]=="D" and int(strpln[4:6])==number:
                for iff in range(confs):
                    for nt in range(times):
                        tmp=f.readline().split()
                        G[iff,nt,0] = tmp[1]
                        G[iff,nt,1] = tmp[2]
    f.close()
    for j in range(times):
        FF.append(BootStrap(boots,68))
        FF[-1].Import(G[:,j,par], bin=bin)
        FF[-1].Stats()
    return FF

def ReadEvxptData(file, par):
    f = open(file)
    x,y,err = [], [], []
    for line in f:
        strpln = line.rstrip()
        if len(strpln) > 0:
            if strpln[0] == "+" and strpln[1] == "D" and strpln[2] == "A" and strpln[6] == str(par):
                tmp = f.readline().split()
                while tmp[0][0] != "+" and tmp[0][1] != "N" and tmp[0][2] != "U":
                    tmp = f.readline().split()
                if tmp[0][0] == "+" and tmp[0][1] == "N" and tmp[0][2] == "U":
                    for  i in range(int(tmp[0][9:])):
                        tmp = f.readline().split()
                        x.append(float(tmp[0]))
                        y.append(float(tmp[1]))
                        err.append(float(tmp[2]))
    return x, y, err
