#!/usr/bin/env python
import numpy as np
from array import array

startseed=1234

class BootStrap:
    def __init__(self,Nboot,Confidence):
        self.nboot=Nboot
        self.confidence=Confidence
        self.Avg=0.0
        self.Std=0.0
        self.values=np.zeros((self.nboot))
        self.Raw=np.zeros(0)
    def Stats(self):
        self.Avg=np.average(self.values)
        tmp=(self.values-self.Avg)**2
        self.Std=np.sqrt(np.average(tmp)*self.nboot/(self.nboot-1))
        return
    def OldStats(self):
        tmp=np.sort(self.values,0)
        omit=(100-self.confidence)/2
        ilo=int((omit*self.nboot)/100)
        ihi=int(self.nboot-1-(omit*self.nboot)/100)
        self.Std=(tmp[ihi]-tmp[ilo])/2
        self.Avg=np.average(self.values)
        #self.Avg=np.average(tmp[ilo:ihi])
        return
    def Import(self,data,bin=1,weights=[]):
        if len(weights)>0 and bin!=1:
            print ("Weights only currently supported for binsize=1")
            assert 1==0
        self.nconf=int(len(data)/bin)
        if bin>1:
            tmpdata=np.array([np.average(ien) for ien in np.split(np.array(data[0:bin*(int(len(data)/bin))]),self.nconf)])
        else:
            tmpdata=np.array(data)
        #print(startseed)
        #print(type(self.nconf))
        myseed=startseed*self.nconf/self.nboot
        #print(myseed)
        #print(type(myseed))
        np.random.seed(int(myseed))
        #locranint=np.random.random_integers
        locranint=np.random.randint
        #print(type(self.nconf))
        if len(weights)>0:
            tmpweight=np.array(weights)
            self.Avg=np.average(np.multiply(tmpdata,tmpweight))/np.sum(tmpweight)
        else:
            self.Avg=np.average(tmpdata)
        for iboot in range(self.nboot):
            rint=locranint(0,int(self.nconf)-1,int(self.nconf))
            if len(weights)>0:
                tw2=tmpweight[rint]                
                td2=np.multiply(tmpdata[rint],tw2)
                self.values[iboot]=np.average(td2)/np.sum(tw2)
            else:
                self.values[iboot]=np.average(tmpdata[rint])
    def Create(self,mean,stdev):
        self.Avg=mean
        self.Std=stdev
        self.values=np.random.normal(mean,stdev,self.nboot)
    def Constant(self,const):
        self.Avg=const
        self.values.fill(const)
    def write(self,file):
        iclose=False
        try:
            # If file is not open, open it as a fileobject
            fo=open(file,'wb')
            iclose=True
        except:
            # If already a fileobject, we will append to it
            fo=file
        np.array(self.nboot).tofile(fo)
        np.array(self.Avg).tofile(fo)
        np.array(self.Std).tofile(fo)
        self.values.tofile(fo)
        
        # If file was a new file, close it
        if iclose:
            fo.close()
    def read(self,file):
        iclose=False
        try:
            # If file is not open, open it as a fileobject
            fo=open(file,'rb')
            iclose=True
        except:
            # If already a fileobject, we will append to it
            fo=file
        ti=array('i')
        ti.read(fo,1)
        self.nboot=ti[0]
        tf=array('d')
        tf.read(fo,2)
        self.Avg=tf[0]
        self.Std=tf[1]
        tf=array('d')
        tf.read(fo,self.nboot)
        self.values=np.array(tf)
        
        # If file was a new file, close it
        if iclose:
            fo.close()
    def __mul__(self,fac):
        result=BootStrap(self.nboot, self.confidence)
        try:
            real=float(fac)
            result.Avg=self.Avg*real
            result.values=self.values*real            
        except:
            try:
                tnboot=fac.nboot
                result.Avg=self.Avg*fac.Avg
                result.values=self.values*fac.values
            except:
                print ("ERROR: UNknown boot multiply")
                print (tnboot,fac.nboot)
                assert 1==0
        return result
    def __rmul__(self,real=float(1)):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=self.Avg*real
        result.values=self.values*real
        return result
    def __imul__(self,real=float(1)):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=self.Avg*real
        result.values=self.values*real
        return result
    def __add__(self,fac):
        result=BootStrap(self.nboot, self.confidence)
        try:
            real=float(fac)
            result.Avg=self.Avg+real
            result.values = self.values+real
            # for iboot in range(self.nboot):
            #     result.values[iboot]=self.values[iboot]+real            
        except:
            try:
                tnboot=fac.nboot
                result.Avg=self.Avg+fac.Avg
                # for iboot in range(self.nboot):
                #     result.values[iboot]=self.values[iboot]+fac.values[iboot]
                result.values = self.values+fac.values
            except:
                assert 1==0
                print ("ERROR: UNknown boot multiply")
        result.Stats()
        return result
    def __radd__(self,real=float(1)):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=self.Avg+real
        for iboot in range(self.nboot):
            result.values[iboot]=self.values[iboot]+real
        return result
    def __iadd__(self,real=float(1)):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=self.Avg+real
        for iboot in range(self.nboot):
            result.values[iboot]=self.values[iboot]+real
        return result
    def __sub__(self,fac):
        result=BootStrap(self.nboot, self.confidence)
        try:
            real=float(fac)
            result.Avg=self.Avg-real
            for iboot in range(self.nboot):
                result.values[iboot]=self.values[iboot]-real            
        except:
            try:
                tnboot=fac.nboot
                result.Avg=self.Avg-fac.Avg
                for iboot in range(self.nboot):
                    result.values[iboot]=self.values[iboot]-fac.values[iboot]
            except:
                assert 1==0
                print ("ERROR: UNknown boot multiply")
        return result
    def __rsub__(self,real=float(1)):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=real-self.Avg
        for iboot in range(self.nboot):
            result.values[iboot]=real-self.values[iboot]
        return result
    def __isub__(self,real=float(1)):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=self.Avg-real
        for iboot in range(self.nboot):
            result.values[iboot]=self.values[iboot]-real
        return result
    def __pow__(self,real):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=self.Avg**real
        result.values=self.values**real
        return result
    def exp(self,real):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=np.exp(self.Avg*real)
        for iboot in range(self.nboot):
            result.values[iboot]=np.exp(self.values[iboot]*real)
        return result
    def __abs__(self):
        result=BootStrap(self.nboot, self.confidence)
        result.Avg=abs(self.Avg)
        result.values=abs(self.values)
        return result
    def __ge__(self,real):
        tav=False
        tboot=False
        if self.Avg>=real:
            tav=True
        if (self.values>=real).all():
            tboot=True
        result=tav*tboot
        return result

