'''Created on Feb20, 2014
SVM classify
@author: Aidan
'''
from numpy import *
import inspect 
from time import sleep


def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


class svmTrain:
    '''This is the SVM Train'''
    def __init__(self,dataMatIn, classLabels, C = 0.06, toler = 0.001, kTup=('lin', 0)):  # Initialize the structure with the parameters 
        self.surpportVectorAlphas = None# column vector
        self.surpportVector = None
        self.surpportVectorClass = None# column vector
        self.b = 0
        
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.totalcount = shape(dataMatIn)[0]
        self.feturecount = shape(dataMatIn)[1]
        self.alphas = mat(zeros((self.totalcount,1)))# column vector
        
        self.eCache = mat(zeros((self.totalcount,2))) #first column is valid flag
        self.KTup = kTup
        self.K = mat(zeros((self.totalcount,self.totalcount)))
        for i in range(self.totalcount):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], self.KTup)

    def svmSurpportVecsGet(self):
        '''
         get the surpport vectors'''
        self.surpportVectorAlphas = (self.alphas[self.alphas>0]).T
        surpportVecIndex = nonzero(self.alphas.A > 0)[0]
        self.surpportVector = self.X[surpportVecIndex]
        self.surpportVectorClass = self.labelMat[surpportVecIndex]
        return self.surpportVectorAlphas, self.surpportVector,\
               self.surpportVectorClass

    def svmClassify(self, dataIn):
        K = kernelTrans(self.surpportVector, dataIn, self.KTup)
        predict = K.T * multiply(self.surpportVectorAlphas, self.surpportVectorClass) +self.b
        return sign(predict).item()
        #print 'predict', predict
        return predict

    def __calcEk(self, k):
        fXk = float(multiply(self.alphas,self.labelMat).T*self.K[:,k] + self.b)
        Ek = fXk - float(self.labelMat[k])
        return Ek

    def __selectJrand(self, i,m):
        j=i #we want to select any J not equal to i
        while (j==i):
            j = int(random.uniform(0,m))
        return j

                                   
    def __selectJ(self, i, Ei):         #this is the second choice -heurstic, and calcs Ej
        maxK = -1; maxDeltaE = 0; Ej = 0
        self.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
        validEcacheList = nonzero(self.eCache[:,0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue #don't calc for i, waste of time
                Ek = self.__calcEk(k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:   #in this case (first time around) we don't have any valid eCache values
            j = self.__selectJrand(i,self.totalcount)
            Ej = self.__calcEk(j)
        return j, Ej

    def __updateEk(self, k):#after any alpha has changed update the new value in the cache
        Ek = self.__calcEk(k)
        self.eCache[k] = [1,Ek]

    def __clipAlpha(self, aj,H,L):
        if aj > H: 
            aj = H
        if L > aj:
            aj = L
        return aj
        
    def __updateAlphaPair(self, i):
        Ei = self.__calcEk(i)
        if ((self.labelMat[i]*Ei < -self.tol) and (self.alphas[i] < self.C)) or \
           ((self.labelMat[i]*Ei > self.tol) and (self.alphas[i] > 0)):
            j,Ej = self.__selectJ(i, Ei) #this has been changed from selectJrand
            alphaIold = self.alphas[i].copy(); alphaJold = self.alphas[j].copy();
            if (self.labelMat[i] != self.labelMat[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L==H:
                print "L==H";
                return 0
            eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j] #changed for kernel
            if eta >= 0:
                print "eta>=0";
                return 0
            self.alphas[j] -= self.labelMat[j]*(Ei - Ej)/eta
            self.alphas[j] = self.__clipAlpha(self.alphas[j],H,L)
            self.__updateEk(j) #added this for the Ecache
            if (abs(self.alphas[j] - alphaJold) < 0.00001):
                #print "j not moving enough";
                return 0
            self.alphas[i] += self.labelMat[j]*self.labelMat[i]*(alphaJold - self.alphas[j])#update i by the same amount as j
            self.__updateEk(i) #added this for the Ecache                    #the update is in the oppostie direction
            b1 = self.b - Ei- self.labelMat[i]*(self.alphas[i]-alphaIold)*self.K[i,i] \
                 - self.labelMat[j]*(self.alphas[j]-alphaJold)*self.K[i,j]
            b2 = self.b - Ej- self.labelMat[i]*(self.alphas[i]-alphaIold)*self.K[i,j] \
                 - self.labelMat[j]*(self.alphas[j]-alphaJold)*self.K[j,j]
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]): self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]): self.b = b2
            else: self.b = (b1 + b2)/2.0
            return 1
        else: return 0

    def smoP(self, maxIter = 40):    #full Platt SMO
        iter = 0
        entireSet = True;
        alphaPairsChanged = 0
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:   #go over all
                for i in range(self.totalcount):        
                    alphaPairsChanged += self.__updateAlphaPair(i)
                    #print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            else:#go over non-bound (railed) alphas
                nonBoundIs = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.__updateAlphaPair(i)
                    #print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                iter += 1
            if entireSet:
                entireSet = False #toggle entire set loop
            elif (alphaPairsChanged == 0):
                entireSet = True  
            #print "iteration number: %d" % iter
        return self.b,self.alphas


class svmClassifer:
    ''' This is SVM classifer, the default type of the surpport vector
    is numpy matrix. in order to store the classifier with JSON, call
    svmSurpportVec2List() before save. call svmSurpportVecList2M()
    before use the classifier'''
    def __init__(self,b, surpportVectorAlphas,\
                 surpportVector, surpportVectorClass, KTup =('lin', 0), **args):
        obj_list = inspect.stack()[1][-2]
        self.__name__ = obj_list[0].split('=')[0].strip()
        
        self.surpportVectorAlphas = surpportVectorAlphas
        self.surpportVector = surpportVector
        self.surpportVectorClass = surpportVectorClass
        self.KTup = KTup
        self.b = b

    def svmSurpportVecsGet(self):
        '''
         get the surpport vectors'''
        return self.surpportVectorAlphas,\
               self.surpportVector,self.surpportVectorClass
    
    def svmSurpportVecSet(self,VectorAlphas, Vector, VectorClass):
        self.surpportVectorAlphas = VectorAlphas
        self.surpportVector = Vector
        self.surpportVectorClass = VectorClass

    def jsonDumpsTransfer(self):
        '''essential transformation to Python basic type in order to
        store as json. dumps as objectname.json if filename missed '''
        #pdb.set_trace()
        self.surpportVectorAlphas = self.surpportVectorAlphas.tolist()
        self.surpportVector = self.surpportVector.tolist()
        self.surpportVectorClass = self.surpportVectorClass.tolist()
        self.b = self.b.tolist()

    def jsonDumps(self, filename=None):
        '''dumps to json file'''
        self.jsonDumpsTransfer()
        if not filename:
            jsonfile = self.__name__+'.json'
        else: jsonfile = filename
        objectDumps2File(self, jsonfile)

    '''def svmSurpportVecM2List(self):
        self.surpportVectorAlphas = self.surpportVectorAlphas.tolist()
        self.surpportVector = self.surpportVector.tolist()
        self.surpportVectorClass = self.surpportVectorClass.tolist()
        self.b = self.b.tolist()'''
        
    def jsonLoadTransfer(self):
        self.surpportVectorAlphas = mat(self.surpportVectorAlphas)
        self.surpportVector = mat(self.surpportVector)
        self.surpportVectorClass = mat(self.surpportVectorClass)
        self.b = mat(self.b)
    
    def svmClassify(self, dataIn):
        K = kernelTrans(self.surpportVector, dataIn, self.KTup)
        predict = K.T * multiply(self.surpportVectorAlphas, self.surpportVectorClass) +self.b
        return sign(predict).item()
        #print 'predict', predict
        return predict

    

            
