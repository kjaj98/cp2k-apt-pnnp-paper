import numpy as np
import math

class Box:
#    def __init__(self,a1,a2,a3,b1,b2,b3,c1,c2,c3):
    def __init__(self, *args, **kwargs):
        self.mTrans = np.zeros((3,3))

        # This will read the different cp2k options how to define a cell
        if 'cellfn' in kwargs:
            with open(kwargs['cellfn'], 'r') as fin:
                for line in fin:
                    aLine = line.split()
                    if(len(aLine) == 0):
                        continue

                    if aLine[0] == 'A':
                        kwargs['a'] = [float(aLine[1]), float(aLine[2]), float(aLine[3])]
                    if aLine[0] == 'B':
                        kwargs['b'] = [float(aLine[1]), float(aLine[2]), float(aLine[3])]
                    if aLine[0] == 'C':
                        kwargs['c'] = [float(aLine[1]), float(aLine[2]), float(aLine[3])]

                    if aLine[0] == 'ABC':
                        kwargs['abc'] = [float(aLine[1]), float(aLine[2]), float(aLine[3])]
                    if aLine[0] == 'ALPHA_BETA_GAMMA':
                        kwargs['alpha_beta_gamma'] = [float(aLine[1]), float(aLine[2]), float(aLine[3])]
        

        if 'a' in kwargs and 'b' in kwargs and 'c' in kwargs:
            self.mTrans[0][0] = kwargs['a'][0]
            self.mTrans[0][1] = kwargs['a'][1] 
            self.mTrans[0][2] = kwargs['a'][2] 
            self.mTrans[1][0] = kwargs['b'][0] 
            self.mTrans[1][1] = kwargs['b'][1] 
            self.mTrans[1][2] = kwargs['b'][2] 
            self.mTrans[2][0] = kwargs['c'][0] 
            self.mTrans[2][1] = kwargs['c'][1] 
            self.mTrans[2][2] = kwargs['c'][2] 

        # THIS IF CLAUSE IS OLD (BACKWARD COMPATIBILITY)
        if len(args) == 9:
            self.mTrans[0][0] = args[0]
            self.mTrans[0][1] = args[1]
            self.mTrans[0][2] = args[2]
            self.mTrans[1][0] = args[3]
            self.mTrans[1][1] = args[4]
            self.mTrans[1][2] = args[5]
            self.mTrans[2][0] = args[6]
            self.mTrans[2][1] = args[7]
            self.mTrans[2][2] = args[8]

        if 'abc' in kwargs and 'alpha_beta_gamma' in kwargs:
            abc = kwargs['abc']
            alpha_beta_gamma = kwargs['alpha_beta_gamma']
            alpha = alpha_beta_gamma[0] / 180.0 * math.pi
            beta = alpha_beta_gamma[1] / 180.0 * math.pi
            gamma = alpha_beta_gamma[2] / 180.0 * math.pi

            self.mTrans[0][0] = abc[0]
            self.mTrans[0][1] = 0.0
            self.mTrans[0][2] = 0.0

            self.mTrans[1][0] = abc[1] * math.cos(gamma)
            self.mTrans[1][1] = abc[1] * math.sin(gamma)
            self.mTrans[1][2] = 0.0

            self.mTrans[2][0] = abc[2] * math.cos(beta)
            self.mTrans[2][1] = (abc[2] * math.cos(alpha) - self.mTrans[2][0] * math.cos(gamma)) / math.sin(gamma)
            self.mTrans[2][2] = math.sqrt(math.pow(abc[2],2.0) - math.pow(self.mTrans[2][1], 2.0) - math.pow(self.mTrans[2][0],2.0))

            self.mTrans = self.mTrans.transpose()

        
        self.mInvTrans = np.linalg.inv(self.mTrans)
#        print(self.mTrans)
#        print("==============")
#        print(self.mInvTrans)

    def pbc(self,v):
#        print(v)
#        print(self.mInvTrans)
        vInt = self.mInvTrans.dot(v)
#        print(vInt)
        vCorr = vInt.round()
#        print(vCorr)
#        print(self.mTrans.dot(vInt - vCorr))
        return self.mTrans.dot(vInt - vCorr)
    
    def pbc_array(self,v):
        vInt = np.einsum('ij,nj->ni', self.mInvTrans, v)
        vCorr = vInt.round()
        return np.einsum('ij,nj->ni', self.mTrans, vInt - vCorr)

    def abs2frac(self, v):
        return self.mInvTrans.dot(v)

    def frac2abs(self, v):
        return self.mTrans.dot(v)

