# runBibliometricSparseCoding.py
#
# Bryan Daniels
# 10.20.2015
#
# Branched from runSparsenessProblem.py
#
# Changing to remove references to lmbda (only lmbda2 now)
#
# Removed: kappa (4.18.2018)
#
# Things to consider removing:
#   sortedRules

#import sys
#sys.path.append("../")

from Sparseness import *
from SparsenessPrediction import *
from outputTag import nextFileNumString
#import sys
from scipy import sort #, stats
import time

from readData import *

outputDirectory = '.'

# set tag to command-line argument unless there isn't a valid one
if len(sys.argv) < 2:
    prefix = nextFileNumString(outputDirectory)
elif (sys.argv[1].find('&') != -1) or (sys.argv[1].find('>') != -1)         \
    or (sys.argv[1].find('|') != -1):
    prefix = nextFileNumString(outputDirectory)
else:
    prefix = sys.argv[1]

descriptiveString = ""

lmbda = -scipy.inf #0.05
lmbda2 = 0.5
sigma = 1.e-2 # 1.e-1
sigma2 = 1. #1. # 0.5 # 1. #5.e-3
c = 0.05
constrainRows = False
constrainCols = False
avextol = 1.e-6 # 1.e-7 
maxiter = 250 #None 
orthogStart = True

SName = 'SLog' #'S2'
squashFuncName = 'squash' #'squashShifted' # 'identity' 

indVar = 'lmbda2'

# do prediction for selected values of lmbda2
        
lmbda2List = list( scipy.arange(-1.5,0.2,0.2) )
log = True

lmbda2List.reverse() # use to start large

# use participants that appear more than once
if False:
    nameDict = over1nameDict
    dataMatrix = over1participantsMatrix
else:
    nameDict = over5nameDict
    dataMatrix = over5participantsMatrix


N = scipy.shape(dataMatrix)[-1] # 141
def dataFunc(shuffleSeed): # include half as in-sample, half as out-of-sample
    scipy.random.seed(shuffleSeed)
    dataNonzeroOnly = filter(lambda x: sum(x) > 0,dataMatrix)
    shuffledData = shuffle(dataNonzeroOnly) # 6.9.2014
    numNonzeroManuscripts = len(shuffledData)
    inSampleData = shuffledData[:numNonzeroManuscripts/2]
    outOfSampleData = shuffledData[numNonzeroManuscripts/2:]
    return inSampleData,outOfSampleData



    
runFullBasisPrediction = True
zeroUnseenPairs = False
shiftJDiag = False
findJmatrix = False # for comparing with Ising
Jmatrix = None
skipPredictNext = False #True

seedList = [200] #200+50*int(sys.argv[2])+scipy.arange(50) #[0]
shuffleSeedList = [0] #range(5,10) #[int(sys.argv[2])] #[9] #range(8,10) #range(10) #range(6,8)
skipSeedList = [] #[ (8,seed) for seed in range(5) ] #[ (4,0), (4,1), (4,2) ]
randomStart = True # if False, use PCA start 
randomAMult = 1. 
randomPhiMult = 1. 


# 2.7.2011 for running just prediction using pre-minimized bases
#preMinimizedFullPrefix = 'v0099_vary_lmbda_lmbda2_shuffledNoUd_sigma_'+     \
#    '1e-2_sigma2_1._SLog_randomStart_NOrampSparseness_orthogStart_'+        \
#    'outOfSample500600_N47_lmbda0_lmbda2lineFocused_avextol1e-6_'+          \
#    'constrainSizeNew_NOincreasedLmbda2Prediction_squash'
#updateDataDictDict = False

# "normal" behavior
preMinimizedFullPrefix = None
updateDataDictDict = False



def minimizeAndStoreBasis(dataDictDict,lmbda,lmbda2,prevLmbda2=None):
    
    # for testing
    #print "minimizeAndStoreBasis: "
    #print (prevLmbda2),"-->",(lmbda2)
    #return {}
    # end for testing
    
    if log:
        lmbda,lmbda2 = 10**lmbda, 10**lmbda2
        if prevLmbda2 is not None: prevLmbda2 = 10**prevLmbda2
    
    # 8.17.2011 change lmbda if really close to one we've already done
    eps = 1e-6
    ks = dataDictDict.keys()
    for k in ks:
        lmbda2Key = k
        if abs(lmbda2Key-lmbda2) < eps:
            lmbda2 = lmbda2Key
        
    if dataDictDict.has_key(lmbda2):
        # 2.7.2011 skip minimization if we've already done it
        if updateDataDictDict:
            dataDict = dataDictDict[lmbda2]
            sp = dataDict['sp']
            phi = dataDict['phi']
            basis = dataDict['basis']
            dataDict['Efunc'] = sp.Efunc(sp.pack(phi,basis.T))
            dataDict['avgCost'] = sp.avgCost(phi,aBest=basis.T)
            dataDict['numNonzeroBasisVectors'] = sp.numNonzeroBasisVectors(basis.T)
            dataDict['avgGroupSize'] = sp.avgGroupSize(basis.T)
            dataDict['pid'] = os.getpid()
        save(scipy.transpose(inSampleData),                             \
            fullPrefixShuffleSeed+'_inSampleImages.data')
        return dataDictDict
    
    # 2.2.2011
    startTime = time.clock()
    
    # 2.10.2011
    if prevLmbda2 is None: # 2.1.2011
        startKey = None
        phiStart = None
        aStart = None
    elif dataDictDict.has_key(prevLmbda2):
        startKey = prevLmbda2
        phiStart = dataDictDict[startKey]['phi']
        aStart = dataDictDict[startKey]['basis'].T
    else:
        raise Exception("minimizeAndStoreBasis Error: dataDictDict "+  \
            "does not contain key "+str(prevLmbda2))
        
    sp = SparsenessProblem(                                             \
        timeSeriesDataMatrix(inSampleData,1).T,                         \
        SName,lmbda,sigma,N,c=c,veryVerbose=True,                       \
        constrainRows=constrainRows,constrainCols=constrainCols,        \
        lmbda2=lmbda2,sigma2=sigma2,seed=seed,phiStart=phiStart,        \
        aStart=aStart,randomStart=randomStart,randomAMult=randomAMult,  \
        randomPhiMult=randomPhiMult,avextol=avextol,                    \
        orthogStart=orthogStart,squashFuncName=squashFuncName,          \
        maxiter=maxiter)
    if not skipPredictNext:
        phi,basis = sp.findSparseRepresentation(True)
        #rules = sp.sortedRules(basis,includeIndices=True,              \
        #    threshold=sigma,nameDict=nameDict,phi=phi)
    else:
        phi,basis = scipy.array([[0,1]]),scipy.array([[0,1]])
        rules = []
    #contributions,basis = sp.basisVectorContributions(phi,True)
         
    # store data
    dataDict = {}
    dataDict['startKey'] = startKey # start from previous solution
    dataDict['nameDict'] = nameDict
    dataDict['sp'] = sp
    dataDict['phi'] = phi
    dataDict['basis'] = basis
    #dataDict['rules'] = rules
    dataDict['lmbda'] = lmbda
    dataDict['lmbda2'] = lmbda2
    dataDict['indVar'] = indVar
    dataDict['avextol'] = avextol
    dataDict['maxiter'] = maxiter
    dataDict['SName'] = SName
    if not skipPredictNext:
        dataDict['ENormFunc'] = sp.ENormFunc(sp.pack(phi,basis.T))
        dataDict['avgCost'] = sp.avgCost(phi,aBest=basis.T)
        dataDict['numNonzeroBasisVectors'] = sp.numNonzeroBasisVectors(basis.T)
        dataDict['avgGroupSize'] = sp.avgGroupSize(basis.T) 
    dataDict['minimizationTimeMinutes'] = (time.clock() - startTime)/60.
    dataDict['pid'] = os.getpid()

    # 3.26.2011
    save(sp.images,fullPrefixShuffleSeed+'_inSampleImages.data')
    save(scipy.transpose(outOfSampleData),                              \
        fullPrefixShuffleSeed+'_outOfSampleImages.data')
    dataDict['sp'].images = []
    
    dataDictDict[lmbda2] = dataDict
    save(dataDictDict,fullPrefixSeed+'_dataDictDict.data')
    
    return dataDictDict
    
def checkPredictionsAndStoreData(predictionsDictDict,dataDictDict,      \
    lmbda,lmbda2):
    
    # for testing
    #print "checkPredictions: ",lmbda2
    #return {}
    # end for testing
    
    if log:
        lmbda, lmbda2 = 10**lmbda, 10**lmbda2
    
    # 2.16.2011
    startTime = time.clock()
    
    # 8.17.2011 change lmbda if really close to one we've already done
    eps = 1e-6
    ks = dataDictDict.keys()
    for k in ks:
        lmbda2Key = k
        if abs(lmbda2Key-lmbda2) < eps:
            lmbda2 = lmbda2Key
    
    dataDict = dataDictDict[lmbda2]
    predictionsDict = copy.deepcopy(dataDict)
    
    sp = dataDict['sp']
    if len(sp.images) == 0:
        sp.images = load(fullPrefixShuffleSeed+'_inSampleImages.data')
    basis = dataDict['basis']
    
    if runFullBasisPrediction:
        #percentRight,percentRightFreq,percentRightCov,percentRightPCA =     \
        #    checkPredictions(sp,basis,outOfSampleData)
        #percentRight,percentRightFreq,percentRightPCA,percentRightIsing,    \
        #    percentRightPerfect,meanCost =                                  \
        #    checkPredictions(sp,basis,outOfSampleData,fixSingularC=True)
        propRightList,numRightLists,predictedLists,                          \
            predictionFightsLists,numNonzeroAndEntLists =                    \
              checkPredictions(sp,basis,outOfSampleData,
                fixSingularC=True,returnLists=True,
                zeroUnseenPairs=zeroUnseenPairs,shiftJDiag=shiftJDiag,
                findJmatrix=findJmatrix,Jmatrix=Jmatrix,
                skipPredictNext=skipPredictNext)
        numNonzeroLists,entLists = numNonzeroAndEntLists
        percentRight,percentRightFreq,percentRightPCA,percentRightIsing,    \
            percentRightPerfect,meanCost = propRightList
        predictionsDict['percentRight'] = percentRight
        predictionsDict['percentRightFreq'] = percentRightFreq
        #predictionsDict['percentRightCov'] = percentRightCov
        predictionsDict['percentRightPCA'] = percentRightPCA
        predictionsDict['percentRightIsing'] = percentRightIsing
        predictionsDict['percentRightPerfect'] = percentRightPerfect
        predictionsDict['meanReconstructionCost'] = meanCost
        predictionsDict['predictionTimeMinutes'] = (time.clock() - startTime)/60.
        predictionsDict['predictionFightsLists'] = predictionFightsLists
        predictionsDict['predictedLists'] = predictedLists
        predictionsDict['numRightLists'] = numRightLists 

    
    predictionsDictDict[lmbda2] = predictionsDict
    sp.images = []
    save(predictionsDictDict,fullPrefixSeed+'_predictionsDictDict.data')
    
    # could also eventually include numNonzero stuff here; see runSparsenessProblem.py
    
    return predictionsDictDict



phiStart,aStart = None,None

fullPrefix = prefix+"_vary_"+indVar+"_"+descriptiveString

for shuffleSeed in shuffleSeedList:
  inSampleData,outOfSampleData = dataFunc(shuffleSeed)
  fullPrefixShuffleSeed = fullPrefix + '_shuffleSeed'+str(shuffleSeed)
  for seed in seedList:
    fullPrefixSeed = fullPrefix +                                           \
            '_shuffleSeed'+str(shuffleSeed)+'_seed'+str(seed)

    if preMinimizedFullPrefix is not None:
      dataDictDict = load(                                                  \
          preMinimizedFullPrefix+'_seed'+str(seed)+'_dataDictDict.data')
    else:
      dataDictDict = {}
    predictionsDictDict = {}
  
    if ( (shuffleSeed,seed) not in skipSeedList ):
      
      prevLmbda2 = None
      
      for lmbda2Des in lmbda2List:
        dataDictDict = minimizeAndStoreBasis(dataDictDict,                  \
            lmbda,lmbda2Des,prevLmbda2=prevLmbda2)
        predictionsDictDict = checkPredictionsAndStoreData(                 \
            predictionsDictDict,dataDictDict,lmbda,lmbda2Des)
            
  
        
