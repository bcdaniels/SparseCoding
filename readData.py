# readData.py
#
# 4.18.2018
#
# Make participants matrix for use with sparse coding code, starting
# with dictionary from Dirk Wintergruen.
#


import scipy,pylab
import pandas
from simplePickle import load

#filename = 'Data/180418/coms2.pickle' # smaller initial dataset
filename = 'Data/180418/all.pickle'


def makeParticipantsMatrix(participantsDataDict,groupSubsetDict,
    verbose=False):
    """
    Create participants binary matrix with shape
    (#meetings)x(#participants).
    
    groupSubsetDict         : Dictionary mapping names to indices
    """
    numMeetings = len(participantsDataDict)
    numParticipants = len(groupSubsetDict.keys())
    participantsMatrix = scipy.zeros((numMeetings,numParticipants))
    #skippedIndividuals = []
    for meetingi,participants in enumerate(participantsDataDict.values()):
        #boutParticipants = scipy.zeros(numCitedPapers)
        for participant in participants:
            if participant in groupSubsetDict.keys():
                i = groupSubsetDict[participant]
                participantsMatrix[meetingi][i] = 1
            # else:
            #     skippedIndividuals.append(participant)
    #if verbose: print "Skipped individuals:",skippedIndividuals
    return participantsMatrix


# () load data

rawData = load(filename)

# find unique authors
allParticipants = []
for meeting in rawData.values():
    allParticipants.extend(meeting)
uniqueParticipants = scipy.unique(allParticipants)

# translates index to name
allParticipantsDict = dict( enumerate(uniqueParticipants) )
# translates name to index
allIndexDict = dict((v,k) for k, v in allParticipantsDict.iteritems())

# allParticipantsMatrix has shape (#meetings)x(#uniqueParticipants)
# currently takes a couple of minutes
allParticipantsMatrix = makeParticipantsMatrix(rawData,allIndexDict)

# rank participants by frequency
freqs = scipy.sum(allParticipantsMatrix,axis=0)
participantIndicesOver1 = pylab.find(freqs>1)
participantIndicesOver5 = pylab.find(freqs>5)

# translates name to index
over1indexDict = dict( (allParticipantsDict[i],newIndex) \
    for newIndex,i in enumerate(participantIndicesOver1) )
over5indexDict = dict( (allParticipantsDict[i],newIndex) \
    for newIndex,i in enumerate(participantIndicesOver5) )
# translates index to name
over1nameDict = dict( (v,k) for k, v in over1indexDict.iteritems())
over5nameDict = dict( (v,k) for k, v in over5indexDict.iteritems())

over1participantsMatrix = makeParticipantsMatrix(rawData,over1indexDict)
over5participantsMatrix = makeParticipantsMatrix(rawData,over5indexDict)











