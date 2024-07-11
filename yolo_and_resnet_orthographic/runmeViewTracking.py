import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
#from programs.bgsub import bgsub
import pickle
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy import stats
from ultralytics import YOLO
import time
from programs.bgsub import bgsub

def createCostMatrix(fishList1, fishList2):
    fishList1, fishList2 = fishList1.copy(), fishList2.copy() # This is to stop modifying the inputs from the function
    indices = (0, 1) if len(fishList1) <= len(fishList2) else (1, 0) # This is to put them back into order
    smallestList, biggestList = (fishList1, fishList2) if len(fishList1) <= len(fishList2) else (fishList2, fishList1)
    amountOfFish2Add = len(biggestList) - len(smallestList)
        
    for _ in range(amountOfFish2Add):
        fish = np.ones((12,2))
        fish[...] = np.inf
        smallestList.append(fish)
    
    smallestList, biggestList = np.array(smallestList), np.array(biggestList)
    
    
    #temp = [np.array(smallestList), np.array(biggestList)]
    ## Lets order them back into fishList1 and fishList2
    #fishList1, fishList2 = temp[indices[0]], temp[indices[1]]
    
    # Lets turn them into a form that allows broadcasting
    # fishList2 will be the colomns
    biggestList = biggestList[None, ...]
    smallestList = smallestList[:,None, ...]
    
    #fishList2 = fishList2[None, ...]
    #fishList1 = fishList1[:,None,...]
    cost = np.sum( np.sum((biggestList - smallestList) ** 2, axis = -1) ** .5, axis = -1) / 12
    cost[cost == np.inf] = 999
    #print(cost.shape)
    #print(cost)
    return cost, indices[0]

def getLossCombinationsFast(pastFishList, newFishList):
    # We need the order of the smallest in a certain position to be able to kick out the fake fish
    # That is also why we return a flag in createCostMatrix
    smallestLen = len(pastFishList) if len(pastFishList) <= len(newFishList) else len(newFishList)
    pastFishList, newFishList = pastFishList.copy(), newFishList.copy()
    costMatrix, flag = createCostMatrix(pastFishList, newFishList)
    row_ind, col_ind = linear_sum_assignment(costMatrix)
    cost = costMatrix[row_ind, col_ind]
    
    row_ind = row_ind[:smallestLen]
    col_ind = col_ind[:smallestLen]
    cost = cost[:smallestLen]
    
#     return cost, row_ind, col_ind
    if flag == 0:
        return cost, row_ind, col_ind
    else:
        return cost, col_ind, row_ind

def closenessTracking(dataList):
    # This algorithm does not use optical flow because it takes significantly longer
    # and the frame rate is so high that this algorithm is still good without it
    # The full algorithm for tracking with improved storage
    trackingList = []
    # On Average how close should the keypoints be to be considered the past fish
    proximityThreshold = 20

    amountOfFrames = len(video)
    #amountOfFrames = 2000
    data0 = dataList[0]
    keypointsList = data0[2]
    boxList = data0[1]

    lastFishSeen = 0
    instanceList = []
    for fishIdx, (box, keypoints) in enumerate(zip(boxList, keypointsList)):
        instanceList.append([lastFishSeen, box, keypoints])
        lastFishSeen += 1

    trackingList.append(instanceList)
    pastFrame = video[0]
    # amountOfFrames = 1000
    for frameIdx in tqdm(range(1,amountOfFrames)):
        # Getting the data for this frame
        data = dataList[frameIdx]
        keypointsList = data[2]
        boxList = data[1]

        #data = [ [box, keypoints] for (box, keypoints) in zip(boxList, keypointsList) if np.any(keypoints[:,1] > 144) ]
        data = [ [box, keypoints] for (box, keypoints) in zip(boxList, keypointsList) ]
        boxList = [box for (box, keypoints) in data]
        keypointsList = [keypoints for (box, keypoints) in data]

        pastKeypointsList = trackingList[-1]
        #newKeypointsListDummy = [None for _ in pastKeypointsList]
        newKeypointsListDummy = []

        # Lets get only the fish that can still be tracked
        trackableKeypointsIndices = [idx for idx, el in enumerate(pastKeypointsList) ]
        trackableKeypoints = [ pastKeypointsList[idx][2] for idx in trackableKeypointsIndices  ]

    #     # Estimating where the list should be based on optical flow
    #     nextFrame = video[frameIdx]
    #     flow = cv.calcOpticalFlowFarneback(pastFrame, nextFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     xFlow, yFlow = flow[..., 0], flow[..., 1]

        estimateKeypointsList = []
        for keypoints in trackableKeypoints:
            estimateKeypoints = np.copy(keypoints)
    #         keypointsAsIndices = np.copy(keypoints)
    #         keypointsAsIndices = np.round(keypointsAsIndices)
    #         #keypointsAsIndices[:, 1] = np.clip(keypointsAsIndices[:,1],0, 488 - 1)
    #         #keypointsAsIndices[:, 0] = np.clip(keypointsAsIndices[:,0],0, 648 - 1)
    #         keypointsAsIndices = keypointsAsIndices.astype(np.uint8)
    #         xVals = xFlow[keypointsAsIndices[:,1], keypointsAsIndices[:,0]]
    #         yVals = yFlow[keypointsAsIndices[:,1], keypointsAsIndices[:,0]]
    #         #yVals *= -1
    #         #xVals *= -1
    #         estimateKeypoints[:,0] += xVals
    #         estimateKeypoints[:,1] += yVals
            estimateKeypointsList.append(estimateKeypoints)

        if len(trackableKeypoints) == 0:
            # This is the instance in which all of the nextFish, if there are any, are considered new
            for (box,fish) in zip(boxList, keypointsList):
                newKeypointsListDummy.append([lastFishSeen, box, fish])
                lastFishSeen += 1

        elif len(keypointsList) != 0:
            # Not else in case keypointsList is empty
            # This is the case in which both the past fish and the next fish are not empty
    #         print('just before combinations')
            smallestLossInstances, pastFishOptimalCom, nextFishOptimalCom = getLossCombinationsFast(estimateKeypointsList, keypointsList)

            for (lossInstance, pastFishIdx, nextFishIdx) in zip(smallestLossInstances, pastFishOptimalCom, nextFishOptimalCom):

                idOfPastFish = pastKeypointsList[pastFishIdx][0]

                dummyListIdx = trackableKeypointsIndices[pastFishIdx]
                fish = keypointsList[nextFishIdx]
                box = boxList[nextFishIdx]

                if lossInstance < proximityThreshold:
                    # This fish corresponds to one of the fish in the past frame

                    #newKeypointsListDummy[dummyListIdx] = fish
                    newKeypointsListDummy.append([idOfPastFish, box, fish])
                else:
                    # This is a new fish
                    #newKeypointsListDummy.append(fish)
                    newKeypointsListDummy.append([lastFishSeen, box, fish])
                    lastFishSeen += 1
            # Now lets add the fish that were not considered as previous fish as new fish
            for nextFishIdx, fish in enumerate(keypointsList):
                if nextFishIdx not in nextFishOptimalCom:
                    newKeypointsListDummy.append([lastFishSeen, boxList[nextFishIdx] ,fish])
                    lastFishSeen += 1



        trackingList.append(newKeypointsListDummy) # No Longer A Dummy  :P
    
    tqdm._instances.clear()
    return trackingList
#     pastFrame = nextFrame


def getRidOfBogusIds(trackingList):
    # Function for clearing out data
#     framesRequired = 3
    framesRequired = 3


    idsSeen = []
    badIds = []

    # The Starting Case
    pastIdsMaster = [[] for _ in range(framesRequired)]
    for idx in range(framesRequired):
        ids = [fishId for (fishId,_, __) in trackingList[idx]]
        pastIdsMaster[idx] = ids

        if idx > 0:
            # check that none of the previous have already disapeared
            previousIds = pastIdsMaster[-1]
            for fishId in previousIds:
                if fishId not in ids: badIds.append(fishId)
    pastIdsMaster = pastIdsMaster[1:]

    # The Middle Case
    for frameIdx in range(framesRequired - 1, len(trackingList) - 1 ):
        frameData = trackingList[frameIdx]
        ids = [fishId for (fishId, _, __) in frameData]
        frameData = trackingList[frameIdx + 1]
        nextFrameIds = [fishId for (fishId, _, __) in frameData]

        for fishId in ids:
            if fishId not in nextFrameIds:
                # Lets check that it appeared atleast at the amount of framesRequired

                amountSeen = 1 # Already is in the frame of focus
                for pastIdList in pastIdsMaster:
                    if fishId in pastIdList: amountSeen += 1

                if amountSeen < framesRequired: badIds.append(fishId)
        pastIdsMaster = pastIdsMaster[1:]
        pastIdsMaster.append(ids)

    # The End Case, any new Ids simply discard them
    pastIds = [fishId for (fishId, _, __) in trackingList[-2]]
    ids = [fishId for (fishId, _, __) in trackingList[-1]]

    for fishId in ids:
        if fishId not in pastIds: badIds.append(fishId)

    newTrackingList = []
    for frameData in trackingList:
        currated = [[fishId, box, fish] for (fishId, box, fish) in frameData if fishId not in badIds ]
        newTrackingList.append(currated)
    
    return newTrackingList

def connectSwimBouts(newTrackingList):
    # Lets connect the swim bouts
    maxGap = 25
    #maxGap = 75
    #maxGap = 150
    closenessThreshold = 20
    connectionsData = []

    amountOfFrames = len(newTrackingList)

    #for frameIdx in range(0, len(trackingList) - 1 ):
    for frameIdx in range(1, len(newTrackingList) - 2 ):
        ids = [fishId for (fishId, box, fish) in newTrackingList[frameIdx]]

        newIds = [fishId for (fishId, box, fish) in newTrackingList[frameIdx + 1]]
        for (fishId, box, fish) in newTrackingList[frameIdx]:

            if fishId not in newIds:
                # Let see if we can find it in one of the new frames
                pastFish = None
                for (pastFishId,tempBox, tempFish) in newTrackingList[frameIdx - 1]:
                    if pastFishId == fishId: pastFish = tempFish
                if pastFish is None: continue
                com = fish[2,:]
                pastCom = pastFish[2,:]
                v = com - pastCom

                for offsetIdx in range(2, maxGap + 2):
                    if (frameIdx + offsetIdx) >= amountOfFrames: break 
                    estimateFish = (v * offsetIdx) + fish
                    offsetData = [ [offsetFishId, offsetBox, offsetFish] for (offsetFishId,offsetBox, offsetFish) in newTrackingList[frameIdx + offsetIdx] if offsetFishId not in ids]
                    if len(offsetData) == 0: continue
                    # Lets get the optimal combination
                    offsetFish = [offsetDataEl[2] for offsetDataEl in offsetData]
#                     smallestLossInstances, pastFishOptimalCom, nextFishOptimalCom = getLossOfCombinations([fish], offsetFish)
                    # NOTE: Lets see how the modification makes it behave
                    smallestLossInstances, pastFishOptimalCom, nextFishOptimalCom = getLossCombinationsFast([fish], offsetFish)


                    if smallestLossInstances[0] < closenessThreshold:
                        # We found its connection
                        connectionsData.append([fishId, offsetData[nextFishOptimalCom[0]][0] ]  )
                        break

    # Lets parse the connectionsData, so that it is easier to rewrite the ids
    if len(connectionsData) == 0: connectionsDataBetter[0] = connectionsData
    connectionsDataBetter = []
    for (con1, con2) in connectionsData:
        wasInList = False
        for idx, el in enumerate(connectionsDataBetter):
            if con1 in el:
                wasInList = True
                connectionsDataBetter[idx] += [con2]
                break

        if not wasInList: connectionsDataBetter.append([con1, con2])

    newTrackingList2 = []
    for frameData in newTrackingList:
        temp = []
        for (fishId, box, fish) in frameData:

            for el in connectionsDataBetter:
                if fishId in el: 
                    fishId = min(el)
                    break

            temp.append([fishId, box, fish])
        newTrackingList2.append(temp)

    # Writting it with the lowestIdx possible
    smallestId = 0
    idDic = {}

    newTrackingList3 = []
    for frameData in newTrackingList2:
        temp = []
        for (fishId,box,fish) in frameData:
            if fishId not in idDic:
                idDic[fishId] = smallestId
                smallestId += 1
            temp.append([idDic[fishId],box, fish]  )

        newTrackingList3.append(temp)
    
    return newTrackingList3


def fillOutData(newTrackingList3):
    masterD = dict()

    for frameIdx, frameData in enumerate(newTrackingList3):
        frameIds = [fishId for (fishId,box , fish) in frameData ]

        for idSeen in masterD:
            d = masterD[idSeen]

            if idSeen in frameIds:
                if not d['past']:


                    gapList = d['gaps']
                    lastSeen = d['lastSeen']
                    gapList.append( [lastSeen+1, frameIdx - 1])
                    d['gaps'] = gapList
                d['past'] = 1
                d['end'] = frameIdx
            else:
                if d['past']:
                    d['past'] = 0
                    d['lastSeen'] = frameIdx - 1

            masterD[idSeen] = d

        for fishId in frameIds:
            if fishId not in masterD:
                newD = {'past':1,'start': frameIdx, 'end': frameIdx, 'gaps':[], 'lastSeen': frameIdx}
                masterD[fishId] = newD

    return masterD


def fillOutListWithData(newTrackingList3, d):
    newTrackingList4 = newTrackingList3.copy()
    for key in d:
        gaps = d[key]['gaps']

        for gap in gaps:
            s, e = gap
            length = (e - s) + 1
            divisions = length + 1

            [startFish, startBox] = [[fish, box] for (fishId, box, fish) in newTrackingList3[s - 1] if fishId == key ][0]
            [endFish, endBox] = [[fish, box] for (fishId, box, fish) in newTrackingList3[e + 1] if fishId == key ][0]
            distBox = endBox - startBox
            distBox = distBox / divisions
            
            distFish = endFish - startFish
            distFish = distFish / divisions
            
            for offsetIdx in range(length):
                newFish = startFish + ((offsetIdx + 1) * distFish)
                newBox = startBox + ((offsetIdx + 1) * distBox )
                frameData = newTrackingList4[s + offsetIdx]

                frameData.append([key, newBox, newFish])
                newTrackingList4[s + offsetIdx] = frameData
                
    return newTrackingList4



def turnIdsToSmallest(newTrackingList2):
    smallestId = 0
    idDic = {}

    newTrackingList3 = []
    for frameData in newTrackingList2:
        temp = []
        for (fishId, box,fish) in frameData:
            if fishId not in idDic:
                idDic[fishId] = smallestId
                smallestId += 1
            temp.append([idDic[fishId], box, fish]  )

        newTrackingList3.append(temp)
        
    return newTrackingList3

def connectOverlapBouts(newTrackingList4, d2):
    print('Starting')

    proximityThreshold = 20
    maxHalucinationLen = 2
    linkList = [] # Linking Data
    unlinkablelist = []
    startingLinks = []
    comsMasterList = []
    # Algorith for linking the bouts
    bar = tqdm(total = len(newTrackingList4))
    for frameIdx, frameData in enumerate(newTrackingList4):
        bar.update(1)
        startingLinks = [el[0] for el in linkList] # This are the connections that were already linked

        proximityList = []
        for (fishId, _, fish) in frameData:
            # Lets check its not a halucination
            s, e = d2[fishId]['start'], d2[fishId]['end']
            length = (e - s) + 1
            if length < maxHalucinationLen: continue

            foundProximity = False
            proximityIdx = None

            for comIdx, proxData in enumerate(proximityList):
                overlapP = proxData['overlapP']
                dist = np.sum(((fish[2,:] - overlapP) **2)) ** .5

                if dist < proximityThreshold:
                    foundProximity = True
                    proximityIdx = comIdx
                    break


            if foundProximity:


                proxData = proximityList[proximityIdx]
                fishes = proxData['fishes']
                fishes.append([fishId, fish])
                proxData['fishes'] = fishes

                # Update OverlapP
                coms = [fish[2,:] for (_ ,fish) in fishes]
                coms = np.array(coms)

                overlapP = np.mean(coms, axis = 0)

                proxData['overlapP'] = overlapP

                proximityList[proximityIdx] = proxData

            else:
                proxData = {'fishes': [[fishId, fish]], 'overlapP':fish[2,:]}
                proximityList.append(proxData)


        # Lets check if there are any overlapping fishes
        for proxData in proximityList:
            if len(proxData['fishes']) > 1:
                # It is an overlap instance
                overlapFishInfo = proxData['fishes'].copy()
                overlapFishInfoIds = [fishId for (fishId, fish) in overlapFishInfo]
    #             print('the overlap fish info: ',overlapFishInfoIds)
                # Checking to make sure it is not a case we already linked
                shouldBreak = False
                for fishId in overlapFishInfoIds: 
                    if fishId in (startingLinks + unlinkablelist): 
                        shouldBreak = True
                        break
                if shouldBreak: break

                overlapP = proxData['overlapP']
    #             print('og overlapP:', overlapP)
                comsList = [overlapP]
                amountList = [len(overlapFishInfo)]

                needLinkList = []

                for offsetIdx, otherFrameData in enumerate(newTrackingList4[frameIdx + 1:]):
                    offsetFrameIdx = frameIdx + 1 + offsetIdx

                    newOverlapInfo = [[fishId, fish] for (fishId, _, fish) in otherFrameData if fishId in overlapFishInfoIds]
                    newOverlapInfoIds = [fishId for (fishId, fish) in newOverlapInfo]

                    nonOverlapInfo = [[fishId, fish] for (fishId, _, fish) in otherFrameData if fishId not in overlapFishInfoIds]
                    nonOverlapInfoIds = [fishId for (fishId, fish) in nonOverlapInfo]


                    overlapFish2Add = []
                    fish2Remove = []

                    # Lets check if some of the new fish entered the overlapArea
                    for (fishId, fish) in nonOverlapInfo:
    #                     if fishId == 5: 
    #                         print('the distance of 5:', np.sum((fish[2,:] - overlapP)** 2) ** .5)
    #                         print('the frameIdx:', offsetFrameIdx)
    #                         print('overlap point:', overlapP)

                        if np.sum((fish[2,:] - overlapP)** 2) ** .5 < proximityThreshold:
                            overlapFish2Add.append([fishId, fish])
    #                 temp = [fishId for (fishId, fish) in overlapFish2Add]
    #                 if 5 in temp and 6 in temp: print('your list:', overlapFish2Add)

                    if len(needLinkList) > 0 and len(overlapFish2Add) > 0:
                        # Let create a list of overlapFish2Add which has fish that where previously not clearly define before
                        overlapFish2AddDummy = []
                        for (fishId, fish) in overlapFish2Add:
                            amountDefined = (offsetFrameIdx - d2[fishId]['start']) + 1
                            if amountDefined < 50: overlapFish2AddDummy.append([fishId, fish])
                        if len(overlapFish2AddDummy) > 0:
                            needLinkListFish = [fish for (fishId, fish) in needLinkList]
                            overlapFish2AddFish = [fish for (fishId, fish) in overlapFish2AddDummy]

                            # Finding the optimal combination, first output ignored since the are/where in the overlap region
                            _, needLinkListIndices, overlapFish2AddIndices = getLossCombinationsFast(needLinkListFish, overlapFish2AddFish)

                            optimalFishIds = [needLinkList[linkListIdx][0] for linkListIdx in needLinkListIndices]
                            optimalOverlapIds = [overlapFish2AddDummy[overlapListIdx][0] for overlapListIdx in overlapFish2AddIndices]

                            for (id1, id2) in zip(optimalFishIds, optimalOverlapIds):  linkList.append([id1, id2])


                            needLinkList = [[fishId, fish] for (fishId, fish) in needLinkList if fishId not in  optimalFishIds]


                    # Lets check that the fish in the overlap region have not disapeared or have left the overlap region
                    for (fishId, fish) in overlapFishInfo:
                        if fishId not in newOverlapInfoIds:
                            # It has disapeared
                            needLinkList.append([fishId, fish])
                            # fish2Remove.append(fishId) # Technically not needed
                        else:
                            # Its there, lets just check that it has not left
                            if np.sum((fish[2,:] - overlapP) ** 2) ** .5 > proximityThreshold:
                                fish2Remove.append(fishId)


                    # Lets update the lists
                    overlapFishInfo = [[fishId, fish] for (fishId, fish) in newOverlapInfo if fishId not in fish2Remove]
                    for (fishId, fish) in overlapFish2Add: overlapFishInfo.append([fishId, fish])
                    overlapFishInfoIds = [fishId for (fishId, fish) in overlapFishInfo]

                    if len(overlapFishInfo) <= 1 and len(needLinkList) == 0:
                        # Our overlapping region has only one fish and there are no fish need to be linked
                        if len(overlapFishInfo) > 0:  unlinkablelist += [overlapFishInfo[0][0]]
                        unlinkablelist += fish2Remove
                        break
                    else:

                        # Otherwise lets update our com and continue
                        if len(overlapFishInfo) >= 1:
                            coms = [fish[2,:] for (fishId, fish) in  overlapFishInfo]
                            coms = np.array(coms)

                            overlapP = np.mean(coms, axis = 0)

                        comsList.append(overlapP)
                        amountList.append(len(overlapFishInfo) + len(needLinkList))
    #             for (fishId, fish) in needLinkList: unlinkablelist.append(fishId)
                    if offsetFrameIdx + 1 >= len(newTrackingList4): 
                        unlinkablelist += needLinkList
                        unlinkablelist += overlapFishInfoIds
                if offsetFrameIdx + 1 < len(newTrackingList4):
                    comsMasterList.append([frameIdx, comsList, amountList])
    #             print(offsetIdx)
    #             print('linkList: ',linkList)

    tqdm._instances.clear()
        
    # Start of the next algorithm
    # Let create a master link list
    connectionsDataBetter = []

    for link in linkList:
        l1, l2 = link

        wasInList = False
        idx = None
        for (elIdx, el) in enumerate(connectionsDataBetter):
            if l1 in el:
                connectionsDataBetter[elIdx] += [l2]
                wasInList = True
                idx = elIdx
                break
            elif l2 in el:
                connectionsDataBetter[elIdx] += [l1]
                wasInList = True
                idx = elIdx
                break
        if wasInList:
            if l1 not in connectionsDataBetter[idx]: connectionsDataBetter[idx] += [l1]
            if l2 not in connectionsDataBetter[idx]: connectionsDataBetter[idx] += [l2]
        else:
            connectionsDataBetter.append( [l1,l2])
        
    # Now lets rewrite of trackingList 
    newTrackingList5 = []
    for frameData in newTrackingList4:
        temp = []
        for (fishId, box, fish) in frameData:

            for el in connectionsDataBetter:
                if fishId in el: 
                    fishId = min(el)
                    break

            temp.append([fishId, box, fish])
        newTrackingList5.append(temp)
        
    return newTrackingList5, comsMasterList

def connectOverlapBouts3(newTrackingList4, d2):
    print('Starting')
    timeThreshold = 500 # 1 second for us at 500 fps

    proximityThreshold = 20
    maxHalucinationLen = 2
    linkList = [] # Linking Data
    unlinkablelist = []
    startingLinks = []
    comsMasterList = []
    startingData = []
    # Algorith for linking the bouts
    bar = tqdm(total = len(newTrackingList4))
    for frameIdx, frameData in enumerate(newTrackingList4):
        bar.update(1)
        startingLinks = [el[0] for el in linkList] # This are the connections that were already linked

        proximityList = []
        for (fishId, _, fish) in frameData:
            # Lets check its not a halucination
            s, e = d2[fishId]['start'], d2[fishId]['end']
            length = (e - s) + 1
            if length < maxHalucinationLen: continue

            foundProximity = False
            proximityIdx = None

            for comIdx, proxData in enumerate(proximityList):
                overlapP = proxData['overlapP']
                dist = np.sum(((fish[2,:] - overlapP) **2)) ** .5

                if dist < proximityThreshold:
                    foundProximity = True
                    proximityIdx = comIdx
                    break


            if foundProximity:


                proxData = proximityList[proximityIdx]
                fishes = proxData['fishes']
                fishes.append([fishId, fish])
                proxData['fishes'] = fishes

                # Update OverlapP
                coms = [fish[2,:] for (_ ,fish) in fishes]
                coms = np.array(coms)

                overlapP = np.mean(coms, axis = 0)

                proxData['overlapP'] = overlapP

                proximityList[proximityIdx] = proxData

            else:
                proxData = {'fishes': [[fishId, fish]], 'overlapP':fish[2,:]}
                proximityList.append(proxData)


        # Lets check if there are any overlapping fishes
        for proxData in proximityList:
            if len(proxData['fishes']) > 1:
                # It is an overlap instance
                overlapFishInfo = proxData['fishes'].copy()
                overlapFishInfoIds = [fishId for (fishId, fish) in overlapFishInfo]
    #             print('the overlap fish info: ',overlapFishInfoIds)
                # Checking to make sure it is not a case we already linked

                shouldBreak = False
                for (fishInfoIds, startFrameIdx, endIdx) in startingData:
                    if len(fishInfoIds) == len(overlapFishInfoIds) and ( (frameIdx >= startFrameIdx) and (frameIdx <= endIdx)) :
                        # It could be that we have come across the same instance
                        areTheyAllThere = True
                        for fishId in fishInfoIds:
                            if fishId not in overlapFishInfoIds: False # A difference is enough to prove that they are different
                        if areTheyAllThere == True: 
                            shouldBreak = True
                            break
                if shouldBreak: break
                
                
                startingData.append([overlapFishInfoIds, frameIdx]) # We will add the ending frameIdx at the end
                
                overlapP = proxData['overlapP']
    #             print('og overlapP:', overlapP)
                comsList = [overlapP]
                amountList = [len(overlapFishInfo)]

                needLinkList = []

                for offsetIdx, otherFrameData in enumerate(newTrackingList4[frameIdx + 1:]):
                    offsetFrameIdx = frameIdx + 1 + offsetIdx

                    newOverlapInfo = [[fishId, fish] for (fishId, _, fish) in otherFrameData if fishId in overlapFishInfoIds]
                    newOverlapInfoIds = [fishId for (fishId, fish) in newOverlapInfo]

                    nonOverlapInfo = [[fishId, fish] for (fishId, _, fish) in otherFrameData if fishId not in overlapFishInfoIds]
                    nonOverlapInfoIds = [fishId for (fishId, fish) in nonOverlapInfo]


                    overlapFish2Add = []
                    fish2Remove = []

                    # Lets check if some of the new fish entered the overlapArea
                    for (fishId, fish) in nonOverlapInfo:
    #                     if fishId == 5: 
    #                         print('the distance of 5:', np.sum((fish[2,:] - overlapP)** 2) ** .5)
    #                         print('the frameIdx:', offsetFrameIdx)
    #                         print('overlap point:', overlapP)

                        if np.sum((fish[2,:] - overlapP)** 2) ** .5 < proximityThreshold:
                            overlapFish2Add.append([fishId, fish])
    #                 temp = [fishId for (fishId, fish) in overlapFish2Add]
    #                 if 5 in temp and 6 in temp: print('your list:', overlapFish2Add)

                    if len(needLinkList) > 0 and len(overlapFish2Add) > 0:
                        # Let create a list of overlapFish2Add which has fish that where previously not clearly define before
                        overlapFish2AddDummy = []
                        for (fishId, fish) in overlapFish2Add:
                            amountDefined = (offsetFrameIdx - d2[fishId]['start']) + 1
                            if amountDefined < 50: overlapFish2AddDummy.append([fishId, fish])
                        if len(overlapFish2AddDummy) > 0:
                            needLinkListFish = [fish for (fishId, fish) in needLinkList]
                            overlapFish2AddFish = [fish for (fishId, fish) in overlapFish2AddDummy]

                            # Finding the optimal combination, first output ignored since the are/where in the overlap region
                            _, needLinkListIndices, overlapFish2AddIndices = getLossCombinationsFast(needLinkListFish, overlapFish2AddFish)

                            optimalFishIds = [needLinkList[linkListIdx][0] for linkListIdx in needLinkListIndices]
                            optimalOverlapIds = [overlapFish2AddDummy[overlapListIdx][0] for overlapListIdx in overlapFish2AddIndices]

                            for (id1, id2) in zip(optimalFishIds, optimalOverlapIds):  linkList.append([id1, id2])


                            needLinkList = [[fishId, fish] for (fishId, fish) in needLinkList if fishId not in  optimalFishIds]


                    # Lets check that the fish in the overlap region have not disapeared or have left the overlap region
                    for (fishId, fish) in overlapFishInfo:
                        if fishId not in newOverlapInfoIds:
                            # It has disapeared
                            needLinkList.append([fishId, fish])
                            # fish2Remove.append(fishId) # Technically not needed
                        else:
                            # Its there, lets just check that it has not left
                            if np.sum((fish[2,:] - overlapP) ** 2) ** .5 > proximityThreshold:
                                fish2Remove.append(fishId)


                    # Lets update the lists
                    overlapFishInfo = [[fishId, fish] for (fishId, fish) in newOverlapInfo if fishId not in fish2Remove]
                    for (fishId, fish) in overlapFish2Add: overlapFishInfo.append([fishId, fish])
                    overlapFishInfoIds = [fishId for (fishId, fish) in overlapFishInfo]

                    if len(overlapFishInfo) <= 1 and len(needLinkList) == 0:
                        # Our overlapping region has only one fish and there are no fish need to be linked
                        if len(overlapFishInfo) > 0:  unlinkablelist += [overlapFishInfo[0][0]]
                        unlinkablelist += fish2Remove
                        break
                    else:

                        # Otherwise lets update our com and continue
                        if len(overlapFishInfo) >= 1:
                            coms = [fish[2,:] for (fishId, fish) in  overlapFishInfo]
                            coms = np.array(coms)

                            overlapP = np.mean(coms, axis = 0)

                        comsList.append(overlapP)
                        amountList.append(len(overlapFishInfo) + len(needLinkList))
                    
                    if offsetFrameIdx + 1 >= len(newTrackingList4):
                        unlinkablelist += needLinkList
                        unlinkablelist += overlapFishInfo
                        
    #             for (fishId, fish) in needLinkList: unlinkablelist.append(fishId)
                startingData[-1] = startingData[-1] + [offsetFrameIdx] # Adding the endIdx
                if offsetFrameIdx + 1 < len(newTrackingList4):
                    comsMasterList.append([frameIdx, comsList, amountList])
    #             print(offsetIdx)
    #             print('linkList: ',linkList)

        tqdm._instances.clear()
        
    # Start of the next algorithm
    # Let create a master link list
    connectionsDataBetter = []

    for link in linkList:
        l1, l2 = link

        wasInList = False
        idx = None
        for (elIdx, el) in enumerate(connectionsDataBetter):
            if l1 in el:
                connectionsDataBetter[elIdx] += [l2]
                wasInList = True
                idx = elIdx
                break
            elif l2 in el:
                connectionsDataBetter[elIdx] += [l1]
                wasInList = True
                idx = elIdx
                break
        if wasInList:
            if l1 not in connectionsDataBetter[idx]: connectionsDataBetter[idx] += [l1]
            if l2 not in connectionsDataBetter[idx]: connectionsDataBetter[idx] += [l2]
        else:
            connectionsDataBetter.append( [l1,l2])
        
    # Now lets rewrite of trackingList 
    newTrackingList5 = []
    for frameData in newTrackingList4:
        temp = []
        for (fishId, box, fish) in frameData:

            for el in connectionsDataBetter:
                if fishId in el: 
                    fishId = min(el)
                    break

            temp.append([fishId, box, fish])
        newTrackingList5.append(temp)
        
    return newTrackingList5, comsMasterList


def eraseIncorrectPredictions(comsMasterList, newTrackingList5, trackingList):
    proximityThreshold = 20

    masterAmount = []
    for comData in tqdm(comsMasterList):
        startIdx, coms, amounts = comData
        amountOfComs = len(coms)
        amountList = []
        for offsetIdx in range(amountOfComs):
            frameData = newTrackingList5[startIdx + offsetIdx]
            com = coms[offsetIdx]
            amount = 0
            for (fishId, box, fish) in frameData:
                dist = np.sum((fish[2,:] - com)**2)**.5
                if dist < proximityThreshold: amount += 1
            amountList.append(amount)
        amountList = np.array( amountList )
        m = stats.mode(amountList)
        masterAmount.append(m)
    
    
    delIndices = [[] for _ in range(len(newTrackingList5))]
    for (modeAmount, comData) in tqdm(zip(masterAmount, comsMasterList)):
        startIdx, coms, amounts = comData
        amountOfComs = len(coms)
        amountList = []
        for offsetIdx in range(amountOfComs):
            frameData = trackingList[startIdx + offsetIdx]
            com = coms[offsetIdx]
            amount = amounts[offsetIdx]
            badIdx = []

            for fishIdx, (fishId, box, fish) in enumerate(frameData):
                dist = np.sum((fish[2,:] - com)**2)**.5
                if dist < proximityThreshold: badIdx.append(fishIdx )

            if len(badIdx) < amount: delIndices[startIdx + offsetIdx] += badIdx
    
    
    # Actually deleting them
    trackingListCopy = []
    for (badIndices, frameData) in zip(delIndices, trackingList):
        temp = []
        for fishIdx, (fishId, box, fish) in enumerate(frameData):
            if fishIdx not in badIndices: temp.append([fishId, box, fish])
        trackingListCopy.append(temp)
    
    # Converting it into a dataList
    dataList = []
    for frameData in trackingListCopy:
        boxList = []
        keypointsList = []
        _ = 55
        for (fishId, box, fish) in frameData:
            boxList.append(box)
            keypointsList.append(fish)
        dataList.append([_, boxList, keypointsList])
        
    return dataList


def yoloFishEstimation(dataList):
    trackingListOG = closenessTracking(dataList)
    
    trackingList = getRidOfBogusIds(trackingListOG)
    trackingList = connectSwimBouts(trackingList)
    data = fillOutData(trackingList)
    trackingList = fillOutListWithData(trackingList, data)
    print('len after fillout: ', len(trackingList))
    data = fillOutData(trackingList)
    trackingList, comsMasterList = connectOverlapBouts(trackingList, data)
    print('len after connect: ', len(trackingList))
    
    # The bout ids were connected, lets just fill in the gaps where there is no fish
    trackingList = turnIdsToSmallest(trackingList)
    data = fillOutData(trackingList)
    trackingList = fillOutListWithData(trackingList, data)

    
    return trackingList


def yoloFishEstimationWithErasing(dataList):
    trackingListOG = closenessTracking(dataList)
    tqdm._instances.clear()
    print('doing the initial analysis')
    trackingList = getRidOfBogusIds(trackingListOG)
    trackingList = connectSwimBouts(trackingList)
    data = fillOutData(trackingList)
    trackingList = fillOutListWithData(trackingList, data)
    tqdm._instances.clear()
    print('getting gap data and connecting the bouts')
    data = fillOutData(trackingList)
    trackingList, comsMasterList = connectOverlapBouts3(trackingList, data)
    tqdm._instances.clear()
    print('filling out the data in which the bouts were not defined')

    # The bout ids were connected, lets just fill in the gaps where the fish predictions disappear
    trackingList = turnIdsToSmallest(trackingList)
    data = fillOutData(trackingList)
    trackingList = fillOutListWithData(trackingList, data)
    
    tqdm._instances.clear()
    print('erasing the instances in which there were bad indices')
    # Lets erase the incorrect iou predictions
    dataList = eraseIncorrectPredictions(comsMasterList, trackingList, trackingListOG)

    # Now that we have created a dataList without the intersecting fish, we can repeate our connections again
    # Note: It is possible to improve this algorithm to only recconect the fish for which the data was bad, since
    # The algorithm is going to produce the same output for the ones that were not bad
    trackingListOG = closenessTracking(dataList)

    trackingList = getRidOfBogusIds(trackingListOG)
    trackingList = connectSwimBouts(trackingList)
    data = fillOutData(trackingList)
    trackingList = fillOutListWithData(trackingList, data)
    tqdm._instances.clear()
    print('getting gap data and connecting the bouts')
    data = fillOutData(trackingList)
    trackingList, comsMasterList = connectOverlapBouts3(trackingList, data)
    tqdm._instances.clear()
    print('filling out the data in which the bouts were not defined')

    # The bout ids were connected, lets just fill in the gaps where there is no fish
    trackingList = turnIdsToSmallest(trackingList)
    data = fillOutData(trackingList)
    trackingList = fillOutListWithData(trackingList, data)



    return trackingList

def bgsubList(folderName, filenames):

    frames = []
    for filename in filenames:
        frame = cv.imread(folderName + '/' + filename)
        grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(grayImage)

    frameCount = len(frames)
    nSampFrame = min(np.fix(frameCount / 2), 100)

    # Creating a numpy array of the the sample frames
    firstSampFrame = True
    secondSampFrame = True
    for frameNum in np.fix(np.linspace(1, frameCount, int(nSampFrame))):
        if firstSampFrame:
            firstSampFrame = False
            sampFrames = frames[0]
            continue
        if secondSampFrame:
            secondSampFrame = False
            sampFrames = np.stack((sampFrames, frames[int((frameNum - 1))]), axis=0)
            continue
        sampFrames = np.vstack((sampFrames, np.array([frames[int((frameNum - 1))]])))

    sampFrames.sort(0)

    videobg = sampFrames[int(np.fix(nSampFrame * .9))]

    outputVid = []
    for frame in frames:
        # Subtract foreground from background image by allowing values beyond the 0 to 255 range of uint8
        difference_img = np.int16(videobg) - np.int16(frame)

        # Clip values in [0,255] range
        difference_img = np.clip(difference_img, 0, 255)

        # Convert difference image to uint8 for saving to video
        difference_img = np.uint8(difference_img)
        outputVid.append(difference_img)

    return outputVid

def isBoxFarFromEdge(box):
    edgeThreshold = 5
    imageSizeX, imageSizeY = 640, 640
    xDistance = np.min(imageSizeX - box[[0,2]])
    xDistance2 = np.min(box[[0,2]])
    xDistanceMin = min(xDistance, xDistance2)
    yDistance = np.min(imageSizeY - box[[1,3]])
    yDistance2 = np.min(box[[1,3]])
    yDistanceMin = min(yDistance, yDistance2)
    minDist = min(yDistanceMin, xDistanceMin)
    
    return minDist > edgeThreshold


# First lets get the bgsubVideo
folder = '../yolo_and_resnet_orthographic/020116_012/'
folder = '2020116_012/'
#folder = 'videos/020116_028_no_overlap/'
files = os.listdir(folder)
files = [fileName for fileName in files if fileName.endswith('.bmp')]
files.sort()

video = bgsubList(folder[:-1], files)

#videoPath = '../yolo_and_resnet/20180103/1200/010318_1200_s2.avi'
#cap = cv.VideoCapture(videoPath)
#video = bgsub(cap)
#cap.release()
#del cap

# Let load the YOLO model
yoloWeights = 'inputs/weights/orthographic_yolo/best.pt'
model = YOLO(yoloWeights)

shouldGetData = True

if shouldGetData:
    # Now lets get the predictions from YOLO in a format for our tracking algorithm
    batchSize = 500
    #batchSize = 1
    fc = 0
    dataList = []
    frames = []
    amountOfFrames = len(video)
    pbar = tqdm(total=amountOfFrames)

    while True:
        frame = video[fc]
        frame = frame.astype(float)
        frame *= 255 / np.max(frame)
        frame = frame.astype(np.uint8)
        frame = np.stack((frame, frame, frame), axis = 2)

        frames.append( frame )

        fc += 1

        if len(frames) >= batchSize or fc == amountOfFrames:
            results = model.predict(frames, verbose = False, stream = True)
            for resultIdx, result in enumerate(results):
                frame = frames[resultIdx]
                confidence_mask = result.boxes.conf.cpu().numpy() > .6  # Making sure the predictions are accurate
                boxes =  result.boxes.xyxy.cpu().numpy()[confidence_mask]
                classes = result.boxes.cls.cpu().numpy()[confidence_mask]
                keypoints = result.keypoints.xy.cpu().numpy()[confidence_mask]
            
                mask = [isBoxFarFromEdge(box) for box in boxes]  # Making sure the boxes are not really close to the edge
            
                boxes = boxes[mask]
                classes = classes[mask]
                keypoints = keypoints[mask]
                
                # Making sure the box be captures is not a fish that is barely visible
                mask = []
                for box in boxes:
                    box = np.clip(box, 0, 639)
                    box = box.astype(int)
                    sx, sy, bx, by = box
                    cutout = frame[sy:by + 1, sx:bx + 1]
                    maxBr = np.max(cutout)
                    if maxBr < 100:
                        mask.append(False)
                    else:
                        mask.append(True)
                
                boxes = boxes[mask]
                classes = classes[mask]
                keypoints = keypoints[mask] 

                data = [classes, boxes, keypoints]
                dataList.append(data)
            pbar.update(len(frames))
            if fc >= amountOfFrames: break
            frames = []

if not shouldGetData:
    print('DONE')
    with open('OrthographicDataList.pkl', 'rb') as b:
        dataList = pickle.load(b)

savePath = 'OrthographicDataList.pkl'
with open(savePath, 'wb') as b:
    pickle.dump(dataList, b)

print('beggining the tracking')
startTime = time.time()
trackingList = yoloFishEstimationWithErasing(dataList)
endTime = time.time()
print('tracking duration: ', endTime - startTime)

## To save the tracking information
#savePath = 'trackingList.pkl'
#with open(savePath, 'wb') as b:
#    pickle.dump(trackingList, b)

print('tracking estimation done, drawing the video')

imageSizeX, imageSizeY = 640, 640
fps = 500
#fps = cap.get( cv.CAP_PROP_FPS )
fourcc = cv.VideoWriter_fourcc('M','J','P','G')
#fourcc = cv.VideoWriter_fourcc(*'DIVX')
#outputPath = 'outputs/superimposed/' + videoName + '.avi'
outputPath = 'sideVideo.avi'
out = cv.VideoWriter(outputPath, fourcc  , int( fps  ) ,(int(imageSizeX) , int( imageSizeY )))
red = [0,0,255]
green = [0,255,0]

# for frameIdx, data in tqdm(enumerate(newTrackingList4)):
# for frameIdx, data in tqdm(enumerate(trackingList)):
for frameIdx, data in tqdm(enumerate(trackingList), total = len(trackingList)):
# for frameIdx, data in tqdm(enumerate(newTrackingList42)):
# for frameIdx, data in tqdm(enumerate(trackingList)):
# for frameIdx, data in tqdm(enumerate(newTrackingList2)):
# for frameIdx, data in tqdm(enumerate(trackingListCopy)):
# for frameIdx, data in tqdm(enumerate(newTrackingList3)):

    frame = video[frameIdx]
    frame = np.stack([frame, frame, frame], axis = 2)
    
    
    for (fishId, box, fish) in data:
        com = fish[2,:]
        com = com.astype(int)
        font = cv.FONT_HERSHEY_SIMPLEX 
  
        # Use putText() method for 
        # inserting text on video 
        
        # Drawing the keypoints
        for keypointIdx in range(len(fish)):
            keypoint = fish[keypointIdx, :]
            color = green if keypointIdx < 10 else red
            keypoint = keypoint.astype(int)
            frame = cv.circle(frame, keypoint, 4, color, -1)
        
        frame = cv.putText(frame, str(fishId), (com[0], com[1]), font, 1, (0, 255, 255), 2, cv.LINE_4) 
        sx, sy, bx, by = box.astype(int)
        frame = cv.rectangle(frame, (sx, sy), (bx, by), color=red, thickness=2)

        
    out.write(frame)   
    
out.release()




































