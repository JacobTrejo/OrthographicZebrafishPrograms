import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
#from programs.bgsub import bgsub
import pickle
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy import stats
from programs.bgsub import bgsub
from programs.getBackGround import multipleBgsubList
from ultralytics import YOLO


def deleteDuplicates(dataList):
    dataList = dataList.copy()
    distanceThreshold = 3
    newDataList = []
    for data in dataList:
        keypointsList = data[2]
        # keypointsList = keypointsList.copy() # I guess to make sure we dont modify dataList
        boxList = data[1]

        classList = data[0]  # Useless, but lets keep it for now

        sortedKeypointsList = []

        for idx, (cls, box, keypoints) in enumerate(zip(classList, boxList, keypointsList)):
            foundNear = False
            for idx2, el in enumerate(sortedKeypointsList):
                cls2 = el[0]
                box2 = el[1]
                keypointsList = el[2]
                keypoints2 = keypointsList[0]
                avgDist = np.sum(np.sum((keypoints[:10] - keypoints2[:10]) ** 2, axis=1) ** .5) / 10
                if avgDist < distanceThreshold:
                    keypointsList.append(keypoints)
                    cls2.append(cls)
                    box2.append(box)
                    sortedKeypointsList[idx2] = [cls2, box2, keypointsList]
                    foundNear = True
                    break
            if not foundNear:
                sortedKeypointsList.append([[cls], [box], [keypoints]])

        classList2 = []
        boxList2 = []
        keypointsList2 = []

        for el in sortedKeypointsList:
            cls = el[0][0]
            box = el[1][0]
            keypoints = el[2][0]

            classList2.append(cls)
            boxList2.append(box)
            keypointsList2.append(keypoints)
        data2 = [classList2, boxList2, keypointsList2]
        newDataList.append(data2)
    return newDataList

def isBoxFarFromEdge(box):
    edgeThreshold = 10 # Twice of which was already removed from the data, aka edgeThreshold of 5
    imageSizeX, imageSizeY = 640, 640
    xDistance = np.min(imageSizeX - box[[0,2]])
    xDistance2 = np.min(box[[0,2]])
    xDistanceMin = min(xDistance, xDistance2)
    yDistance = np.min(imageSizeY - box[[1,3]])
    yDistance2 = np.min(box[[1,3]])
    yDistanceMin = min(yDistance, yDistance2)
    minDist = min(yDistanceMin, xDistanceMin)

    return minDist > edgeThreshold


def createCostMatrix(fishList1, fishList2):
    fishList1, fishList2 = fishList1.copy(), fishList2.copy()  # This is to stop modifying the inputs from the function
    indices = (0, 1) if len(fishList1) <= len(fishList2) else (1, 0)  # This is to put them back into order
    smallestList, biggestList = (fishList1, fishList2) if len(fishList1) <= len(fishList2) else (fishList2, fishList1)
    amountOfFish2Add = len(biggestList) - len(smallestList)

    for _ in range(amountOfFish2Add):
        fish = np.ones((12, 2))
        fish[...] = np.inf
        smallestList.append(fish)

    smallestList, biggestList = np.array(smallestList), np.array(biggestList)

    # temp = [np.array(smallestList), np.array(biggestList)]
    ## Lets order them back into fishList1 and fishList2
    # fishList1, fishList2 = temp[indices[0]], temp[indices[1]]

    # Lets turn them into a form that allows broadcasting
    # fishList2 will be the colomns
    biggestList = biggestList[None, ...]
    smallestList = smallestList[:, None, ...]

    # fishList2 = fishList2[None, ...]
    # fishList1 = fishList1[:,None,...]
    cost = np.sum(np.sum((biggestList - smallestList) ** 2, axis=-1) ** .5, axis=-1) / 12
    cost[cost == np.inf] = 999
    # print(cost.shape)
    # print(cost)
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
    proximityThreshold = 15

    amountOfFrames = len(video)
    # amountOfFrames = 2000
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
    for frameIdx in tqdm(range(1, amountOfFrames)):
        # Getting the data for this frame
        data = dataList[frameIdx]
        keypointsList = data[2]
        boxList = data[1]

        #         data = [ [box, keypoints] for (box, keypoints) in zip(boxList, keypointsList) if np.any(keypoints[:,1] > 144) ]
        data = [[box, keypoints] for (box, keypoints) in zip(boxList, keypointsList)]
        boxList = [box for (box, keypoints) in data]
        keypointsList = [keypoints for (box, keypoints) in data]

        pastKeypointsList = trackingList[-1]
        # newKeypointsListDummy = [None for _ in pastKeypointsList]
        newKeypointsListDummy = []

        # Lets get only the fish that can still be tracked
        trackableKeypointsIndices = [idx for idx, el in enumerate(pastKeypointsList)]
        trackableKeypoints = [pastKeypointsList[idx][2] for idx in trackableKeypointsIndices]

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
            for (box, fish) in zip(boxList, keypointsList):
                newKeypointsListDummy.append([lastFishSeen, box, fish])
                lastFishSeen += 1

        elif len(keypointsList) != 0:
            # Not else in case keypointsList is empty
            # This is the case in which both the past fish and the next fish are not empty
            #         print('just before combinations')
            smallestLossInstances, pastFishOptimalCom, nextFishOptimalCom = getLossCombinationsFast(
                estimateKeypointsList, keypointsList)

            for (lossInstance, pastFishIdx, nextFishIdx) in zip(smallestLossInstances, pastFishOptimalCom,
                                                                nextFishOptimalCom):

                idOfPastFish = pastKeypointsList[pastFishIdx][0]

                dummyListIdx = trackableKeypointsIndices[pastFishIdx]
                fish = keypointsList[nextFishIdx]
                box = boxList[nextFishIdx]

                if lossInstance < proximityThreshold:
                    # This fish corresponds to one of the fish in the past frame

                    # newKeypointsListDummy[dummyListIdx] = fish
                    newKeypointsListDummy.append([idOfPastFish, box, fish])
                else:
                    # This is a new fish
                    # newKeypointsListDummy.append(fish)
                    newKeypointsListDummy.append([lastFishSeen, box, fish])
                    lastFishSeen += 1
            # Now lets add the fish that were not considered as previous fish as new fish
            for nextFishIdx, fish in enumerate(keypointsList):
                if nextFishIdx not in nextFishOptimalCom:
                    newKeypointsListDummy.append([lastFishSeen, boxList[nextFishIdx], fish])
                    lastFishSeen += 1

        trackingList.append(newKeypointsListDummy)  # No Longer A Dummy  :P

    return trackingList


#     pastFrame = nextFrame
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
        ids = [fishId for (fishId, _, __) in trackingList[idx]]
        pastIdsMaster[idx] = ids

        if idx > 0:
            # check that none of the previous have already disapeared
            previousIds = pastIdsMaster[-1]
            for fishId in previousIds:
                if fishId not in ids: badIds.append(fishId)
    pastIdsMaster = pastIdsMaster[1:]

    # The Middle Case
    for frameIdx in range(framesRequired - 1, len(trackingList) - 1):
        frameData = trackingList[frameIdx]
        ids = [fishId for (fishId, _, __) in frameData]
        frameData = trackingList[frameIdx + 1]
        nextFrameIds = [fishId for (fishId, _, __) in frameData]

        for fishId in ids:
            if fishId not in nextFrameIds:
                # Lets check that it appeared atleast at the amount of framesRequired

                amountSeen = 1  # Already is in the frame of focus
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
        currated = [[fishId, box, fish] for (fishId, box, fish) in frameData if fishId not in badIds]
        newTrackingList.append(currated)

    return newTrackingList


def connectSwimBouts(newTrackingList):
    # Lets connect the swim bouts
    maxGap = 25
    maxGap = 75
    # maxGap = 150
    closenessThreshold = 20
    connectionsData = []

    amountOfFrames = len(newTrackingList)

    # for frameIdx in range(0, len(trackingList) - 1 ):
    for frameIdx in range(1, len(newTrackingList) - 2):
        ids = [fishId for (fishId, box, fish) in newTrackingList[frameIdx]]

        newIds = [fishId for (fishId, box, fish) in newTrackingList[frameIdx + 1]]
        for (fishId, box, fish) in newTrackingList[frameIdx]:

            if fishId not in newIds:
                # Let see if we can find it in one of the new frames
                pastFish = None
                for (pastFishId, tempBox, tempFish) in newTrackingList[frameIdx - 1]:
                    if pastFishId == fishId: pastFish = tempFish
                if pastFish is None: continue
                com = fish[2, :]
                pastCom = pastFish[2, :]
                v = com - pastCom

                for offsetIdx in range(2, maxGap + 2):
                    if (frameIdx + offsetIdx) >= amountOfFrames: break
                    estimateFish = (v * offsetIdx) + fish
                    offsetData = [[offsetFishId, offsetBox, offsetFish] for (offsetFishId, offsetBox, offsetFish) in
                                  newTrackingList[frameIdx + offsetIdx] if offsetFishId not in ids]
                    if len(offsetData) == 0: continue
                    # Lets get the optimal combination
                    offsetFish = [offsetDataEl[2] for offsetDataEl in offsetData]
                    #                     smallestLossInstances, pastFishOptimalCom, nextFishOptimalCom = getLossOfCombinations([fish], offsetFish)
                    # NOTE: Lets see how the modification makes it behave
                    smallestLossInstances, pastFishOptimalCom, nextFishOptimalCom = getLossCombinationsFast([fish],
                                                                                                            offsetFish)

                    if smallestLossInstances[0] < closenessThreshold:
                        # We found its connection
                        connectionsData.append([fishId, offsetData[nextFishOptimalCom[0]][0]])
                        break

    # Lets parse the connectionsData, so that it is easier to rewrite the ids
    #     if len(connectionsData) == 0: connectionsDataBetter[0] = connectionsData

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
        for (fishId, box, fish) in frameData:
            if fishId not in idDic:
                idDic[fishId] = smallestId
                smallestId += 1
            temp.append([idDic[fishId], box, fish])

        newTrackingList3.append(temp)

    return newTrackingList3


def fillOutData(newTrackingList3):
    masterD = dict()

    for frameIdx, frameData in enumerate(newTrackingList3):
        frameIds = [fishId for (fishId, box, fish) in frameData]

        for idSeen in masterD:
            d = masterD[idSeen]

            if idSeen in frameIds:
                if not d['past']:
                    gapList = d['gaps']
                    lastSeen = d['lastSeen']
                    gapList.append([lastSeen + 1, frameIdx - 1])
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
                newD = {'past': 1, 'start': frameIdx, 'end': frameIdx, 'gaps': [], 'lastSeen': frameIdx}
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

            [startFish, startBox] = [[fish, box] for (fishId, box, fish) in newTrackingList3[s - 1] if fishId == key][0]
            [endFish, endBox] = [[fish, box] for (fishId, box, fish) in newTrackingList3[e + 1] if fishId == key][0]
            distBox = endBox - startBox
            distBox = distBox / divisions

            distFish = endFish - startFish
            distFish = distFish / divisions

            for offsetIdx in range(length):
                newFish = startFish + ((offsetIdx + 1) * distFish)
                newBox = startBox + ((offsetIdx + 1) * distBox)
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
        for (fishId, box, fish) in frameData:
            if fishId not in idDic:
                idDic[fishId] = smallestId
                smallestId += 1
            temp.append([idDic[fishId], box, fish])

        newTrackingList3.append(temp)

    return newTrackingList3





# wasChecked = checkIfRecorded(frameIdx, overlapP, comsMasterList)
# Check if there are                 comsMasterList.append([frameIdx, comsList, amountList])
# wasChecked = checkIfRecorded(frameIdx, overlapP, comsMasterList)
# Check if there are                 comsMasterList.append([frameIdx, comsList, amountList])
def checkIfRecorded(frameIdx, overlapP, comsMasterList):
    foundNear = False
    distTreshold = 20  # The distance of how close the COMS have to be to each other

    for (startIdx, comsList, _, amountList) in comsMasterList:
        offsetIdx = frameIdx - startIdx
        if offsetIdx >= 0 and offsetIdx < len(comsList):
            dist = np.sum((comsList[offsetIdx] - overlapP) ** 2) ** .5
            if dist < distTreshold:
                foundNear = True
                break

    return foundNear


def connectOverlapBouts8(newTrackingList4, d2):
    print('Starting')
    timeThreshold = 500  # 1 second for us at 500 fps

    proximityThreshold = 20
    proximityThreshold = 40
    proximityThreshold = 60

    maxHalucinationLen = 20
    linkList = []  # Linking Data
    unlinkablelist = []
    startingLinks = []
    comsMasterList = []
    startingData = []
    # Algorith for linking the bouts
    bar = tqdm(total=len(newTrackingList4))
    for frameIdx, frameData in enumerate(newTrackingList4):
        bar.update(1)
        startingLinks = [el[0] for el in linkList]  # This are the connections that were already linked

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
                fishes = proxData['fishes'].copy()
                fishes.append([fishId, fish])

                coms = [fish2[2, :] for (_, fish2) in fishes]
                coms = np.array(coms)

                overlapP = np.mean(coms, axis=0)
                isOneNotThere = False
                for (_, fish2) in fishes:
                    if np.sum(((fish2[2, :] - overlapP) ** 2)) ** .5 > proximityThreshold / 2:
                        isOneNotThere = True
                        break

                #                 dist = np.sum(((fish[2,:] - overlapP) **2)) ** .5

                if not isOneNotThere:
                    foundProximity = True
                    proximityIdx = comIdx
                    break

            if foundProximity:

                proxData = proximityList[proximityIdx]
                fishes = proxData['fishes']
                fishes.append([fishId, fish])
                proxData['fishes'] = fishes

                # Update OverlapP
                coms = [fish[2, :] for (_, fish) in fishes]
                coms = np.array(coms)

                overlapP = np.mean(coms, axis=0)

                proxData['overlapP'] = overlapP

                proximityList[proximityIdx] = proxData

            else:
                proxData = {'fishes': [[fishId, fish]], 'overlapP': fish[2, :]}
                proximityList.append(proxData)

        #         if frameIdx == 1521:
        #             print('1521 list:')
        #             print(proximityList)

        # Lets check if there are any overlapping fishes
        shouldPrint = False
        for proxData in proximityList:
            overlapFishInfo = proxData['fishes'].copy()
            overlapFishInfoIds = [fishId for (fishId, fish) in overlapFishInfo]

            #             if frameIdx > 1555 and frameIdx < 1555 + 100 and 9 in overlapFishInfoIds:
            #                 shouldPrint = True
            #                 print('overlapFishInfoIds', overlapFishInfoIds,' at ', frameIdx)

            if len(proxData['fishes']) > 1:
                # It is an overlap instance
                overlapFishInfo = proxData['fishes'].copy()
                overlapFishInfoIds = [fishId for (fishId, fish) in overlapFishInfo]
                #             print('the overlap fish info: ',overlapFishInfoIds)
                # Checking to make sure it is not a case we already linked
                #                 if shouldPrint:
                #                     print(overlapFishInfoIds)
                #                     print(startingData)
                #                 if frameIdx > 1555 and frameIdx < 1555 + 100 and 9 in overlapFishInfoIds:
                #                     print('9 in at', frameIdx)

                ## Checking if it was already recorded
                #                 shouldBreak = False
                #                 for (fishInfoIds, startFrameIdx, endIdx) in startingData:
                #                     if len(fishInfoIds) == len(overlapFishInfoIds) and ( (frameIdx >= startFrameIdx) and (frameIdx <= endIdx)) :
                #                         # It could be that we have come across the same instance

                #                         areTheyAllThere = True
                #                         for fishId in fishInfoIds:
                #                             if fishId not in overlapFishInfoIds: areTheyAllThere = False # A difference is enough to prove that they are different

                #                         if areTheyAllThere == True:
                #                             shouldBreak = True
                #                             break
                # #                 if shouldBreak: break
                # #                 if shouldBreak and shouldPrint: print('blame: ', startingData)

                #                 if shouldBreak: continue

                if frameIdx >= 1521 and 6 in overlapFishInfoIds and 24 in overlapFishInfoIds:
                    shouldPrint = True
                else:
                    shouldPrint = False
                #                 if frameIdx > 1555 and frameIdx < 1555 + 100 and 9 in overlapFishInfoIds:
                #                     print('9 out')

                overlapP = proxData['overlapP']

                wasChecked = checkIfRecorded(frameIdx, overlapP, comsMasterList)
                if wasChecked: continue

                startingData.append([overlapFishInfoIds, frameIdx])  # We will add the ending frameIdx at the end

                # Check if there are                 comsMasterList.append([frameIdx, comsList, amountList])

                #             print('og overlapP:', overlapP)
                comsList = [overlapP]
                comsRadius = [proximityThreshold / 2]
                amountList = [len(overlapFishInfo)]

                needLinkList = []
                needLinkListRelative = []

                for offsetIdx, otherFrameData in enumerate(newTrackingList4[frameIdx + 1:]):
                    justLeftDistances = []
                    offsetFrameIdx = frameIdx + 1 + offsetIdx

                    #                     if shouldPrint: print(offsetFrameIdx)

                    #                     if frameIdx >= 1521 and 6 in overlapFishInfoIds:
                    #                         print('offsetFrameIdx: ', offsetFrameIdx)

                    newOverlapInfo = [[fishId, fish] for (fishId, _, fish) in otherFrameData if
                                      fishId in overlapFishInfoIds]
                    newOverlapInfoIds = [fishId for (fishId, fish) in newOverlapInfo]

                    nonOverlapInfo = [[fishId, fish] for (fishId, _, fish) in otherFrameData if
                                      fishId not in overlapFishInfoIds]
                    nonOverlapInfoIds = [fishId for (fishId, fish) in nonOverlapInfo]

                    overlapFish2Add = []
                    fish2Remove = []
                    subtractAmount = 0  # This is to subtract any halucinations that appear when fish are close

                    # Lets check if some of the new fish entered the overlapArea
                    for (fishId, fish) in nonOverlapInfo:
                        #                     if fishId == 5:
                        #                         print('the distance of 5:', np.sum((fish[2,:] - overlapP)** 2) ** .5)
                        #                         print('the frameIdx:', offsetFrameIdx)
                        #                         print('overlap point:', overlapP)

                        if np.sum((fish[2, :] - overlapP) ** 2) ** .5 < (proximityThreshold / 2):
                            # Lets just check that we need one added if it just appeared
                            if not (
                            (offsetFrameIdx - d2[fishId]['start'] < maxHalucinationLen and len(needLinkList) == 0)):
                                overlapFish2Add.append([fishId, fish])
                    #                 temp = [fishId for (fishId, fish) in overlapFish2Add]
                    #                 if 5 in temp and 6 in temp: print('your list:', overlapFish2Add)

                    if len(needLinkList) > 0 and len(overlapFish2Add) > 0:
                        # Let create a list of overlapFish2Add which has fish that where previously not clearly define before
                        overlapFish2AddDummy = []
                        for (fishId, fish) in overlapFish2Add:
                            amountDefined = (offsetFrameIdx - d2[fishId]['start']) + 1
                            if amountDefined < 50: overlapFish2AddDummy.append(
                                [fishId, fish])  # This is to make sure the fish just appeared
                        if len(overlapFish2AddDummy) > 0:
                            needLinkListFish = [fish for (fishId, fish, _) in needLinkList]
                            overlapFish2AddFish = [fish for (fishId, fish) in overlapFish2AddDummy]

                            # Finding the optimal combination, first output ignored since the are/where in the overlap region
                            _, needLinkListIndices, overlapFish2AddIndices = getLossCombinationsFast(needLinkListFish,
                                                                                                     overlapFish2AddFish)

                            optimalFishIds = [needLinkList[linkListIdx][0] for linkListIdx in needLinkListIndices]
                            optimalOverlapIds = [overlapFish2AddDummy[overlapListIdx][0] for overlapListIdx in
                                                 overlapFish2AddIndices]

                            for (id1, id2) in zip(optimalFishIds, optimalOverlapIds):  linkList.append([id1, id2])

                            needLinkList = [[fishId, fish, buddyId] for (fishId, fish, buddyId) in needLinkList if
                                            fishId not in optimalFishIds]

                    # Lets check that the fish in the overlap region have not disapeared or have left the overlap region
                    for (fishId, fish) in overlapFishInfo:
                        if fishId not in newOverlapInfoIds:
                            # It has disapeared
                            # Lets just check that it was not near an edge
                            pastFrameData = newTrackingList4[offsetFrameIdx - 1]
                            pastBox = None
                            for (pastFishId, box, pastFish) in pastFrameData:
                                if fishId == pastFishId:
                                    pastBox = box
                                    break
                            if isBoxFarFromEdge(pastBox):
                                needLinkList.append([fishId, fish])

                                # Lets find its closest buddy
                                minDist = 999
                                minDistId = None
                                for (fishId2, fish2) in newOverlapInfo:
                                    dist = np.sum((fish2[2, :] - fish[2, :]) ** 2) ** .5
                                    if dist < minDist:
                                        minDist = dist
                                        minDistId = fishId2
                                needLinkList[-1].append(minDistId)


                            else:
                                fish2Remove.append(fishId)  # Technically not needed ?
                        else:
                            # Its there, lets just check that it has not left
                            for (fishId2, fish2) in newOverlapInfo:
                                if fishId2 == fishId:
                                    fish = fish2
                                    break
                            dist = np.sum((fish[2, :] - overlapP) ** 2) ** .5
                            #                             if offsetFrameIdx > 1900 and fishId == 26 or fishId == 10 : print('dist', dist)

                            if dist > (proximityThreshold / 2):
                                justLeftDistances.append(dist)
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

                        comsList.append(overlapP)
                        if len(justLeftDistances) == 0:
                            comsRadius.append(proximityThreshold / 2)
                        else:
                            comsRadius.append(max(justLeftDistances))

                        # Otherwise lets update our com and continue
                        if len(overlapFishInfo) >= 1:
                            coms = [fish[2, :] for (fishId, fish) in overlapFishInfo]

                            # Lets put more weight on some due to disapearence
                            for (_, __, buddyId) in needLinkList:

                                for (fishId, fish) in overlapFishInfo:
                                    if fishId == buddyId:
                                        coms.append(fish[2, :])

                            coms = np.array(coms)

                            overlapP = np.mean(coms, axis=0)

                        # comsList.append(overlapP)
                        amountList.append(len(overlapFishInfo) + len(needLinkList))

                    if offsetFrameIdx + 1 >= len(newTrackingList4):
                        unlinkablelist += needLinkList
                        unlinkablelist += overlapFishInfo

                #             for (fishId, fish) in needLinkList: unlinkablelist.append(fishId)
                startingData[-1] = startingData[-1] + [offsetFrameIdx]  # Adding the endIdx

                comsMasterList.append([frameIdx, comsList, comsRadius, amountList])

        #                 if offsetFrameIdx + 1 <= len(newTrackingList4):
        #                     comsMasterList.append([frameIdx, comsList, amountList])

        #                     if shouldPrint:
        #                         print('we added to comsMasterList')

        #             print(offsetIdx)
        #             print('linkList: ',linkList)

        tqdm._instances.clear()

    # Start of the next algorithm
    # Let create a master link list
    connectionsDataBetter = []
    print('link list', linkList)
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
            connectionsDataBetter.append([l1, l2])
    # print('connections: ', connectionsDataBetter)
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


def eraseIncorrectPredictions2(comsMasterList, newTrackingList5, trackingList):
    trackingList = trackingList.copy()
    proximityThreshold = 20
    proximityThreshold = 40
    proximityThreshold = 60

    masterAmount = []
    for comData in tqdm(comsMasterList):
        startIdx, coms, _, amounts = comData
        amountOfComs = len(coms)
        amountList = []
        for offsetIdx in range(amountOfComs):
            frameData = newTrackingList5[startIdx + offsetIdx]
            com = coms[offsetIdx]
            amount = 0
            for (fishId, box, fish) in frameData:
                dist = np.sum((fish[2, :] - com) ** 2) ** .5
                if dist < (proximityThreshold / 2): amount += 1
            amountList.append(amount)
        amountList = np.array(amountList)
        m = stats.mode(amountList)
        masterAmount.append(m)

    delIndices = [[] for _ in range(len(newTrackingList5))]
    for (modeAmount, comData) in tqdm(zip(masterAmount, comsMasterList)):
        startIdx, coms, _, amounts = comData
        amountOfComs = len(coms)
        amountList = []
        for offsetIdx in range(amountOfComs):
            frameData = trackingList[startIdx + offsetIdx]
            com = coms[offsetIdx]
            amount = amounts[offsetIdx]
            badIdx = []

            for fishIdx, (fishId, box, fish) in enumerate(frameData):
                dist = np.sum((fish[2, :] - com) ** 2) ** .5
                if dist < (proximityThreshold / 2): badIdx.append(fishIdx)

            #             if len(badIdx) < amount: delIndices[startIdx + offsetIdx] += badIdx
            if len(badIdx) != amount: delIndices[startIdx + offsetIdx] += badIdx

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


def connectOverlapBoutsLast2(newTrackingList4, d2, comsMasterList2):
    proximityThreshold = 40
    proximityThreshold = 60
    linkList = []
    unlinkablelist = []
    maxHalucinationLen = 15
    for comIdx, (startIdx, coms, comRadii, amounts) in enumerate(comsMasterList2):
        needLinkList = []
        overlapFishInfo = []
        com = coms[0]

        frameData = newTrackingList4[startIdx]
        for (fishId, box, fish) in frameData:
            dist = np.sum((fish[2, :] - com) ** 2) ** .5
            if dist < (proximityThreshold / 2): overlapFishInfo.append([fishId, fish])

        if len(overlapFishInfo) < 2: continue
        overlapFishInfoIds = [fishId for (fishId, fish) in overlapFishInfo]

        for offsetIdx, (overlapP, overlapPRadius) in enumerate(zip(coms[1:], comRadii[1:])):
            offsetFrameIdx = (startIdx + 1) + offsetIdx

            otherFrameData = newTrackingList4[offsetFrameIdx]
            #                     if shouldPrint: print(offsetFrameIdx)

            #                     if frameIdx >= 1521 and 6 in overlapFishInfoIds:
            #                         print('offsetFrameIdx: ', offsetFrameIdx)

            newOverlapInfo = [[fishId, fish] for (fishId, _, fish) in otherFrameData if fishId in overlapFishInfoIds]
            newOverlapInfoIds = [fishId for (fishId, fish) in newOverlapInfo]

            nonOverlapInfo = [[fishId, fish] for (fishId, _, fish) in otherFrameData if
                              fishId not in overlapFishInfoIds]
            nonOverlapInfoIds = [fishId for (fishId, fish) in nonOverlapInfo]

            overlapFish2Add = []
            fish2Remove = []
            edgeFish = []
            subtractAmount = 0  # This is to subtract any halucinations that appear when fish are close

            if overlapPRadius > (proximityThreshold / 2):
                # print('looking for Edge Fish')
                for (fishId, _, fish) in otherFrameData:
                    dist = np.sum((fish[2, :] - overlapP) ** 2) ** .5
                    if fishId == 28:
                        print(dist)
                        print(overlapPRadius + 0)
                    if dist <= (overlapPRadius + 0) and dist > (proximityThreshold / 2):
                        # print('edgeFish: ', fishId)
                        edgeFish.append([fishId, fish])

            # Lets check if some of the new fish entered the overlapArea
            for (fishId, fish) in nonOverlapInfo:
                #                     if fishId == 5:
                #                         print('the distance of 5:', np.sum((fish[2,:] - overlapP)** 2) ** .5)
                #                         print('the frameIdx:', offsetFrameIdx)
                #                         print('overlap point:', overlapP)

                if np.sum((fish[2, :] - overlapP) ** 2) ** .5 < (proximityThreshold / 2):
                    # Lets just check that we need one added if it just appeared
                    if not ((offsetFrameIdx - d2[fishId]['start'] < maxHalucinationLen and len(needLinkList) == 0)):
                        overlapFish2Add.append([fishId, fish])
            #                 temp = [fishId for (fishId, fish) in overlapFish2Add]
            #                 if 5 in temp and 6 in temp: print('your list:', overlapFish2Add)

            overlapFish2AddTemp = overlapFish2Add.copy()
            overlapFish2AddTemp += edgeFish

            if len(needLinkList) > 0 and len(overlapFish2AddTemp) > 0:
                # Let create a list of overlapFish2Add which has fish that where previously not clearly define before
                overlapFish2AddDummy = []
                for (fishId, fish) in overlapFish2AddTemp:
                    amountDefined = (offsetFrameIdx - d2[fishId]['start']) + 1
                    if amountDefined < 50: overlapFish2AddDummy.append(
                        [fishId, fish])  # This is to make sure the fish just appeared
                if len(overlapFish2AddDummy) > 0:
                    needLinkListFish = [fish for (fishId, fish, _) in needLinkList]
                    overlapFish2AddFish = [fish for (fishId, fish) in overlapFish2AddDummy]

                    # Finding the optimal combination, first output ignored since the are/where in the overlap region
                    _, needLinkListIndices, overlapFish2AddIndices = getLossCombinationsFast(needLinkListFish,
                                                                                             overlapFish2AddFish)

                    optimalFishIds = [needLinkList[linkListIdx][0] for linkListIdx in needLinkListIndices]
                    optimalOverlapIds = [overlapFish2AddDummy[overlapListIdx][0] for overlapListIdx in
                                         overlapFish2AddIndices]

                    if 28 in optimalFishIds + optimalOverlapIds:
                        print('comIdx: ', comIdx, 'frameIdx: ', offsetFrameIdx)
                        for (id1, id2) in zip(optimalFishIds, optimalOverlapIds):
                            print('link: ', id1, id2)

                    for (id1, id2) in zip(optimalFishIds, optimalOverlapIds):  linkList.append([id1, id2])

                    needLinkList = [[fishId, fish, buddyId] for (fishId, fish, buddyId) in needLinkList if
                                    fishId not in optimalFishIds]

            # Lets check that the fish in the overlap region have not disapeared or have left the overlap region
            for (fishId, fish) in overlapFishInfo:
                if fishId not in newOverlapInfoIds:
                    # It has disapeared
                    # Lets just check that it was not near an edge
                    pastFrameData = newTrackingList4[offsetFrameIdx - 1]
                    pastBox = None
                    for (pastFishId, box, pastFish) in pastFrameData:
                        if fishId == pastFishId:
                            pastBox = box
                            break
                    if isBoxFarFromEdge(pastBox):
                        needLinkList.append([fishId, fish])

                        # Lets find its closest buddy
                        minDist = 999
                        minDistId = None
                        for (fishId2, fish2) in newOverlapInfo:
                            dist = np.sum((fish2[2, :] - fish[2, :]) ** 2) ** .5
                            if dist < minDist:
                                minDist = dist
                                minDistId = fishId2
                        needLinkList[-1].append(minDistId)


                    else:
                        fish2Remove.append(fishId)  # Technically not needed
                else:
                    # Its there, lets just check that it has not left
                    for (fishId2, fish2) in newOverlapInfo:
                        if fishId2 == fishId:
                            fish = fish2
                            break
                    dist = np.sum((fish[2, :] - overlapP) ** 2) ** .5
                    #                             if offsetFrameIdx > 1900 and fishId == 26 or fishId == 10 : print('dist', dist)

                    if dist > (proximityThreshold / 2):
                        fish2Remove.append(fishId)

            # Lets update the lists
            overlapFishInfo = [[fishId, fish] for (fishId, fish) in newOverlapInfo if fishId not in fish2Remove]
            for (fishId, fish) in overlapFish2Add: overlapFishInfo.append([fishId, fish])
            overlapFishInfoIds = [fishId for (fishId, fish) in overlapFishInfo]

            needLinkListIds = [fishId for (fishId, fish, _) in needLinkList]

            # if comIdx == 12:
            #    print('need link list: ', needLinkListIds, 'amount: ', len(needLinkList) + len(overlapFishInfo),'frameIdx: ', offsetFrameIdx)

            if len(overlapFishInfo) <= 1 and len(needLinkList) == 0:
                # Our overlapping region has only one fish and there are no fish need to be linked
                if len(overlapFishInfo) > 0:  unlinkablelist += [overlapFishInfo[0][0]]
                unlinkablelist += fish2Remove
                break
            else:

                # Otherwise lets update our com and continue
                if len(overlapFishInfo) >= 1:
                    coms = [fish[2, :] for (fishId, fish) in overlapFishInfo]

                    # Lets put more weight on some due to disapearence
                    for (_, __, buddyId) in needLinkList:

                        for (fishId, fish) in overlapFishInfo:
                            if fishId == buddyId:
                                coms.append(fish[2, :])

            #                     coms = np.array(coms)

            #                     overlapP = np.mean(coms, axis = 0)

            #                 comsList.append(overlapP)
            #                 amountList.append(len(overlapFishInfo) + len(needLinkList))

            if offsetFrameIdx + 1 >= len(newTrackingList4):
                unlinkablelist += needLinkList
                unlinkablelist += overlapFishInfo

    connectionsDataBetter = []
    print('link list', linkList)
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
            connectionsDataBetter.append([l1, l2])
    # print('connections: ', connectionsDataBetter)
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

    return newTrackingList5, comsMasterList2


def yoloFishEstimationPlusErasing(dataList):
    dataList2 = deleteDuplicates(dataList)

    trackingListOG = closenessTracking(dataList2)
    # trackingListOG = trackingList
    print('doing the initial analysis')
    trackingList2 = getRidOfBogusIds(trackingListOG)
    trackingList3 = connectSwimBouts(trackingList2)
    data = fillOutData(trackingList3)
    trackingList4 = fillOutListWithData(trackingList3, data)
    data = fillOutData(trackingList4)
    print('connectOverlapBouts algorithm starting')
    # trackingList5, comsMasterList = connectOverlapBouts3(trackingList4, data)
    # trackingList5, comsMasterList = connectOverlapBouts4(trackingList4, data)
    # trackingList5, comsMasterList = connectOverlapBouts6(trackingList4, data)
    # trackingList5, comsMasterList = connectOverlapBouts7(trackingList4, data)
    trackingList5, comsMasterList = connectOverlapBouts8(trackingList4, data)

    trackingList5 = turnIdsToSmallest(trackingList5)
    data2 = fillOutData(trackingList5)
    trackingList6 = fillOutListWithData(trackingList5, data2)
    # dataList3 = eraseIncorrectPredictions(comsMasterList, trackingList6, trackingListOG)
    # dataList3 = eraseIncorrectPredictions2(comsMasterList, trackingList6, trackingListOG)
    dataList3 = eraseIncorrectPredictions2(comsMasterList, trackingList6, trackingList6)

    trackingListOG2 = closenessTracking(dataList3)
    trackingList7 = getRidOfBogusIds(trackingListOG2)
    trackingList8 = connectSwimBouts(trackingList7)
    # trackingList8 = connectSwimBoutsConfident(trackingList7)

    data3 = fillOutData(trackingList8)
    trackingList9 = fillOutListWithData(trackingList8, data3)

    data4 = fillOutData(trackingList9)
    print('we are connecting the overlap again')
    # trackingList10, comsMasterList2 = connectOverlapBouts3(trackingList9, data4)
    # trackingList10, comsMasterList2 = connectOverlapBouts4(trackingList9, data4)
    # trackingList10, comsMasterList2 = connectOverlapBouts6(trackingList9, data4)
    # trackingList10, comsMasterList2 = connectOverlapBoutsLast(trackingList9, data4, comsMasterList)
    trackingList10, comsMasterList2 = connectOverlapBoutsLast2(trackingList9, data4, comsMasterList)

    trackingList10 = turnIdsToSmallest(trackingList10)
    data5 = fillOutData(trackingList10)
    trackingList = fillOutListWithData(trackingList10, data5)

    return trackingList, trackingListOG

# Lets get the video
folder1 = 'overlapVideos/020116_012/'
folder2 = 'overlapVideos/020116_013/'
folder3 = 'overlapVideos/020116_014/'
# The first folder is the one that actually is getting background subtraction
# The others are so that we get a better background
video = multipleBgsubList(folder1, folder2, folder3)

yoloWeights = 'inputs/weights/orthographic_yolo/best.pt'
model = YOLO(yoloWeights)

shouldGetData = True
shouldSaveDataList = True
savePath = 'OrthographicDataList.pkl'


if shouldGetData:
    # Now lets get the predictions from YOLO in a format for our tracking algorithm
    batchSize = 500
    # batchSize = 1
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
        frame = np.stack((frame, frame, frame), axis=2)

        frames.append(frame)

        fc += 1

        if len(frames) >= batchSize or fc == amountOfFrames:
            results = model.predict(frames, verbose=False, stream=True)
            for resultIdx, result in enumerate(results):
                frame = frames[resultIdx]
                confidence_mask = result.boxes.conf.cpu().numpy() > .6  # Making sure the predictions are accurate
                boxes = result.boxes.xyxy.cpu().numpy()[confidence_mask]
                classes = result.boxes.cls.cpu().numpy()[confidence_mask]
                keypoints = result.keypoints.xy.cpu().numpy()[confidence_mask]

                mask = [isBoxFarFromEdge(box) for box in
                        boxes]  # Making sure the boxes are not really close to the edge

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

if shouldSaveDataList:
    with open(savePath, 'wb') as b:
        pickle.dump(dataList, b)

dataList = deleteDuplicates(dataList)

trackingList, trackingListOG = yoloFishEstimationPlusErasing(dataList)

imageSizeX, imageSizeY = 640, 640
fps = 500
# fps = cap.get( cv.CAP_PROP_FPS )
fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
# fourcc = cv.VideoWriter_fourcc(*'DIVX')
# outputPath = 'outputs/superimposed/' + videoName + '.avi'
outputPath = 'sideVideo.avi'
out = cv.VideoWriter(outputPath, fourcc, int(fps), (int(imageSizeX), int(imageSizeY)))
red = [0, 0, 255]
green = [0, 255, 0]

# for frameIdx, data in tqdm(enumerate(newTrackingList4)):
# for frameIdx, data in tqdm(enumerate(trackingList)):
for frameIdx, data in tqdm(enumerate(trackingList), total=len(trackingList)):
    # for frameIdx, data in tqdm(enumerate(newTrackingList42)):
    # for frameIdx, data in tqdm(enumerate(trackingList)):
    # for frameIdx, data in tqdm(enumerate(newTrackingList2)):
    # for frameIdx, data in tqdm(enumerate(trackingListCopy)):
    # for frameIdx, data in tqdm(enumerate(newTrackingList3)):

    frame = video[frameIdx]
    frame = np.stack([frame, frame, frame], axis=2)

    for (fishId, box, fish) in data:
        com = fish[2, :]
        com = com.astype(int)
        font = cv.FONT_HERSHEY_SIMPLEX

        # Use putText() method for
        # inserting text on video

        # Drawing the keypoints
        for keypointIdx in range(len(fish)):
            keypoint = fish[keypointIdx, :]
            color = green if keypointIdx < 10 else red
            keypoint = keypoint.astype(int)
            frame = cv.circle(frame, keypoint, 1, color, -1)

        frame = cv.putText(frame, str(fishId), (com[0], com[1]), font, 1, (0, 255, 255), 2, cv.LINE_4)
        sx, sy, bx, by = box.astype(int)
        frame = cv.rectangle(frame, (sx, sy), (bx, by), color=red, thickness=2)

    out.write(frame)

out.release()































