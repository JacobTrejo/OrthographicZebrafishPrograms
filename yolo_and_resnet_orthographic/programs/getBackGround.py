import numpy as np
import cv2 as cv
import os
import pdb
import random


def bgsubList(folderName, filenames):
    '''
    This version is less prone to memory errors
    '''

    frameCount = len(filenames)
    nSampFrame = min(np.fix(frameCount / 2), 100)

    frameNumbers = list((np.fix(np.linspace(1, frameCount, int(nSampFrame))) - 1).astype(int))
    firstSampFrame = True
    secondSampFrame = True
    for frameNumber in frameNumbers:
        frame = cv.imread(folderName + '/' + filenames[frameNumber])
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if firstSampFrame:
            firstSampFrame = False
            sampFrames = frame
            continue
        if secondSampFrame:
            secondSampFrame = False
            sampFrames = np.stack((sampFrames, frame), axis=0)
            continue
        # sampFrames = np.stack((sampFrames, frame), axis = 0)
        sampFrames = np.vstack((sampFrames, frame[None, :, :]))

    sampFrames.sort(0)

    videobg = sampFrames[int(np.fix(nSampFrame * .9))]

    outputVid = []
    for frameIdx in range(frameCount):
        frame = cv.imread(folderName + '/' + filenames[frameIdx])
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Subtract foreground from background image by allowing values beyond the 0 to 255 range of uint8
        difference_img = np.int16(videobg) - np.int16(frame)

        # Clip values in [0,255] range
        difference_img = np.clip(difference_img, 0, 255)

        # Convert difference image to uint8 for saving to video
        difference_img = np.uint8(difference_img)
        outputVid.append(difference_img)

    return outputVid


def bgsubListV2(folderName):
    '''
    This version is less prone to memory errors
    '''

    files = os.listdir(folderName)
    filenames = [fileName for fileName in files if fileName.endswith('.bmp')]
    filenames.sort()

    frameCount = len(filenames)
    nSampFrame = min(np.fix(frameCount / 2), 100)

    frameNumbers = list((np.fix(np.linspace(1, frameCount, int(nSampFrame))) - 1).astype(int))
    firstSampFrame = True
    secondSampFrame = True
    for frameNumber in frameNumbers:
        frame = cv.imread(folderName + '/' + filenames[frameNumber])
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if firstSampFrame:
            firstSampFrame = False
            sampFrames = frame
            continue
        if secondSampFrame:
            secondSampFrame = False
            sampFrames = np.stack((sampFrames, frame), axis=0)
            continue
        # sampFrames = np.stack((sampFrames, frame), axis = 0)
        sampFrames = np.vstack((sampFrames, frame[None, :, :]))

    sampFrames.sort(0)

    videobg = sampFrames[int(np.fix(nSampFrame * .9))]

    indices = np.unravel_index(np.argsort(videobg, axis=None), videobg.shape)
    x, y = indices
    amount = len(x)
    ratio = .15
    bottomValues = videobg[x[:int(amount * ratio)], y[:int(amount * ratio)]]
    # topValues = videobg[ x[int(amount * .9):], y[int(amount * .9):]  ]

    amountOfBottoms = int(amount * ratio)
    randomIndices = random.sample(range(amountOfBottoms, amount), amountOfBottoms)
    videobg[x[:int(amount * ratio)], y[:int(amount * ratio)]] = videobg[x[randomIndices], y[randomIndices]]

    # amountOfTops = amount - int(amount * .9)
    # amountOfBottoms = amount - amountOfTops
    # randomIndices = random.sample( range(amountOfBottoms) ,amountOfTops)
    # randomValues = videobg[ y[randomIndices], x[randomIndices] ]
    # videobg[ x[int(amount * .9):], y[int(amount * .9):] ] = randomValues

    # cv.imwrite('sideVideo.png', videobg)
    # exit()

    # print('top', np.max(topValues))
    # print('bottom', np.max(bottomValues))
    # print(len(indices))
    # print(indices[0].shape)
    # exit()

    outputVid = []
    for frameIdx in range(frameCount):
        frame = cv.imread(folderName + '/' + filenames[frameIdx])
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Subtract foreground from background image by allowing values beyond the 0 to 255 range of uint8
        difference_img = np.int16(videobg) - np.int16(frame)

        # Clip values in [0,255] range
        difference_img = np.clip(difference_img, 0, 255)

        # Convert difference image to uint8 for saving to video
        difference_img = np.uint8(difference_img)
        outputVid.append(difference_img)

    return outputVid


def multipleBgsubList(*folderNames):
    '''
    This version is less prone to memory errors
    '''

    for folderIdx, folderName in enumerate(folderNames):
        files = os.listdir(folderName)
        filenames = [fileName for fileName in files if fileName.endswith('.bmp')]
        filenames.sort()
        frameCount = len(filenames)
        nSampFrame = min(np.fix(frameCount / 2), 100)

        frameNumbers = list((np.fix(np.linspace(1, frameCount, int(nSampFrame))) - 1).astype(int))
        firstSampFrame = True
        secondSampFrame = True
        for frameNumber in frameNumbers:
            frame = cv.imread(folderName + '/' + filenames[frameNumber])
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if firstSampFrame:
                firstSampFrame = False
                sampFrames = frame
                continue
            if secondSampFrame:
                secondSampFrame = False
                sampFrames = np.stack((sampFrames, frame), axis=0)
                continue
            # sampFrames = np.stack((sampFrames, frame), axis = 0)
            sampFrames = np.vstack((sampFrames, frame[None, :, :]))

        sampFrames.sort(0)

        if folderIdx == 0:
            videobg = sampFrames[int(np.fix(nSampFrame * .9))]
            cv.imwrite('bg.png', videobg)
        else:
            temp = sampFrames[int(np.fix(nSampFrame * .9))]
            cv.imwrite('bg' + str(folderIdx) + '.png', temp)
            videobg = np.maximum(videobg, temp)

    cv.imwrite('bg_final.png', videobg)
    # exit()

    folderName = folderNames[0]
    files = os.listdir(folderName)
    files.sort()
    filenames = [fileName for fileName in files if fileName.endswith('.bmp')]

    outputVid = []
    for frameIdx in range(frameCount):
        frame = cv.imread(folderName + '/' + filenames[frameIdx])
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # if frameIdx == 100:
        #    cv.imwrite('frameOg.png', frame)

        # Subtract foreground from background image by allowing values beyond the 0 to 255 range of uint8
        difference_img = np.int16(videobg) - np.int16(frame)

        # Clip values in [0,255] range
        difference_img = np.clip(difference_img, 0, 255)

        # Convert difference image to uint8 for saving to video
        difference_img = np.uint8(difference_img)
        outputVid.append(difference_img)

    return outputVid


if __name__ == '__main__':
    # folderName = '020116_023/'
    # files = os.listdir(folderName)
    # files = [fileName for fileName in files if fileName.endswith('.bmp')]
    # files.sort()

    # video = bgsubList(folderName, files)
    # video = multipleBgsubList('020116_023/','020116_024/')

    # video = multipleBgsubList('overlapVideos/020116_001/','overlapVideos/020116_002/')
    video = multipleBgsubList('overlapVideos/020116_001/')

    print('you are in getBackgroud')  # Just to check in you are running extra computations
    # video = bgsubListV2('020116_023/')

    imageSizeX, imageSizeY = 640, 640
    fps = 500
    # fps = cap.get( cv.CAP_PROP_FPS )
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv.VideoWriter_fourcc(*'DIVX')
    # outputPath = 'outputs/superimposed/' + videoName + '.avi'
    outputPath = 'sideVideo.avi'
    out = cv.VideoWriter(outputPath, fourcc, int(fps), (int(imageSizeX), int(imageSizeY)))

    # for frameIdx, frame in enumerate(video):
    #    if frameIdx == 100:
    #        cv.imwrite('frame.png', frame)

    for frame in video:
        frame = np.stack((frame, frame, frame), axis=2)
        out.write(frame)

    out.release()










