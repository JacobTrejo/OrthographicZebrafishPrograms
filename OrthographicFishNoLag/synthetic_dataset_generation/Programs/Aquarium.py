from Programs.Config import Config
import numpy as np
import cv2 as cv
from Programs.programsForGeneratingFishFrames import generateRandomConfiguration, generateRandomConfigurationNoLagChunks, generateRandomConfigurationFast, generateRandomConfigurationFastStates
from Programs.Auxilary import add_noise_static_noise, add_patchy_noise, mergeViews, createDepthArr, add_noisy_spots
from Programs.programsForDrawingImage_for_frames import f_x_to_model_bigger
from itertools import groupby
from pycocotools import mask as maskUtils
import pdb
# Configuration variables
imageSizeY, imageSizeX = Config.imageSizeY, Config.imageSizeX

class Aquarium:
    # Static Variables
    aquariumVariables = ['fishInAllViews', 'fishInEdges','overlapping']
    fishVectListKey = 'fishVectList'

    # upperbounds for  0, overlap, 1 fish overlap, 2 fish overlap ....
    # overlapMarker = [.5, .95, 1.0]
    overlapMarker = Config.overlapMarker
    constantFishesInArena = Config.constantFishesInArena
    def overloaded_constructorOriginal(self, **kwargs):
        aquariumVariablesDict = {'fishInAllViews':0, 'fishInEdges':0, 'overlapping':0}

        wasAnAquariumVariableDetected = False
        wasAnAquariumPassed = False
        for key in kwargs:
            if key in Aquarium.aquariumVariables:
                aquariumVariablesDict[key] = kwargs.get(key)
                wasAnAquariumVariableDetected = True
            if key is Aquarium.fishVectListKey:
                wasAnAquariumPassed = True
                fishVectList = kwargs.get(key)

        if not wasAnAquariumPassed:
            if wasAnAquariumVariableDetected:
                fishVectList = self.generateFishListGivenVariables(aquariumVariablesDict)
            else:
                fishesInView = Config.maxFishesInView
                fishesInEdge = np.random.poisson(Config.averageFishInEdges)
                overlappingFish = 0
                for _ in range(fishesInView + fishesInEdge):
                    shouldItOverlap = True if np.random.rand() < Config.overlappingFishFrequency else False
                    if shouldItOverlap: overlappingFish += 1
                # fishVectList = generateRandomConfiguration(fishesInView, fishesInEdge, overlappingFish)
                #fishVectList = generateRandomConfigurationNoLagChunks(fishesInView, fishesInEdge, overlappingFish)

                fishVectList = generateRandomConfigurationFast(fishesInView, fishesInEdge, overlappingFish)
        return fishVectList

    def overloaded_constructor(self, **kwargs):
        aquariumVariablesDict = {'fishInAllViews':0, 'fishInEdges':0, 'overlapping':0}

        wasAnAquariumVariableDetected = False
        wasAnAquariumPassed = False
        for key in kwargs:
            if key in Aquarium.aquariumVariables:
                aquariumVariablesDict[key] = kwargs.get(key)
                wasAnAquariumVariableDetected = True
            if key is Aquarium.fishVectListKey:
                wasAnAquariumPassed = True
                fishVectList = kwargs.get(key)

        if not wasAnAquariumPassed:
            if wasAnAquariumVariableDetected:
                fishVectList = self.generateFishListGivenVariables(aquariumVariablesDict)
            else:
                overlapStates = []
                fishesInView = Config.maxFishesInView
                fishesInEdge = np.random.poisson(Config.averageFishInEdges)
                overlappingFish = 0
                for _ in range(fishesInView + fishesInEdge):
                    overlapState = np.random.rand()
                    overlapStates.append(overlapState)
                    #shouldItOverlap = True if np.random.rand() < Config.overlappingFishFrequency else False
                    #if shouldItOverlap: overlappingFish += 1
                # fishVectList = generateRandomConfiguration(fishesInView, fishesInEdge, overlappingFish)
                #fishVectList = generateRandomConfigurationNoLagChunks(fishesInView, fishesInEdge, overlappingFish)
                #fishVectList = generateRandomConfigurationFast(fishesInView, fishesInEdge, overlappingFish)
                fishVectList = generateRandomConfigurationFastStates(fishesInView, fishesInEdge, overlapStates, Aquarium.overlapMarker)
        return fishVectList


    def generateFishListGivenVariables(self, aquariumVariablesDict):
        fishInAllViews = aquariumVariablesDict.get('fishInAllViews')
        overlapping = aquariumVariablesDict.get('overlapping')
        fishInEdges = aquariumVariablesDict.get('fishInEdges')
        # fishVectList = generateRandomConfiguration(fishInAllViews, fishInEdges, overlapping)
        # fishVectList = generateRandomConfigurationNoLag(fishInAllViews, fishInEdges, overlapping)
        # fishVectList = generateRandomConfigurationNoLagChunks(fishInAllViews, fishInEdges, overlapping)
        fishVectList = generateRandomConfigurationFast(fishInAllViews, fishInEdges, overlapping) 
        return fishVectList


    def __init__(self, frame_idx, cam='bottom', **kwargs):
        # Getting the configuration settings
        self.cam = cam
        maxFishesInView = Config.maxFishesInView
        averageFishInEdges = Config.averageFishInEdges
        overlappingFishFrequency = Config.overlappingFishFrequency
        self.shouldAddStaticNoise = Config.shouldAddStaticNoise
        self.shouldAddPatchyNoise = Config.shouldAddStaticNoise
        self.shouldAddNoisySpots = Config.shouldAddNoisySpots
        self.shouldSaveAnnotations = Config.shouldSaveAnnotations
        self.shouldSaveImages = Config.shouldSaveImages

        fishVectList = self.overloaded_constructor(**kwargs)
        self.fish_list = []
        for fishVect in fishVectList:
            fish = Fish(fishVect, cam)
            self.fish_list.append(fish)

        self.views_list = []
        self.finalViews = []
        self.frame_idx = frame_idx
        # NOTE: the following variable is more of a constant
        self.amount_of_cameras = 1

    def add_static_noise_to_views(self):
        for viewIdx, view in enumerate(self.finalViews):
            graymodel = view[0]
            depth = view[1]
            noisey_graymodel = add_noise_static_noise(graymodel)
            # TODO: dont use tuples since they are immutable
            noisey_view = (noisey_graymodel, depth)
            # updating
            self.finalViews[viewIdx] = noisey_view

    def add_patchy_noise_to_views(self):
        for viewIdx, view in enumerate(self.finalViews):
            graymodel = view[0]
            depth = view[1]
            noisey_graymodel = add_patchy_noise(graymodel, self.fish_list)
            # TODO: dont use tuples since they are immutable
            noisey_view = (noisey_graymodel, depth)
            # updating
            self.finalViews[viewIdx] = noisey_view

    def add_noisy_spots_to_views(self):
        for viewIdx, view in enumerate(self.finalViews):
            graymodel = view[0]
            depth = view[1]
            noisey_graymodel = add_noisy_spots(graymodel, self.fish_list)
            noisey_view = (noisey_graymodel, depth)
            self.finalViews[viewIdx] = noisey_view

    def save_annotations(self):
        biggestIdx4TrainingData = Config.biggestIdx4TrainingData
        dataDirectory = Config.dataDirectory
        subFolder = 'train/' if self.frame_idx < biggestIdx4TrainingData else 'val/'
        labelsPath = dataDirectory + '/' + 'labels/' + subFolder
        strIdxInFormat = format(self.frame_idx, '06d')
        filename = 'zebrafish_' + strIdxInFormat + '.txt'
        labelsPath += filename

        # Creating the annotations
        f = open(labelsPath, 'w')

        for fish in (self.fish_list):
            # for fish in (fishVectList + overlappingFishVectList):
            boundingBox = fish.boundingBox
            segmentation = fish.segmentation
            area = fish.area
            # Should add a method to the bounding box, boundingBox.isSmallFishOnEdge()
            if fish.is_valid_fish:
                f.write(str(0) + ' ')
                f.write(
                    str(boundingBox.getCenterX() / imageSizeX) + ' ' + str(boundingBox.getCenterY() / imageSizeY) + ' ')
                f.write(
                    str(boundingBox.getWidth() / imageSizeX) + ' ' + str(boundingBox.getHeight() / imageSizeY) + ' ')

                xArr = fish.xs
                yArr = fish.ys
                vis = fish.vis
                for pointIdx in range(12):
                    # Visibility is set to zero if they are out of bounds
                    # Just got to clip them so that YOLO does not throw an error
                    x = np.clip(xArr[pointIdx], 0, imageSizeX - 1)
                    y = np.clip(yArr[pointIdx], 0, imageSizeY - 1)
                    f.write(str(x / imageSizeX) + ' ' + str(y / imageSizeY)
                            + ' ' + str(int(vis[pointIdx])) + ' ')
                
                f.write(str(segmentation) + ' ')
                f.write(str(area) + ' ')
                f.write('\n')

    def save_image(self):
        biggestIdx4TrainingData = Config.biggestIdx4TrainingData
        dataDirectory = Config.dataDirectory

        subFolder = 'train/' if self.frame_idx < biggestIdx4TrainingData else 'val/'
        imagesPath = dataDirectory + '/' + 'images/' + subFolder
        strIdxInFormat = format(self.frame_idx, '06d')
        filename = 'zebrafish_' + strIdxInFormat + '.png'
        imagesPath += filename
        success = cv.imwrite(imagesPath, self.finalViews[0][0])

    # For Debbuging
    def get_image(self):
        return self.finalViews[0][0]


    def draw(self):
        # drawing the fishes
        for fish in self.fish_list:
            fish.draw()
            self.views_list.append(fish.views)

        # merging the images
        if len(self.views_list) != 0:
            self.finalViews = mergeViews(self.views_list)
        else:
            for viewIdx in range(self.amount_of_cameras):
                view = (np.zeros((imageSizeY, imageSizeX)), np.zeros((imageSizeY, imageSizeX)))
                self.finalViews.append(view)

        # updating the visibility for the cases where a fish ends up covering another fish
        for fishIdx, fish in enumerate(self.fish_list):
            fish.update_visibility(self.finalViews)
            # You have update the fish list, because python is wierd
            self.fish_list[fishIdx] = fish

        if self.shouldAddStaticNoise:
            self.add_static_noise_to_views()

        if self.shouldAddPatchyNoise:
            self.add_patchy_noise_to_views()

        if self.shouldAddNoisySpots:
            self.add_noisy_spots_to_views()

class Fish:
    class BoundingBox:
        BoundingBoxThreshold = Config.boundingBoxThreshold
        def __init__(self, smallY, bigY, smallX, bigX):
            self.smallY = smallY
            self.bigY = bigY
            self.smallX = smallX
            self.bigX = bigX

        def getHeight(self):
            return (self.bigY - self.smallY)

        def getWidth(self):
            return (self.bigX - self.smallX)

        def getCenterX(self):
            return ((self.bigX + self.smallX) / 2)

        def getCenterY(self):
            return ((self.bigY + self.smallY) / 2)

        def isValidBox(self):
            height = self.getHeight()
            width = self.getWidth()

            if (height <= Fish.BoundingBox.BoundingBoxThreshold) or (width <= Fish.BoundingBox.BoundingBoxThreshold):
                return False
            else:
                return True

    def __init__(self, fishVect, cam='bottom'):
        self.seglen = fishVect[0]
        self.z = fishVect[1]
        self.x = fishVect[2:]
        self.cam = cam


    def get_mask_annotation(self, graymodel):
        mask = np.zeros((graymodel.shape))
        mask[graymodel > 0] = 1
        return mask


    def binary_mask_to_rle(self, binary_mask):
        """
        This method was obtained from https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
        """

        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        return rle
    

    def binary_mask_to_rle_pycoco(self, binary_mask):
        rle = maskUtils.encode(np.asarray(binary_mask, order="F"))
        return rle
    
    def binary_mask_to_coco_polygons(self, binary_image):
        # Find contours in the binary image
        contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.01 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            # Flatten the polygon points and convert to a list
            polygon = approx.flatten().tolist()
            polygons.append(polygon)
        return polygons
    

    def draw(self):
        cam = self.cam
        graymodel, pts = f_x_to_model_bigger(self.x, self.seglen, Config.randomizeFish, cam, imageSizeX, imageSizeY)
        depth = np.ones(pts[0,:].shape) * self.z
        depth_im = createDepthArr(graymodel, pts[0,:], pts[1,:], depth)
        # TODO: fill out these depth images since for the orthographic projections the fish can have spots
        camera1View = (graymodel, depth_im)
        self.views = [camera1View]

        self.pts = pts
        self.graymodel = graymodel

        self.vis = np.zeros((pts.shape[1]))
        self.vis[self.valid_points_masks] = 1

        # marking the depth of the points, will be used later to find their visibility
        marked_depth_at_keypoints = depth_im[self.intYs[self.valid_points_masks],
                                             self.intXs[self.valid_points_masks]]
        self.depth = np.zeros(self.xs.shape)
        self.depth[self.valid_points_masks] = marked_depth_at_keypoints


        # Creating the bounding box
        nonzero_coors = np.array(np.where(graymodel > 0))
        try:
            smallY = np.min(nonzero_coors[0, :])
            bigY = np.max(nonzero_coors[0, :])
            smallX = np.min(nonzero_coors[1, :])
            bigX = np.max(nonzero_coors[1, :])
        except:
            smallY = 0
            bigY = 0
            smallX = 0
            bigX = 0
        self.boundingBox = Fish.BoundingBox(smallY, bigY, smallX, bigX)
        mask = self.get_mask_annotation(graymodel).astype(np.uint8)
        self.area = np.sum(mask == 1)
        rle_mask = self.binary_mask_to_coco_polygons(mask)
        #self.segmentation = rle_mask['counts']
        self.segmentation = rle_mask


    @property
    def xs(self):
        return self.pts[0, :]

    @property
    def ys(self):
        return self.pts[1, :]

    @property
    def intXs(self):
        return np.ceil(self.pts[0, :]).astype(int)

    @property
    def intYs(self):
        return np.ceil(self.pts[1, :]).astype(int)

    @property
    def valid_points_masks(self):
        xs = self.intXs
        ys = self.intYs
        xs_in_bounds = (xs < imageSizeX) * (xs >= 0)
        ys_in_bounds = (ys < imageSizeY) * (ys >= 0)
        return xs_in_bounds * ys_in_bounds

    def amount_of_vis_points(self):
        val_xs = self.pts[0, :][self.valid_points_masks]
        return val_xs.shape[0]

    def update_visibility(self, finalViews):
        finalView1 = finalViews[0]
        finalDepth = finalView1[1]

        previous_marked_depths = self.depth[self.valid_points_masks]
        final_marked_depths = finalDepth[self.intYs[self.valid_points_masks],
                                         self.intXs[self.valid_points_masks]]
        still_vis = final_marked_depths == previous_marked_depths

        # have to do it this way because python is wierd with the references
        tempVis = np.ones((self.vis).shape)
        tempVis[self.valid_points_masks] = still_vis
        self.vis *= tempVis

    @property
    def is_valid_fish(self):
        if (self.amount_of_vis_points() >= 1) and self.boundingBox.isValidBox():
            return True
        else:
            return False
