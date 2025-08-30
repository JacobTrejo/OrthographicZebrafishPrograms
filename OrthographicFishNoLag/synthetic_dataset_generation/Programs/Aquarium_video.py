import pdb
from Programs.Video_Config import Config
import numpy as np
from Programs.Auxilary import roundHalfUp
import cv2 as cv
from Programs.programsForGeneratingFishVideos import generateRandomConfiguration, generateRandomConfigurationNoLagChunks, generateRandomConfigurationFast, generateRandomVideoConfigurationFastStates, generateRandomVideoConfigurationFastStates
from Programs.Auxilary import add_noise_static_noise, add_patchy_noise, mergeViews, createDepthArr
from Programs.programsForDrawingImage_for_videos import f_x_to_model_bigger
from itertools import groupby
from pycocotools import mask as maskUtils
import os
import json
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
                #fishesInEdge = np.random.poisson(Config.averageFishInEdges)
                overlappingFish = 0
                for _ in range(fishesInView):
                    overlapState = np.random.rand()
                    overlapStates.append(overlapState)
                    #shouldItOverlap = True if np.random.rand() < Config.overlappingFishFrequency else False
                    #if shouldItOverlap: overlappingFish += 1
                # fishVectList = generateRandomConfiguration(fishesInView, fishesInEdge, overlappingFish)
                #fishVectList = generateRandomConfigurationNoLagChunks(fishesInView, fishesInEdge, overlappingFish)
                #fishVectList = generateRandomConfigurationFast(fishesInView, fishesInEdge, overlappingFish)
                if Config.maxFrames < 21:
                    numFrames = Config.maxFrames
                else:
                    numFrames = np.random.randint(Config.maxFrames-1, Config.maxFrames)
                fishVectList = generateRandomVideoConfigurationFastStates(fishesInView, [], overlapStates, Aquarium.overlapMarker, numFrames, self.video_idx) # Empty argument is fishesInEdges
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


    def __init__(self, video_idx, **kwargs):
        # Getting the configuration settings
        maxFishesInView = Config.maxFishesInView
        averageFishInEdges = Config.averageFishInEdges
        overlappingFishFrequency = Config.overlappingFishFrequency
        self.shouldAddStaticNoise = Config.shouldAddStaticNoise
        self.shouldAddPatchyNoise = Config.shouldAddStaticNoise
        self.shouldSaveAnnotations = Config.shouldSaveAnnotations
        self.shouldSaveImages = Config.shouldSaveImages
        self.video_idx = video_idx

        # Dataset for json file
        fishVectList = self.overloaded_constructor(**kwargs) # List of length numFish, each element is a (numFrames, 24) ndarray)
        fishVectList_shuffled = []
        for frame_id in range(fishVectList[0].shape[0]):
            fishVectList_shuffled.append([fishVect[frame_id, :] for fishVect in fishVectList])        
        self.fish_list = [[] for _ in range(len(fishVectList_shuffled))] # List of length numFrames

        for frame_id, fishVectList in enumerate(fishVectList_shuffled):
            for fish_id, fishVect in enumerate(fishVectList):
                fish = Fish(fishVect, fish_id=fish_id, video_id=video_idx)
                self.fish_list[frame_id].append(fish)                
        self.views_list = [[] for _ in range(len(self.fish_list))] # List of length numFrames
        self.finalViews = [[] for _ in range(len(self.fish_list))] # List of length numFrames
        # NOTE: the following variable is more of a constant
        self.amount_of_cameras = 1


    def add_static_noise_to_views(self):
        filter_size = 2 * roundHalfUp(np.random.rand()) + 3
        sigma = np.random.rand() + 0.5
        noise_mean = (np.random.rand() * np.random.normal(75, 10)) / 255
        noise_var = noise_var = (np.random.rand() * 60 + 20) / 255 ** 2
        for frame_id in range(len(self.finalViews)):            
            for viewIdx, view in enumerate(self.finalViews[frame_id]):
                graymodel = view[0]
                depth = view[1]
                noisey_graymodel = add_noise_static_noise(graymodel, 
                                                          filter_size=filter_size, 
                                                          sigma=sigma, 
                                                          noise_mean=noise_mean, 
                                                          noise_var=noise_var)
                # TODO: dont use tuples since they are immutable
                noisey_view = (noisey_graymodel, depth)
                # updating
                self.finalViews[frame_id][viewIdx] = noisey_view


    def add_patchy_noise_to_views(self):
        pvar = np.random.poisson(Config.averageAmountOfPatchyNoise)
        for frame_id in range(len(self.finalViews)):            
            for viewIdx, view in enumerate(self.finalViews[frame_id]):
                graymodel = view[0]
                depth = view[1]
                noisey_graymodel = add_patchy_noise(graymodel, self.fish_list[frame_id], pvar)
                # TODO: dont use tuples since they are immutable
                noisey_view = (noisey_graymodel, depth)
                # updating
                self.finalViews[frame_id][viewIdx] = noisey_view


    def save_annotations(self):
        biggestIdx4TrainingData = Config.biggestIdx4TrainingData
        dataDirectory = Config.dataDirectory
        subFolder = 'train/' if self.video_idx < biggestIdx4TrainingData else 'val/'
        labelsPath = dataDirectory + '/' + 'labels/' + subFolder
        strIdxInFormat = format(self.video_idx, '06d')
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


    def save_video_annotations_COCO(self):
        biggestIdx4TrainingData = Config.biggestIdx4TrainingData
        dataDirectory = Config.dataDirectory
        subFolder = 'train/' if self.video_idx < biggestIdx4TrainingData else 'val/'
        labelsPath = dataDirectory + '/' + 'labels/' + subFolder
        strIdxInFormat = format(self.video_idx, '06d')
        filename = 'zebrafish_' + strIdxInFormat + '.json'
        labelsPath += filename
        coco_seg = [[] for _ in range(len(self.fish_list[0]))] # List of length numFish
        coco_pose = [[] for _ in range(len(self.fish_list[0]))] # List of length numFish
        coco_area = [[] for _ in range(len(self.fish_list[0]))] # List of length numFish
        coco_bbox = [[] for _ in range(len(self.fish_list[0]))] # List of length numFish
        videoFolder = dataDirectory + '/' + 'images/' + subFolder
        videoFilePath = os.path.join(videoFolder, 'zebrafish_' + strIdxInFormat + '.mp4')
        video = {"id": self.video_idx, "width": imageSizeX, "height": imageSizeY, "filename": videoFilePath}
        coco_annotations = []
    
        for fish_id, _ in enumerate(self.fish_list[0]):
            coco_seg = []
            coco_pose = []
            coco_area = []
            coco_bbox = []
            visibility = []
            for frame_idx in range(len(self.fish_list)):
                fish = self.fish_list[frame_idx][fish_id]
                boundingBox = [float(fish.boundingBox.getCenterX() / imageSizeX), 
                               float(fish.boundingBox.getCenterY() / imageSizeY), 
                               float(fish.boundingBox.getWidth() / imageSizeX), 
                               float(fish.boundingBox.getHeight() / imageSizeY)]
                
                segmentation = fish.segmentation
                poseX = fish.xs.tolist()
                poseY = fish.ys.tolist()
                pose = [poseX, poseY]
                area = float(fish.area)
                # if fish.is_valid_fish: # implement this to add visibility
                coco_seg.append(segmentation)
                coco_pose.append(pose)
                coco_area.append(area)
                coco_bbox.append(boundingBox)
                vis = [bool(i) for i in fish.vis.tolist()]
                visibility.append(fish.vis.tolist())
        
            coco_annotations.append({"id": fish_id,
                                    "video_id": self.video_idx,
                                    "category_id": 1,
                                    "segmentations": coco_seg,
                                    "poses": coco_pose,
                                    "areas": coco_area,
                                    "bboxes": coco_bbox,
                                    "iscrowd": 0,
                                    "visibility": visibility,
                                    "occlusion": "slight_occlusion"})
        categories = [{"id": 1, "name": "fish", "supercategory": "fish"}]
        dataset = {
            "info": {"description": "Danionella/Zebrafish video dataset"}, 
            "videos": video,
            "annotations": coco_annotations,
            "categories": categories,
        }
        with open(labelsPath, 'w') as f:
            json.dump(dataset, f)
        print(labelsPath)


    def save_image(self):
        biggestIdx4TrainingData = Config.biggestIdx4TrainingData
        dataDirectory = Config.dataDirectory
        subFolder = 'train/' if self.frame_idx < biggestIdx4TrainingData else 'val/'
        imagesPath = dataDirectory + '/' + 'images/' + subFolder
        strIdxInFormat = format(self.frame_idx, '06d')
        filename = 'zebrafish_' + strIdxInFormat + '.png'
        imagesPath += filename
        success = cv.imwrite(imagesPath, self.finalViews[0][0])
            

    def save_video(self):
        biggestIdx4TrainingData = Config.biggestIdx4TrainingData
        dataDirectory = Config.dataDirectory
        subFolder = 'train/' if self.video_idx < biggestIdx4TrainingData else 'val/'
        videoFolder = dataDirectory + '/' + 'images/' + subFolder
        strIdxInFormat = format(self.video_idx, '06d')
        video_data_type = Config.outputVideoDataType
        if video_data_type == 'png':
            for  frame_idx in range(len(self.finalViews)):
                video_folder = 'zebrafish_' + strIdxInFormat
                if not os.path.exists(os.path.join(videoFolder, video_folder)):
                    os.makedirs(os.path.join(videoFolder, video_folder))
                filename = video_folder + '/im_' + str(frame_idx) + '.png'
                imagesPath = os.path.join(videoFolder, filename)
                success = cv.imwrite(imagesPath, self.finalViews[frame_idx][0][0])
                if not success:
                    print(f"Failed to save image: {imagesPath}")
        elif video_data_type == 'mp4':
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = 10.0
            frame_size = (imageSizeX, imageSizeY)
            filename = 'zebrafish_' + strIdxInFormat + '.mp4'
            videoPath = os.path.join(videoFolder, filename)
            out = cv.VideoWriter(videoPath, fourcc, fps, frame_size)
            if not out.isOpened():
                print("Error: Could not open video for writing.")
            else:
                print("Success: Video writer opened successfully.")
            for frame_idx in range(len(self.finalViews)):
                frame = np.uint8(self.finalViews[frame_idx][0][0])
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
                out.write(frame)
            out.release()


    # For Debbuging
    def get_image(self):
        return self.finalViews[0][0][0]

        
    def draw(self):
        # drawing the fishes
        for frame_id in range(len(self.fish_list)):
            for fish in self.fish_list[frame_id]:
                fish.draw()
                self.views_list[frame_id].append(fish.views)
            # merging the images
            if len(self.views_list[frame_id]) != 0:
                self.finalViews[frame_id] = mergeViews(self.views_list[frame_id])
            else:
                for viewIdx in range(self.amount_of_cameras):
                    view = (np.zeros((imageSizeY, imageSizeX)), np.zeros((imageSizeY, imageSizeX)))
                    self.finalViews[frame_id].append(view)
            # updating the visibility for the cases where a fish ends up covering another fish
            for fishIdx, fish in enumerate(self.fish_list[frame_id]):
                fish.update_visibility(self.finalViews[frame_id])
                # You have update the fish list, because python is weird
                self.fish_list[frame_id][fishIdx] = fish
        if self.shouldAddStaticNoise:
            self.add_static_noise_to_views()
        if self.shouldAddPatchyNoise:
            self.add_patchy_noise_to_views()


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

    def __init__(self, fishVect, fish_id=1, video_id=1):
        self.seglen = fishVect[0]
        # self.seglen = 2.4
        self.z = fishVect[1]
        self.x = fishVect[2:]
        self.fish_id = fish_id
        self.video_id = video_id
        self.random_params = self._generate_random_params()

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
    

    def _generate_random_params(self):
        """Generate all random parameters once for this fish"""
        np.random.seed(self.fish_id * (self.video_id + 1) * 1000)
        if Config.randomizeFish:
            return {
                # Eye randomization
                'rand1_eye': np.random.normal(1, 0.1),
                'rand2_eye': np.random.normal(1, 0.1), 
                'rand3_eye': np.random.normal(1, 0.1),
                
                # Belly randomization
                'rand1_belly': np.random.normal(1, 0.1),
                'rand2_belly': np.random.normal(1, 0.1),
                
                # Head randomization
                'rand1_head': np.random.normal(1, 0.1),
                'rand2_head': np.random.normal(1, 0.1),
                
                # Brightness multipliers
                'eyes_br_mult': np.random.normal(1, 0.1),
                'belly_br_mult': np.random.normal(1, 0.1),
                'head_br_mult': np.random.normal(1, 0.1),
                
                # Fish characteristics
                'd_eye_mult': np.random.normal(1, 0.1),
                'c_eyes_mult': np.random.normal(1, 0.1),
                'c_belly_mult': np.random.normal(1, 0.1),
                'c_head_mult': np.random.normal(1, 0.1),
                'random_number_size': np.random.normal(1., 0.1)
            }
        else:
            # No randomization - return 1.0 for all multipliers
            return {key: 1.0 for key in [
                'rand1_eye', 'rand2_eye', 'rand3_eye',
                'rand1_belly', 'rand2_belly', 'rand1_head', 'rand2_head',
                'eyes_br_mult', 'belly_br_mult', 'head_br_mult',
                'd_eye_mult', 'c_eyes_mult', 'c_belly_mult', 'c_head_mult',
                'random_number_size'
            ]}
    

    def draw(self):
        graymodel, pts = f_x_to_model_bigger(self.x, self.seglen, self.random_params, imageSizeX, imageSizeY)
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
