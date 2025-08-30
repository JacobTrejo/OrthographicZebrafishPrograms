import yaml
import warnings
import os
from Programs.load_intrinsic_parameters import load_intrinsic_parameters

class Config:
    """
        Class that obtains all the variables from the configuration file
    """
    #   Default values

    # General Variables
    # 632, 1248 for Chie's videos
    # The following values are set only if they are not defined in Input/config.yaml
    imageSizeY = 890 #700
    imageSizeX = 1645 #1280
    randomizeFish = 1
    dataDirectory = 'dataset'
    amountOfData = 1
    fractionForTraining = .9
    shouldSaveImages = True
    shouldSaveAnnotations = True
    intrinsics_yaml_file = 'Programs/akihiro_intrinsic_parameters.yaml'
    
    # Rendering parameters
    # Zebrafish
    """
    c_eyes = 1.9 
    c_head = 1.04
    c_belly = 0.98
    d_eye = 1
    eyes_br = 235
    head_br = 0.64 # wrt belly
    belly_br = 0.83 # wrt eyes
    eye_w = 0.22
    eye_l = 0.35 
    eye_h = 0.3
    head_w = 0.3
    head_l = 0.86
    head_h = 0.53
    belly_w = 0.29
    belly_l = 0.86
    belly_h = 0.34
    """

    """
    # Danionella: Kristin's videos
    c_eyes = 3.0524
    c_head = 2.5001
    c_belly = 1.3485
    d_eye = 0.7676
    eyes_br = 368.9
    head_br = 0.51 # wrt belly
    belly_br = 0.57 # wrt eyes
    eye_w = 0.1650
    eye_l = 0.2375
    eye_h = 0.3
    head_w = 0.4575
    head_l = 0.7876
    head_h = 0.53
    belly_w = 0.4622
    belly_l = 1.2540
    belly_h = 0.34
    """

    # Danionella: Kristin's videos with wild type fish
    """
    c_eyes = 3.051
    c_head = 2.5001
    c_belly = 1.347
    d_eye = 0.65
    eyes_br = 360.5
    head_br = 0.51 # wrt belly
    belly_br = 0.57 # wrt eyes
    eye_w = 0.1852
    eye_l = 0.261
    eye_h = 0.3
    head_w = 0.563
    head_l = 0.885
    head_h = 0.34
    belly_w = 0.339
    belly_l = 1.467
    belly_h = 0.34
    """

    c_eyes, c_head, c_belly, d_eye, eyes_br, head_br, belly_br, eye_w, eye_l, eye_h, head_w, head_l, head_h, belly_w, belly_l, belly_h, seglen, ball_size, tail_thickness, tail_br = load_intrinsic_parameters(intrinsics_yaml_file)
    averageSizeOfFish = ((seglen -5.6) / 0.1 - 1)* 1.05 + 62
    
    # Danionella: Chie's videos (arena does not have index matched cage)
    """   
    c_eyes = 3.049
    c_head = 2.502
    c_belly = 1.35
    d_eye = 0.770
    eyes_br = 368.901
    eyes_br = 375 # Manually selected
    head_br = 0.510 # wrt belly
    belly_br = 0.573 # wrt eyes
    eye_w = 0.175
    eye_l = 0.248
    eye_h = 0.3
    head_w = 0.475
    head_l = 0.844
    head_l = 0.82 # Manually selected
    head_h = 0.53
    belly_w = 0.324
    belly_l = 1.31
    belly_h = 0.34
    """

    # Danionella: Chie's videos (arena has index matched cage)

    # Noise Variables
    shouldAddPatchyNoise = True
    shouldAddNoisySpots = True
    shouldAddStaticNoise = True
    averageAmountOfPatchyNoise = .3 # Rate of Poisson process
    averageAmountOfNoisySpots = 10 # Rate of Poisson process

    #   Variable relating to the distribution of the fish
    maxFishesInView = 5
    # The following variable is used as the lambda value for a poisson distribution
    averageFishInEdges = 0
    overlappingFishFrequency = .3
    # The following variable is the minimum distance 2 overlapping fishes will be
    maxOverlappingOffset = 10
    # List which controls how many overlaps a single fish will have based on a normal random variable
    # the values in the list are used as an upper bound on the amount
    # it starts from 0 overlaps, 1 overlap, 2 overlap ....
    # ex : for [.5, .95, 1.0] it implies a 50 % of the fish not having an overlap, 45% chance of an overlap with 1 fish
    #      and a 5% chance of an overlap with 2 fish
    overlapMarker = [.50, .95, 1.]
    constantFishesInArena = True
    #   Thresholds
    # This threshold is used when sequential keypoints of a fish have an x or y value that are about the same
    # and stops it from generating a box that captures that part of the fish
    minimumSizeOfBox = 3
    # The following variable is similar to the one above except that it is for the bounding box of the fish passed
    # to Yolo.  This is necessary because sometimes the fish are barely visible at the edge causing the model to
    # learn to detect the edges as fish
    boundingBoxThreshold = 2
    # Value in which the brightness of a fish is considered solid
    visibilityThreshold = 25

    # None for now since it is going to get set after checking the yaml file
    biggestIdx4TrainingData = None

    # TODO: try setting this to a static method to make it more natural
    def __init__(self, pathToYamlFile):
        """
            Essentially just a function to update the variables accordingly
        """
        static_vars = list(vars(Config))[2:-3]

        file = open(pathToYamlFile, 'r')
        config = yaml.safe_load(file)
        keys = config.keys()
        list_of_vars_in_config = list(keys)

        # Updating the static variables
        for var in list_of_vars_in_config:
            if var in static_vars:
                value = config[var]
                line = 'Config.' + var + ' = '

                if not isinstance(value, str):
                    line += str(value)
                else:
                    line += "'" + value + "'"
                exec(line)
            else:
                warnings.warn(var + ' is not a valid variable, could be a spelling issue')

        Config.biggestIdx4TrainingData = Config.amountOfData * Config.fractionForTraining
        Config.dataDirectory += '/'

        # NOTE: the following was just left as an example for now
        # # Writing the variables to the corresponding classes static variables
        # Config.set_aquarium_vars()
        # Config.set_bounding_box_vars()
    # @staticmethod
    # def set_bounding_box_vars():
    #     print('setting the bounding box vars')

# Setting the variables of the Configuration Class
Config('Inputs/config.yaml')
