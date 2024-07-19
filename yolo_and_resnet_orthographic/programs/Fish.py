import numpy as np
from programs.Config import Config
from programs.programsForDrawingImage import f_x_to_model_bigger
from programs.Auxilary import createDepthArr
imageSizeY, imageSizeX = Config.imageSizeY, Config.imageSizeX

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

    def __init__(self, fishVect):
        self.seglen = fishVect[0]
        # self.seglen = 2.4
        self.z = fishVect[1]
        self.x = fishVect[2:]

    def draw(self):
        graymodel, pts = f_x_to_model_bigger(self.x, self.seglen, Config.randomizeFish, imageSizeX, imageSizeY)
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