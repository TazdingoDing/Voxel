import numpy as np
from scipy.spatial import cKDTree
class Voxel():
    #size of each voxel
    size = None
    #raw input numpy array (n * m)
    rawData = None
    #translated numpy array (3 * n), minimal value = 0
    __translatedData = None
    #min, max and range of x, y and z
    shapes = None
    #range of x, y and z
    ranges = None
    #voxelized input data (numpy array xRange * yRange * zRange)
    voxels = None
    #a list of indexes of rawData
    __pointsList = None
    #KDtree
    dataKDTree = None
    def __init__(self, rawData, size = 1, withKDTree = False):
        self.rawData = rawData
        self.size = size
        self.__getNewShapes()
        self.voxels = np.zeros(self.ranges)
        self.__pointsList = self.__makeList(self.voxels.size)
        self.__loadDataIndexToPointsList()
        self.__loadPointsListToVoxels(threshold = 0)
        if withKDTree:
            self.dataKDTree = cKDTree(rawData[:,0:3], leafsize=30)
    def __getNewShapes(self):
        data = np.copy(self.rawData[:,0:3].T)
        maxXYZ = [np.amax(data[i]) for i in range(3)]
        minXYZ = [np.amin(data[i]) for i in range(3)]
        rangeXYZ = [int(maxXYZ[i]/self.size) - int(minXYZ[i]/self.size) + 1 for i in range(3)]
        
        self.ranges = rangeXYZ
        self.__translatedData = np.array([data[i] - minXYZ[i] for i in range(3)])
        self.shapes = {"xMax": maxXYZ[0], "yMax": maxXYZ[1], "zMax": maxXYZ[2],
                      "xMin": minXYZ[0], "yMin": minXYZ[1], "zMin": minXYZ[2], 
                      "x_range": rangeXYZ[0], "y_range": rangeXYZ[1], "z_range": rangeXYZ[2]}
        
    def __makeList(self, length):
        return [None] * length
    def __getPointsListIndexFromIJK(self, i, j, k):
        y_r = self.ranges[1]
        z_r = self.ranges[2]
        return i * y_r * z_r + j * z_r + k

    def __getIJKFromPointsListIndex(self, index):
        y_r = self.ranges[1]
        z_r = self.ranges[2]
        i, tmp = divmod(index, y_r * z_r)
        j, k = divmod(tmp, z_r)

        return i, j, k
    
    def __loadDataIndexToPointsList(self):
        scaledData = self.__translatedData / self.size
        
        #shape[1]: number of points
        for i in range(self.__translatedData.shape[1]):
            I, J, K = [int(scaledData[j][i]) for j in range(3)]
            #print([scaledData[j][i] for j in range(3)])
            index = self.__getPointsListIndexFromIJK(I, J, K)
            if not self.__pointsList[index]:
                self.__pointsList[index] = list()
            self.__pointsList[index].append(i)
    
    def __loadPointsListToVoxels(self, threshold = 3):
        for vi in range(len(self.__pointsList)):
            if self.__pointsList[vi]:
                pointsInside = len(self.__pointsList[vi])
                if (pointsInside > threshold):
                    i, j, k = self.__getIJKFromPointsListIndex(vi)
                    self.voxels[i][j][k] = pointsInside
    def getArrayOfXYZInsideIJK(self, i, j, k):
        return np.array(self.getListOfXYZInsideIJK(i, j, k))
    def getListOfXYZInsideIJK(self, i, j, k):
        pointsListIndex = self.__getPointsListIndexFromIJK(i, j, k)
        pointsIndex = self.__pointsList[pointsListIndex]
        tmp = []
        if (pointsIndex):
            for q in range(len(pointsIndex)):
                tmp.append(list(self.rawData[pointsIndex[q]][0:3]))
        return tmp
    def getArrayOfPointsInsideIJK(self, i, j, k):
        return np.array(self.getListOfPointsInsideIJK(i, j, k))
    def getListOfPointsInsideIJK(self, i, j, k):
        pointsListIndex = self.__getPointsListIndexFromIJK(i, j, k)
        pointsIndex = self.__pointsList[pointsListIndex]
        tmp = []
        if (pointsIndex):
            for q in range(len(pointsIndex)):
                tmp.append(list(self.rawData[pointsIndex[q]]))
        return tmp
    
    def getArrayOfXYZWithinRadius(self, x, y, z, r):
        if self.dataKDTree:
            neighborIndex = self.dataKDTree.query_ball_point([x, y, z], r)
            return self.rawData[neighborIndex, 0:3]
        else:
            raise #didn't build with KDTree option!
    def getListOfXYZWithinRadius(self, x, y, z, r):
        return self.getArrayOfXYZWithinRadius(x, y, z, r).tolist()
    def getArrayOfPointsWithinRadius(self, x, y, z, r):
        if self.dataKDTree:
            neighborIndex = self.dataKDTree.query_ball_point([x, y, z], r)
            return self.rawData[neighborIndex]
        else:
            raise #didn't build with KDTree option!
    def getListOfPointsWithinRadius(self, x, y, z, r):
        return self.getArrayOfPointsWithinRadius(x, y, z, r).tolist()
    
    def getIJKFromXYZ(self, x, y, z):
        xMin = self.shapes["xMin"]
        yMin = self.shapes["yMin"]
        zMin = self.shapes["zMin"]
        return (int((x-xMin)/self.size),int((y-yMin)/self.size),int((z-zMin)/self.size))

    def getRawDataIndexFromXYZ(self, x, y, z):
        i, j, k = self.getIJKFromXYZ(x, y, z)
        pointsListIndex = self.__getPointsListIndexFromIJK(i, j, k)
        XYZ = np.array([x, y, z])
        if self.__pointsList[pointsListIndex]:
            for t in self.__pointsList[pointsListIndex]:
                if np.allclose(XYZ, self.rawData[t][0:3], rtol = 1e-08, atol=1e-10):
                    return t
            return None
        else:
            return None
