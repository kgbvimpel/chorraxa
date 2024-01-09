from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from sklearn.linear_model import LinearRegression
import time

class CentroidTracker():
    def __init__(self, maxDisappeared=1, historySize=4):
        self.highestIDUsed = -1
        self.nextObjectID = 0
        self.usedIDs = set()  # Set to track all used IDs
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.historySize = historySize
        self.history = OrderedDict()
        self.lastUpdateTime = time.time()
        
    def register(self, centroid, bbox):
        # Always use one more than the highest ID ever used
        newID = self.highestIDUsed + 1
        self.objects[newID] = centroid
        self.bboxes[newID] = bbox
        self.disappeared[newID] = 0
        self.highestIDUsed = newID
        self.history[newID] = [(self.lastUpdateTime, centroid)]
        
    def deregister(self, objectID):
        # Deregister the object but do not remove the ID from usedIDs
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]
        del self.history[objectID]

    def is_inside_bbox(self, centroid, bbox):
        startX, startY, endX, endY = bbox
        return startX <= centroid[0] <= endX and startY <= centroid[1] <= endY
        
    def predict_next_positions(self):
        predictions = {}
        for objectID, history in self.history.items():
            if len(history) < 2:
                predictions[objectID] = history[-1][1]
                continue

            times, centroids = zip(*history)
            reg = LinearRegression()
            X = np.array(times).reshape(-1, 1)
            Y = np.array(centroids)
            reg.fit(X - X[0], Y)  # Normalize times for numerical stability
            next_position = reg.predict([[self.lastUpdateTime - times[0]]])[0]
            predictions[objectID] = next_position

        return predictions

    def update(self, rects):
        currentTime = time.time()
        timeElapsed = currentTime - self.lastUpdateTime
        self.lastUpdateTime = currentTime

        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += timeElapsed
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            predictions = self.predict_next_positions()

            usedRows = set()
            usedCols = set()

            for objectID, predictedCentroid in predictions.items():
                distances = []
                for (j, (newCentroid, newBbox)) in enumerate(zip(inputCentroids, rects)):
                    if j in usedCols:
                        continue

                    if self.is_inside_bbox(predictedCentroid, newBbox):
                        distance = dist.euclidean(predictedCentroid, newCentroid)
                        distances.append((distance, j))

                if distances:
                    distances.sort()
                    closestCol = distances[0][1]
                    self.objects[objectID] = inputCentroids[closestCol]
                    self.bboxes[objectID] = rects[closestCol]
                    self.history[objectID].append((currentTime, inputCentroids[closestCol]))
                    if len(self.history[objectID]) > self.historySize:
                        self.history[objectID].pop(0)
                    self.disappeared[objectID] = 0
                    usedRows.add(objectID)
                    usedCols.add(closestCol)

            unusedRows = set(self.objects.keys()).difference(usedRows)
            unusedCols = set(range(0, len(inputCentroids))).difference(usedCols)

            for row in unusedRows:
                self.disappeared[row] += timeElapsed
                if self.disappeared[row] > self.maxDisappeared:
                    self.deregister(row)

            for col in unusedCols:
                self.register(inputCentroids[col], rects[col])

        output = []
        for objectID, centroid in self.objects.items():
            bbox = self.bboxes[objectID]
            output.append((objectID, centroid, bbox))

        return output