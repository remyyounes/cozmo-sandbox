import cv2
import numpy as np
import math
import helpers

#todo
#add support for multiple hands
#add classifier

def extractFingertips(defects, cnt, filterValue, left = None, right = None):
    left = False if left is None else left
    right = False if right is None else right
    if left and right:
        raise ValueError("'left' and 'right' cannot be both True")
    if str(type(left)) != "<class 'bool'>" or str(type(right)) != "<class 'bool'>":
        raise ValueError("left, right: boolean value expected")
    if str(type(filterValue)) != "<class 'int'>":
        raise ValueError("filterValue: integer value expected")
    fingertips = []
    if defects is None or cnt is None:
        return None
    points = processDefects(defects, cnt, startPoints = True)
    #print "bf", len(points)
    points = filterPoints(points, filterValue)
    #print "af", len(points)
    inversePoints = [point[::-1] for point in points]
    inversePoints.sort()
    for i in range(len(points)):
        fingertips.append(inversePoints[i][::-1])
        if (left or right) and i == 3:
            break
        if i == 4:
            break
    if left:
        fingertips.append(max(points))
    if right:
        fingertips.append(min(points))
    return fingertips

def dist(a, b):
    if str(type(a)) != "<class 'tuple'>" and str(type(a)) != "<class 'list'>":
        raise ValueError("lists or tuples expected")
    if str(type(b)) != "<class 'tuple'>" and str(type(b)) != "<class 'list'>":
        raise ValueError("lists or tuples expected")
    return math.sqrt((a[0] - b[0])**2 + (b[1] - a[1])**2)

def captureHistogram(source = None):
    source = 0 if source is None else source
    if source is not None and str(type(source)) != "<class 'int'>":
        raise ValueError("source: integer value expected")
    cap = cv2.VideoCapture(source)

    boxSize = 30
    centerX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    centerY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Place region of the hand inside the boxes and press `A`", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        box1 = (centerX + 0,     centerY + 0,   boxSize, boxSize)
        box2 = (centerX + 50,    centerY + 0,   boxSize, boxSize)
        box3 = (centerX + 0,     centerY + 50,  boxSize, boxSize)
        box4 = (centerX + 50,    centerY + 50,  boxSize, boxSize)

        helpers.drawBoxes(frame, helpers.PURPLE, [box1, box2, box3, box4])

        boxOne = helpers.crop(frame, box1)
        boxTwo = helpers.crop(frame, box2)
        boxThree = helpers.crop(frame, box3)
        boxFour = helpers.crop(frame, box4)

        finalHistImage = np.hstack((boxOne, boxTwo, boxThree, boxFour))
        cv2.imshow("Video Feed", frame)
        key = cv2.waitKey(10)
        if key == 97:
            objectColor = finalHistImage
            cv2.destroyAllWindows()
            break
        if key == 27:
            cv2.destroyAllWindows()
            cap.release()
            exit()

    hsvObjectColor = cv2.cvtColor(objectColor, cv2.COLOR_BGR2HSV)
    objectHist = cv2.calcHist([hsvObjectColor], [0,1], None, [12,15], [0,180,0,256])
    cv2.normalize(objectHist, objectHist, 0,255,cv2.NORM_MINMAX)
    return objectHist

def locateObject(frame, objectHist, BGR = None):
    BGR = False if BGR is None else BGR
    if BGR is not None and str(type(BGR)) != "<class 'bool'>":
        raise ValueError("BGR: boolean value expected")
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    objectSegment = cv2.calcBackProject([hsvFrame], [0,1], objectHist, [0,180,0,256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cv2.filter2D(objectSegment, -1, disc, objectSegment)
    _, threshObjectSegment = cv2.threshold(objectSegment,70,255,cv2.THRESH_BINARY)
    threshObjectSegment = cv2.merge((threshObjectSegment,threshObjectSegment,threshObjectSegment))
    locatedObject = cv2.bitwise_and(frame, threshObjectSegment)
    if BGR:
        return locatedObject
    locatedObjectGray = cv2.cvtColor(locatedObject, cv2.COLOR_BGR2GRAY)
    _, locatedObjectThresh = cv2.threshold(locatedObjectGray, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    locatedObject = cv2.medianBlur(locatedObjectThresh, 5)
    return locatedObject

def detectHand(frame, hist, sketchContours, computeDefects = None, colour = None, thickness = None):
    defects = None
    if str(type(sketchContours)) != "<class 'bool'>":
        raise ValueError("sketchContours: boolean value expected")
    if computeDefects is not None and str(type(computeDefects)) != "<class 'bool'>":
        raise ValueError("computeDefects: boolean value expected")
    if colour is not None and str(type(colour)) != "<class 'tuple'>" and str(type(colour)) != "<class 'list'>":
        raise ValueError("colour: list or tuple expected")
    if thickness is not None and str(type(thickness)) != "<class 'int'>":
        raise ValueError("thickness: integer value expected")
    frame = cv2.flip(frame, 1)
    computeDefects = False if computeDefects is None else computeDefects
    colour = (0,255,0) if colour is None else colour
    thickness = 2 if thickness is None else thickness
    detectedHand = locateObject(frame, hist)
    image, contours, _ = cv2.findContours(detectedHand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    palmArea = 0; flag = None
    for (i, c) in enumerate(contours):
        area = cv2.contourArea(c)
        if area > palmArea:
            palmArea = area
            flag = i
    if flag is not None:
        cnt = contours[flag]
        if computeDefects:
            defects = findDefects(cnt)
        if sketchContours:
            cv2.drawContours(frame, [cnt], 0, colour, thickness)
            return True, frame, cnt, defects
        else:
            return True, frame, cnt, defects
    else:
        return False, frame, None, defects

def findDefects(cnt):
    if cnt is None or cnt == []:
        return None
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)
    return defects

def filterPoints(points, filterValue):
    if points is None:
        return None
    if str(type(points)) != "<class 'list'>" and str(type(points)) != "<class 'tuple'>":
        raise ValueError("points: list or tuple expected")
    if str(type(filterValue)) != "<class 'int'>":
        raise ValueError("filterValue: integer value expected")
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if dist(points[i], points[j]) < filterValue:
                points[j] = (-1,-1)
    points.sort()
    while True:
        if points[0] == (-1, -1):
            del points[0]
        else:
            break
    return points

def plot(frame, points, radius = None, colour = None, thickness = None):
    if points is None or points == []:
        return frame
    if str(type(points)) != "<class 'list'>" and str(type(points)) != "<class 'tuple'>":
        raise ValueError("points: list or tuple expected")
    if radius is not None and str(type(radius)) != "<class 'int'>":
        raise ValueError("radius: integer value expected")
    if colour is not None and str(type(colour)) != "<class 'list'>" and str(type(colour)) != "<class 'tuple'>":
        raise ValueError("colour: list or tuple expected")
    if thickness is not None and str(type(thickness)) != "<class 'int'>":
        raise ValueError("points: list or tuple expected")
    radius = 5 if radius is None else radius
    colour = (0,0,255) if colour is None else colour
    thickness = -1 if thickness is None else thickness
    for point in points:
        cv2.circle(frame, point, radius, colour, thickness)
    return frame

def processDefects(defects, cnt, farPoints = None, startPoints = None, endPoints = None):
    farPoints = False if farPoints is None else farPoints
    endPoints = False if endPoints is None else endPoints
    startPoints = False if startPoints is None else startPoints
    if defects is None or cnt is None:
        return None
    if str(type(farPoints)) != "<class 'bool'>" or str(type(endPoints)) != "<class 'bool'>" or str(type(startPoints)) != "<class 'bool'>":
        raise ValueError("farPoints/endPoints/startPoints: boolean value expected")
    if farPoints and endPoints:
        raise ValueError("Only one can be computed in an instance")
    if farPoints and startPoints:
        raise ValueError("Only one can be computed in an instance")
    if startPoints and endPoints:
        raise ValueError("Only one can be computed in an instance")
    points = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if farPoints:
            points.append(far)
        if endPoints:
            points.append(end)
        if startPoints:
            points.append(start)
    return points
