
import sys
import pyrealsense2 as rs
import csv

import pandasAnalysis
import pandas as pd

sys.path.append('/home/burcak/Desktop/PyOpenPose/build/PyOpenPoseLib')
sys.path.append('/home/burcak/Desktop/libfreenect2/build/lib')

import PyOpenPose as OP
import time
import cv2
import numpy as np
import math
import os
import itertools

from sklearn import svm
from sklearn import preprocessing, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



cwd= os.getcwd()

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

WIDTH = 640
HEIGHT = 480

#set the variables for data analysis

##RIGHT HAND
xPunchR= []
xWaveR=[]
xShootR= []
xStillR= []

lTrainR = []
labelsR = []
xTrainR= []

##LEFT HAND
xPunchL= []
xWaveL=[]
xShootL= []
xStillL= []

lTrainL = []
labelsL = []
xTrainL= []

path = cwd + "/newTrain/"

pathNew = cwd + "/trainData/"

pathProc= cwd + "/processedTrain/"

trainAmount = input("Enter the trial time for training:")

pandasAnalysis.processData()

for i in range(0, int(trainAmount)):

    #READ THE RIGHT HAND TRAINING DATA
    fileNamePunchR = "keyPointsPersonRightHandPunch{0}.csv".format(i)
    fileNameWaveR = "keyPointsPersonRightHandWave{0}.csv".format(i)
    fileNameShootR = "keyPointsPersonRightHandShoot{0}.csv".format(i)
    fileNameStillR = "keyPointsPersonRightHandStill{0}.csv".format(i)
    fileNamePunchLabelsR = "labelsPersonRightHandPunch{0}.csv".format(i)
    fileNameWaveLabelsR = "labelsPersonRightHandWave{0}.csv".format(i)
    fileNameShootLabelsR = "labelsPersonRightHandShoot{0}.csv".format(i)
    fileNameStillLabelsR = "labelsPersonRightHandStill{0}.csv".format(i)


    punchR = open(path + fileNamePunchR, 'r')
    waveR = open(path + fileNameWaveR, 'r')
    shootR= open(path + fileNameShootR, 'r')
    stillR= open(path + fileNameStillR, 'r')
    punchLabelsR = open(path + fileNamePunchLabelsR, 'r')
    waveLabelsR = open(path + fileNameWaveLabelsR, 'r')
    shootLabelsR = open(path + fileNameShootLabelsR, 'r')
    stillLabelsR = open(path + fileNameStillLabelsR, 'r')

    #READ THE LEFT HAND TRAINING DATA

    fileNamePunchL = "keyPointsPersonLeftHandPunch{0}.csv".format(i)
    fileNameWaveL = "keyPointsPersonLeftHandWave{0}.csv".format(i)
    fileNameShootL = "keyPointsPersonLeftHandShoot{0}.csv".format(i)
    fileNameStillL = "keyPointsPersonLeftHandStill{0}.csv".format(i)
    fileNamePunchLabelsL= "labelsPersonLeftHandPunch{0}.csv".format(i)
    fileNameWaveLabelsL = "labelsPersonLeftHandWave{0}.csv".format(i)
    fileNameShootLabelsL = "labelsPersonLeftHandShoot{0}.csv".format(i)
    fileNameStillLabelsL = "labelsPersonLeftHandStill{0}.csv".format(i)


    punchL = open(path + fileNamePunchL, 'r')
    waveL = open(path + fileNameWaveL, 'r')
    shootL= open(path + fileNameShootL, 'r')
    stillL= open(path + fileNameStillL, 'r')
    punchLabelsL = open(path + fileNamePunchLabelsL, 'r')
    waveLabelsL = open(path + fileNameWaveLabelsL, 'r')
    shootLabelsL = open(path + fileNameShootLabelsL, 'r')
    stillLabelsL = open(path + fileNameStillLabelsL, 'r')

    print("reading RIGHT training data")
    with punchR:
        reader = csv.reader(punchR, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            xTrainPunchR = []
            for i in range(0,len(row)):
                xTrainPunchR.append(float(row[i]))
                xy=np.asarray(xTrainPunchR)
            xPunchR.append(xy)
            xTrainR.append(xy)


    with waveR:
        reader = csv.reader(waveR, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            xTrainWaveR = []
            for i in range(0,len(row)):
                xTrainWaveR.append(float(row[i]))
                xx= np.asarray(xTrainWaveR)
            xWaveR.append(xx)
            xTrainR.append(xx)

    with shootR:
        reader = csv.reader(shootR, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            xTrainShootR = []
            for i in range(0,len(row)):
                xTrainShootR.append(float(row[i]))
                xz= np.asarray(xTrainShootR)
            xShootR.append(xz)
            xTrainR.append(xz)

    with stillR:
        reader = csv.reader(stillR, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            xTrainStillR = []
            for i in range(0,len(row)):
                xTrainStillR.append(float(row[i]))
                xn= np.asarray(xTrainStillR)
            xStillR.append(xn)
            xTrainR.append(xn)

    with punchLabelsR:
        reader = csv.reader(punchLabelsR, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            labelsR.append(float(row[0]))

    with waveLabelsR:
        reader = csv.reader(waveLabelsR, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            labelsR.append(float(row[0]))

    with shootLabelsR:
        reader = csv.reader(shootLabelsR, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            labelsR.append(float(row[0]))

    with stillLabelsR:
        reader = csv.reader(stillLabelsR, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            labelsR.append(float(row[0]))

    lTrainR= np.asarray(labelsR)

    print("reading LEFT training data")
    with punchL:
        reader = csv.reader(punchL, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            xTrainPunchL = []
            for i in range(0,len(row)):
                xTrainPunchL.append(float(row[i]))
                xy=np.asarray(xTrainPunchL)
            xPunchL.append(xy)
            xTrainL.append(xy)

    with waveL:
        reader = csv.reader(waveL, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            xTrainWaveL = []
            for i in range(0,len(row)):
                xTrainWaveL.append(float(row[i]))
                xx= np.asarray(xTrainWaveL)
            xWaveL.append(xx)
            xTrainL.append(xx)

    with shootL:
        reader = csv.reader(shootL, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            xTrainShootL = []
            for i in range(0,len(row)):
                xTrainShootL.append(float(row[i]))
                xz= np.asarray(xTrainShootL)
            xShootL.append(xz)
            xTrainL.append(xz)

    with stillL:
        reader = csv.reader(stillL, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            xTrainStillL = []
            for i in range(0,len(row)):
                xTrainStillL.append(float(row[i]))
                xn= np.asarray(xTrainStillL)
            xStillL.append(xn)
            xTrainL.append(xn)

    with punchLabelsL:
        reader = csv.reader(punchLabelsL, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            labelsL.append(float(row[0]))

    with waveLabelsL:
        reader = csv.reader(waveLabelsL, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            labelsL.append(float(row[0]))

    with shootLabelsL:
        reader = csv.reader(shootLabelsL, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            labelsL.append(float(row[0]))

    with stillLabelsL:
        reader = csv.reader(stillLabelsL, delimiter=',')
        for _ in range(5):  # skip the first 5 rows
            next(reader)
        for row in reader:
            labelsL.append(float(row[0]))

    lTrainL= np.asarray(labelsL)

print("Finished reading the training data")

#COMMENT OUT IF YOU WANT TO PLOT THE CONFUSION MATRIX
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''

#DEFINE THE CLASSIFIERS TO COMPARE

##RIGHT HAND

kNeighborsCLF = KNeighborsClassifier(n_neighbors=30)

linearSVCCLF= svm.LinearSVC(penalty='l2', C=3000,multi_class='ovr', max_iter=5000)

svcCLF = svm.SVC(kernel='rbf', C=2000, probability=True)

modelLinearRegressionCLF = LinearRegression(normalize=False)

linearLogisticRegressionCLF = linear_model.LogisticRegression(C=10)

decisionTreeCLF = DecisionTreeClassifier(random_state=10)




# Split into a training set and a test set using a stratified k fold

# split into a training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    xTrainR, lTrainR, test_size=0.25, random_state=42)


print("Fitting the classifier to the training set")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

classifier = clf.fit(xTrainR, lTrainR)


#COMMENT OUT IF YOU WANT TO ENABLE CONFUSION MATRIX
# #############################################################################
'''
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["Punch", "Wave", "Shoot", "Stand_Still"]))

print(confusion_matrix(y_test, y_pred))

cnf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["Punch", "Wave", "Shoot", "Stand_Still"],title='Confusion matrix, without normalization')

plt.show()


# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["Punch", "Wave", "Shoot", "Stand_Still"], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
'''


##########################################################################

##LEFT HAND
'''

kNeighborsCLF_Left = KNeighborsClassifier(n_neighbors=30)

linearSVCCLF_Left= svm.LinearSVC(penalty='l2', C=3000,multi_class='ovr', max_iter=5000)

svcCLF_Left = svm.SVC(kernel='rbf', C=2000, probability=True)

modelLinearRegressionCLF_Left = LinearRegression(normalize=False)

linearLogisticRegressionCLF_Left = linear_model.LogisticRegression(C=10)

decisionTreeCLF_Left = DecisionTreeClassifier(random_state=10)

'''

#CHECK OUTLIERS IN THE DATA USING BOXPLOT
'''
plt.boxplot(xTrain, notch=True, vert=True)

plt.boxplot(xWave, notch=True, vert=True)

plt.boxplot(xShoot, notch= True, vert= True)
'''


print("Fitting to the model- Right Hand ")

svcCLF.fit(xTrainR, lTrainR) #train the svm
decisionTreeCLF.fit(xTrainR, lTrainR) #train the decision tree
kNeighborsCLF.fit(xTrainR, lTrainR) #train the kNN

linearLogisticRegressionCLF.fit(xTrainR, lTrainR)

linearSVCCLF.fit(xTrainR,lTrainR)

modelLinearRegressionCLF.fit(xTrainR, lTrainR)

print("Fitting to the model- Left Hand ")

#svcCLF_Left.fit(xTrainL, lTrainL) #train the svm
#decisionTreeCLF_Left.fit(xTrainL, lTrainL) #train the decision tree
#kNeighborsCLF_Left.fit(xTrainL, lTrainL) #train the kNN

#linearLogisticRegressionCLF_Left.fit(xTrainL, lTrainL)

#linearSVCCLF_Left.fit(xTrainL,lTrainL)

#modelLinearRegressionCLF_Left.fit(xTrainL, lTrainL)

###################################
###COMMENT OUT TO ENABLE THE PCA PLOT
'''
#principal component analysis
pca = PCA(n_components=3)
proj = pca.fit_transform(xTrainR)
plt.scatter(proj[:, 0], proj[:, 1], c=lTrainR)
plt.colorbar()
proj = pca.fit_transform(xTrainL)
plt.scatter(proj[:, 0], proj[:, 1], c=lTrainL)
plt.colorbar()
'''
#################################

scoresLinearSVC = cross_val_score(linearSVCCLF, xTrainR, lTrainR, cv=5)

print(scoresLinearSVC)

print("Accuracy Linear SVC: %0.2f (+/- %0.2f)" % (scoresLinearSVC.mean(), scoresLinearSVC.std() * 2))

##############################################
scoresSVM = cross_val_score(svcCLF, xTrainR, lTrainR, cv=5)

print(scoresSVM)

print("Accuracy SVM: %0.2f (+/- %0.2f)" % (scoresSVM.mean(), scoresSVM.std() * 2))

################################################

scoresKNeighbors = cross_val_score(kNeighborsCLF, xTrainR, lTrainR, cv=5)

print(scoresKNeighbors)

print("Accuracy Nearest Neighbors: %0.2f (+/- %0.2f)" % (scoresKNeighbors.mean(), scoresKNeighbors.std() * 2))

###############################################
scoresLinearRegression = cross_val_score(modelLinearRegressionCLF, xTrainR, lTrainR, cv=5)

print(scoresLinearRegression)

print("Accuracy Linear Regression: %0.2f (+/- %0.2f)" % (scoresLinearRegression.mean(), scoresLinearRegression.std() * 2))

#############################################

scoresLinearLogisticRegression = cross_val_score(linearLogisticRegressionCLF, xTrainR, lTrainR, cv=5)

print(scoresLinearLogisticRegression)

print("Accuracy Linear LOgistig REgression: %0.2f (+/- %0.2f)" % (scoresLinearLogisticRegression.mean(), scoresLinearLogisticRegression.std() * 2))

#############################################

scoresDecisionTree= cross_val_score(decisionTreeCLF, xTrainR, lTrainR, cv=5)

print(scoresDecisionTree)

print("Accuracy Decision Tree: %0.2f (+/- %0.2f)" % (scoresDecisionTree.mean(), scoresDecisionTree.std() * 2))
############################################


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 10  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()

# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

rightData = []

def get3DPosWorld(pose, x, isBody, verticeList):
    if isBody:
        persons = pose[0][x]
    else:
        persons= pose[x]

    hSz = len(persons)
    persons3DWorld = np.zeros((hSz, 3), np.float32)
    for c in range(0, hSz):
        xCoor1 = math.ceil(persons[c][0])
        xCoor2 = xCoor1 - 1
        yCoor1 = math.ceil(persons[c][1])
        yCoor2 = yCoor1 - 1

        if xCoor1>=WIDTH or xCoor2>=WIDTH or yCoor1>=HEIGHT or yCoor2>=HEIGHT:
            continue

#UPDATE THE Z COORDINATE FROM THE VERTICE LIST ACCORDINGLY
        if xCoor2 == -1 or yCoor2 ==-1:
            verticeIndex = yCoor1*WIDTH + xCoor1
            xCoorWorld = verticeList[verticeIndex][0]
            yCoorWorld = verticeList[verticeIndex][1]
            zCoorWorld = verticeList[verticeIndex][2]
        else:
            verticeIndex1 = yCoor1*WIDTH + xCoor1
            verticeIndex2 = yCoor2*WIDTH + xCoor2
            xCoorWorld1= verticeList[verticeIndex1][0]
            yCoorWorld1= verticeList[verticeIndex1][1]
            zCoorWorld1= verticeList[verticeIndex1][2]

            xCoorWorld2= verticeList[verticeIndex2][0]
            yCoorWorld2= verticeList[verticeIndex2][1]
            zCoorWorld2= verticeList[verticeIndex2][2]

            zCoorWorld = (zCoorWorld1 + zCoorWorld2) / 2
            yCoorWorld = (yCoorWorld1 + yCoorWorld2) / 2
            xCoorWorld = (xCoorWorld1 + xCoorWorld2) / 2

        #check the range of the values
        persons3DWorld[c][0] = xCoorWorld
        persons3DWorld[c][1] = yCoorWorld
        persons3DWorld[c][2] = zCoorWorld

        rightData.append(xCoorWorld)
        rightData.append(yCoorWorld)
        rightData.append(zCoorWorld)
    return persons3DWorld

def run():
    with_face = False
    with_hands = True
    download_heatmaps = False
    # with_face = with_hands = False
    #op = OP.OpenPose((320, 240), (368, 368), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                     #download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
    op = OP.OpenPose((320, 240), (368, 368), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, download_heatmaps)
    actual_fps = 0
    #numberWave = 0
    #numberPunch = 0
    paused = False
    delay = {True: 0, False: 1}

    print("Entering main Loop.")

# Streaming loop
    while True:
        start_time = time.time()
        try:

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            pc.map_to(color_frame)

            #visual = pcl.pcl_visualization.CloudViewing()
            points = pc.calculate(aligned_depth_frame)

            vtx = np.asanyarray(points.get_vertices())
            #dfVTX = pd.DataFrame(data=vtx)
            #dfVTX.to_csv(cwd + "PointCloud.csv", sep=',', index=False, header=None)
            lstPersons3dRealWorld = []
            lstHandRight3dRealWorld = []
            lstHandLeft3dRealWorld = []

        except Exception as e:
            print("Failed to grab", e)
            break

        t = time.time()
        op.detectPose(color_image)
        #op.detectFace(rgb)
        op.detectHands(color_image)
        t = time.time() - t
        op_fps = 1.0 / t

        pose = op.getKeypoints(op.KeypointType.POSE)
        leftTemp = op.getKeypoints(op.KeypointType.HAND)[0]
        rightTemp = op.getKeypoints(op.KeypointType.HAND)[1]




        print("Open Pose FPS: ", op_fps)
        print("Actual Pose FPS: ", actual_fps)

        np.set_printoptions(suppress=True)

        res = op.render(color_image)

        if pose[0] is not None:
            # comment out for enabling for multi person
            #numberPersons= len(pose[0])
            #for x in range(0,numberPersons):
                #lstPersons3dRealWorld.append(get3DPosWorld(pose, x, True, vtx))
            lstPersons3dRealWorld.extend(get3DPosWorld(pose, 0, True, vtx))
            normalizedTrainPose = preprocessing.Normalizer().fit_transform(np.asarray(lstPersons3dRealWorld).reshape(1, -1))

            if rightTemp is not None:
                # comment out for enabling for multi person
                numberRightHands = len(rightTemp)
                #for y in range(0, numberRightHands):
                # lstHandRight3dRealWorld.append(get3DPosWorld(rightTemp, y, False, vtx))
                lstHandRight3dRealWorld.extend(get3DPosWorld(rightTemp, 0, False, vtx))

                # add relbow and rshoulder and rwrist and the joint btw shoulders(BODY JOINT NUMBER 1)
                lstHandRight3dRealWorld.append(np.asarray([lstPersons3dRealWorld[2][0], lstPersons3dRealWorld[2][1], lstPersons3dRealWorld[2][2]]))
                lstHandRight3dRealWorld.append(np.asarray([lstPersons3dRealWorld[3][0], lstPersons3dRealWorld[3][1], lstPersons3dRealWorld[3][2]]))
                lstHandRight3dRealWorld.append(np.asarray([lstPersons3dRealWorld[4][0], lstPersons3dRealWorld[4][1], lstPersons3dRealWorld[4][2]]))
                lstHandRight3dRealWorld.append(np.asarray([lstPersons3dRealWorld[1][0], lstPersons3dRealWorld[1][1], lstPersons3dRealWorld[1][2]]))

                # GET THE RELATIVE POSITIONS OF THE JOINTS ACORDINGLY TO THE MIDDLE OF THE SHOULDERS JOINT(BODY JOINT NUMBER 1)
                lenA= len(lstHandRight3dRealWorld)
                for i in range(0, lenA):
                    lstHandRight3dRealWorld[i][0] = lstHandRight3dRealWorld[i][0] - lstHandRight3dRealWorld[24][0]
                    lstHandRight3dRealWorld[i][1] = lstHandRight3dRealWorld[i][1] - lstHandRight3dRealWorld[24][1]
                    lstHandRight3dRealWorld[i][2] = lstHandRight3dRealWorld[i][2] - lstHandRight3dRealWorld[24][2]

                # REMOVE JOINT #1 FROM THE JOINT LIST FOR BETTER RESULTS IN TRAINING
                lstHandRight3dRealWorld = lstHandRight3dRealWorld[0:24]
                normalizedTrainRightHand = preprocessing.Normalizer().fit_transform(np.asarray(lstHandRight3dRealWorld).reshape(1, -1))

            if leftTemp is not None:
                # comment out for enabling for multi person
                #numberLeftHands = len(leftTemp)
                #for z in range(0, numberLeftHands):
                #lstHandLeft3dRealWorld.append(get3DPosWorld(leftTemp, z, False, vtx))
                lstHandLeft3dRealWorld.extend(get3DPosWorld(leftTemp, 0, False, vtx))

                # add lelbow and lshoulder and lwrist and the joint btw shoulders(BODY JOINT NUMBER 1)
                lstHandLeft3dRealWorld.append(np.asarray([lstPersons3dRealWorld[5][0], lstPersons3dRealWorld[5][1], lstPersons3dRealWorld[5][2]]))
                lstHandLeft3dRealWorld.append(np.asarray([lstPersons3dRealWorld[6][0], lstPersons3dRealWorld[6][1], lstPersons3dRealWorld[6][2]]))
                lstHandLeft3dRealWorld.append(np.asarray([lstPersons3dRealWorld[7][0], lstPersons3dRealWorld[7][1], lstPersons3dRealWorld[7][2]]))
                lstHandLeft3dRealWorld.append(np.asarray([lstPersons3dRealWorld[1][0], lstPersons3dRealWorld[1][1], lstPersons3dRealWorld[1][2]]))

                # GET THE RELATIVE POSITIONS OF THE JOINTS ACORDINGLY TO THE MIDDLE OF THE SHOULDERS JOINT(BODY JOINT NUMBER 1)
                lenA = len(lstHandLeft3dRealWorld)
                for i in range(0, lenA):
                    lstHandLeft3dRealWorld[i][0] = -(lstHandLeft3dRealWorld[i][0] - lstHandLeft3dRealWorld[24][0])
                    lstHandLeft3dRealWorld[i][1] = lstHandLeft3dRealWorld[i][1] - lstHandLeft3dRealWorld[24][1]
                    lstHandLeft3dRealWorld[i][2] = lstHandLeft3dRealWorld[i][2] - lstHandLeft3dRealWorld[24][2]
                # REMOVE JOINT #1 FROM THE JOINT LIST FOR BETTER RESULTS IN TRAINING
                lstHandLeft3dRealWorld = lstHandLeft3dRealWorld[0:24]
                normalizedTrainLeftHand = preprocessing.Normalizer().fit_transform(np.asarray(lstHandLeft3dRealWorld).reshape(1, -1))


            if rightTemp.max() > 0 or leftTemp.max() > 0 :
                try:
                    #resultsvcCLF_Right = svcCLF.predict(normalizedTrainRightHand)[0]  # svm result

                    ##the other prediction models are just made for testing- can be used if wanted
                    #resultGestureTree_Right = decisionTreeCLF.predict(normalizedTrainRightHand)[0]
                    #resultKnn_Right = kNeighborsCLF.predict(normalizedTrainRightHand)
                    #resultLinearRegressionCLF_Right = linearLogisticRegressionCLF.predict(normalizedTrainRightHand)
                    #resultLinearRegression_Right = modelLinearRegressionCLF.predict(normalizedTrainRightHand)
                    #resultLinearSVC_Right = linearSVCCLF.predict(normalizedTrainRightHand)
                    #rr= svcCLF.decision_function(normalizedTrainRightHand)
                    #nn = kNeighborsCLF.decision_function(normalizedTrainRightHand)
                    #mm= linearSVCCLF.decision_function(normalizedTrainRightHand)

                    resultsvcCLF_Right = classifier.predict(normalizedTrainRightHand)[0]  # BEST FITTER RESULT

                    resultsvcCLF_Left = classifier.predict(normalizedTrainLeftHand)[0]  # BEST FITTER RESULT

                    if resultsvcCLF_Right == 1:
                        rightGestureName = "Punch"
                    elif resultsvcCLF_Right == 2:
                        rightGestureName = "Wave"
                    elif resultsvcCLF_Right == 3:
                        rightGestureName = "Shoot"
                    elif resultsvcCLF_Right == 4:
                        rightGestureName = "NoGesture"

                    if resultsvcCLF_Left == 1:
                        leftGestureName = "Punch"
                    elif resultsvcCLF_Left == 2:
                        leftGestureName = "Wave"
                    elif resultsvcCLF_Left == 3:
                        leftGestureName = "Shoot"
                    elif resultsvcCLF_Left == 4:
                        leftGestureName = "NoGesture"

                    cv2.putText(res, 'UI FPS = %f, OP FPS = %f, RightGESTURE= %s, LeftGESTURE= %s' % (
                        actual_fps, op_fps, rightGestureName, leftGestureName), (20, 20), 0, 0.4, (0, 0, 255))
                    cv2.imshow("OpenPose result", res)

                except IndexError:
                    print("This class has no label")

            else:
                cv2.putText(res, 'UI FPS = %f, OP FPS = %f, Gesture= No gesture' % (
                    actual_fps, op_fps), (20, 20), 0, 0.5, (0, 0, 255))
                cv2.imshow("OpenPose result", res)
        else:
            cv2.putText(color_image, 'No skeleton detected', (20, 20), 0, 0.5, (0, 0, 255))
            cv2.imshow("OpenPose result", color_image)

        actual_fps = 1.0 / (time.time() - start_time)
        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            pipeline.stop()
            break
if __name__ == '__main__':
    run()