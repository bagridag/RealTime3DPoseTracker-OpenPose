
import sys
import xlsxwriter
import pyrealsense2 as rs
import csv
import time

from sklearn import preprocessing

sys.path.append('/home/burcak/Desktop/PyOpenPose/build/PyOpenPoseLib')
sys.path.append('/home/burcak/Desktop/libfreenect2/build/lib')
import PyOpenPose as OP
import time
import cv2
import numpy as np
import math
import os

cwd= os.getcwd()
pathNew = cwd + "/trainData/"

if not os.path.exists(pathNew):
    os.makedirs(pathNew)

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

WIDTH = 640
HEIGHT = 480



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



#GET 3D POSITIONS OUT OF OPENPOSE

def get3DPos(pose, alignedArr, x, isBody):
    if isBody:
        persons = pose[0][x] #if the array is a Pose array, then get the first person
    else:
        persons= pose[x] #if the array belongs to hands it has no person ID, so just read it.

    hSz = len(persons)
    persons3D = np.zeros((hSz, 4), np.float32)
    for c in range(0, hSz):
        xCoor1 = math.ceil(persons[c][0])
        xCoor2 = xCoor1 - 1
        yCoor1 = math.ceil(persons[c][1])
        yCoor2 = yCoor1 - 1


        #if the coordinates are out of range, continue(for the edges)
        if xCoor1>=WIDTH or xCoor2>=WIDTH or yCoor1>=HEIGHT or yCoor2>=HEIGHT:
            continue

        if xCoor2 == -1 or yCoor2 ==-1:
            zCoor = alignedArr[yCoor1][xCoor1]
        else:
            zCoor1 = alignedArr[yCoor1][xCoor1]
            zCoor2 = alignedArr[yCoor2][xCoor2]
            zCoor = (zCoor1 + zCoor2) / 2


        persons3D[c][0] = persons[c][0]
        persons3D[c][1] = persons[c][1]
        if np.isinf(zCoor):
            persons3D[c][2] = 0
        else:
            persons3D[c][2] = zCoor
        persons3D[c][3] = persons[c][2]
    return persons3D

def get3DPosWorld(pose, x, isBody, verticeList):
    if isBody:
        persons = pose[0][x] #if the array is a Pose array, then get the first person
    else:
        persons= pose[x] #if the array belongs to hands it has no person ID, so just read it.

    hSz = len(persons)
    persons3DWorld = np.zeros((hSz, 3), np.float32)
    for c in range(0, hSz):
        xCoor1 = math.ceil(persons[c][0])
        xCoor2 = xCoor1 - 1
        yCoor1 = math.ceil(persons[c][1])
        yCoor2 = yCoor1 - 1

        # if the coordinates are out of range, continue(for the edges)
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
        #if the confidentiality is smaller than 0.2, write 0 as the keypoints
        #check the range of the values
        if(persons[c][2] < 0.2):
            persons3DWorld[c][0] = 0.0
            persons3DWorld[c][1] = 0.0
            persons3DWorld[c][2] = 0.0
        else:
            persons3DWorld[c][0] = xCoorWorld
            persons3DWorld[c][1] = yCoorWorld
            persons3DWorld[c][2] = zCoorWorld
        #don't include the confidentiality score in the final data- if you want to include uncomment
        #persons3DWorld[c][3] = persons[c][2]
    return persons3DWorld




def run():
    startProcess = False
    with_face = False #enable face if you want to detect face
    with_hands = True
    download_heatmaps = False
    op = OP.OpenPose((320, 240), (368, 368), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                     download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
    # op = OP.OpenPose((320, 240), (240, 240), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, download_heatmaps)
    actual_fps = 0

    paused = False
    delay = {True: 0, False: 1}

    print("Entering main Loop for TRAINING.")

    #GET THE USER INPUTS
    gestureID = input("Enter the gesture you want to store: PUNCH(1) OR WAVE(2) OR SHOOT(3) OR (4) for STILL..:")
    trialTime = input("Enter the training document number:")
    handNumber= input("Enter the hand to to train: RIGHT(1) AND LEFT(2): ")
    timeToTrain = float(input("Enter the amount of time(seconds) that you want to train the data, 20-30 seconds are suggested:"))
    startTrigger = input("Enter (S) to start ..:")

    if startTrigger == 'S' or 's':
        startProcess = True

    print("Starting in 3 seconds...")

    time.sleep(3)

    start = time.time()
    timeLeft = timeToTrain

    # Streaming loop
    while startProcess:
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
            points = pc.calculate(aligned_depth_frame)

            vtx = np.asanyarray(points.get_vertices()) #create a vertex list from the point cloud

            lstPersons3d = []
            lstRightHand3d = []
            lstLeftHand3d = []
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

        res = op.render(color_image)
        cv2.putText(res, 'UI FPS = %f, OP FPS = %f, timeLeft = %f' % (actual_fps, op_fps, timeLeft), (20, 20), 0, 0.5, (0, 0, 255))

        print("Open Pose FPS: ", op_fps)
        print("Actual Pose FPS: ", actual_fps)

        np.set_printoptions(suppress=True)


        if pose[0] is not None:
            #comment out for multi person
            #numberPersons= len(pose[0])
            #for x in range(0,numberPersons):
                #lstPersons3d.append(get3DPos(pose,depth_image, x, True))m
            lstPersons3dRealWorld.extend(get3DPosWorld(pose, 0, True, vtx))
            normalizedTrainPose = preprocessing.Normalizer().fit_transform(np.asarray(lstPersons3dRealWorld).reshape(1, -1))

            if rightTemp is not None and int(handNumber) == 1:
                #comment out for multi person
                #numberRightHands = len(rightTemp)
                #for y in range(0, numberRightHands):
                #lstRightHand3d.append(get3DPos(rightTemp,depth_image, y, False))
                lstHandRight3dRealWorld.extend(get3DPosWorld(rightTemp, 0, False, vtx))

                #add relbow and rshoulder and rwrist, and the joint btw shoulders(BODY JOINT NUMBER 1)
                lstHandRight3dRealWorld.append(np.asarray([lstPersons3dRealWorld[2][0],lstPersons3dRealWorld[2][1], lstPersons3dRealWorld[2][2]]))
                lstHandRight3dRealWorld.append(np.asarray([lstPersons3dRealWorld[3][0], lstPersons3dRealWorld[3][1], lstPersons3dRealWorld[3][2]]))
                lstHandRight3dRealWorld.append(np.asarray([lstPersons3dRealWorld[4][0], lstPersons3dRealWorld[4][1], lstPersons3dRealWorld[4][2]]))
                lstHandRight3dRealWorld.append(np.asarray([lstPersons3dRealWorld[1][0], lstPersons3dRealWorld[1][1], lstPersons3dRealWorld[1][2]]))

                #GET THE RELATIVE POSITIONS OF THE JOINTS ACORDINGLY TO THE MIDDLE OF THE SHOULDERS JOINT(BODY JOINT NUMBER 1)
                lenA = len(lstHandRight3dRealWorld)
                for i in range(0, lenA):
                    lstHandRight3dRealWorld[i][0] = lstHandRight3dRealWorld[i][0] - lstHandRight3dRealWorld[24][0]
                    lstHandRight3dRealWorld[i][1] = lstHandRight3dRealWorld[i][1] - lstHandRight3dRealWorld[24][1]
                    lstHandRight3dRealWorld[i][2] = lstHandRight3dRealWorld[i][2] - lstHandRight3dRealWorld[24][2]
                # REMOVE JOINT #1 FROM THE JOINT LIST FOR BETTER RESULTS IN TRAINING
                lstHandRight3dRealWorld = lstHandRight3dRealWorld[0:24]
                normalizedTrainRightHand = preprocessing.Normalizer().fit_transform(np.asarray(lstHandRight3dRealWorld).reshape(1, -1))

            if leftTemp is not None and int(handNumber) == 2:
                #comment out for multi person
                #numberLeftHands = len(leftTemp)
                #for z in range(0, numberLeftHands):
                    #lstLeftHand3d.append(get3DPos(leftTemp,depth_image, z, False))
                lstHandLeft3dRealWorld.extend(get3DPosWorld(leftTemp, 0, False, vtx))
                #add lelbow and lshoulder and lwrist and the joint btw shoulders(BODY JOINT NUMBER 1)
                lstHandLeft3dRealWorld.append(np.asarray([lstPersons3dRealWorld[5][0], lstPersons3dRealWorld[5][1], lstPersons3dRealWorld[5][2]]))
                lstHandLeft3dRealWorld.append(np.asarray([lstPersons3dRealWorld[6][0], lstPersons3dRealWorld[6][1], lstPersons3dRealWorld[6][2]]))
                lstHandLeft3dRealWorld.append(np.asarray([lstPersons3dRealWorld[7][0], lstPersons3dRealWorld[7][1], lstPersons3dRealWorld[7][2]]))
                lstHandLeft3dRealWorld.append(np.asarray([lstPersons3dRealWorld[1][0], lstPersons3dRealWorld[1][1], lstPersons3dRealWorld[1][2]]))

                # GET THE RELATIVE POSITIONS OF THE JOINTS ACORDINGLY TO THE MIDDLE OF THE SHOULDERS JOINT(BODY JOINT NUMBER 1)
                lenA = len(lstHandLeft3dRealWorld)
                for i in range(0, lenA):
                    lstHandLeft3dRealWorld[i][0] = lstHandLeft3dRealWorld[i][0] - lstHandLeft3dRealWorld[24][0]
                    lstHandLeft3dRealWorld[i][1] = lstHandLeft3dRealWorld[i][1] - lstHandLeft3dRealWorld[24][1]
                    lstHandLeft3dRealWorld[i][2] = lstHandLeft3dRealWorld[i][2] - lstHandLeft3dRealWorld[24][2]
                #REMOVE JOINT #1 FROM THE JOINT LIST FOR BETTER RESULTS IN TRAINING
                lstHandLeft3dRealWorld = lstHandLeft3dRealWorld[0:24]
                normalizedTrainLeftHand = preprocessing.Normalizer().fit_transform(np.asarray(lstHandLeft3dRealWorld).reshape(1, -1))

        #fieldnames = ['X', 'Y', 'Z', 'LABEL']
        if lstPersons3dRealWorld: #just write the pose keypoints once
            #comment out for multi person
            #for personIndex in range(0, numberPersons):

            ##comment out to enable storing pose keypoints
            #if gestureID == '1':
                #fileNamePose = "keyPointsPersonPosePunch{0}.csv".format(personIndex)
                #labelsPose = "labelsPersonPosePunch{0}.csv".format(personIndex)
                #fileNamePose = "keyPointsPersonPosePunch{0}.csv".format(trialTime)
                #labelsPose = "labelsPersonPosePunch{0}.csv".format(trialTime)
            #elif gestureID == '2':
                #fileNamePose = "keyPointsPoseWave{0}.csv".format(personIndex)
                #labelsPose = "labelsPoseWave{0}.csv".format(personIndex)
                #fileNamePose = "keyPointsPersonPoseWave{0}.csv".format(trialTime)
                #labelsPose = "labelsPersonPoseWave{0}.csv".format(trialTime)
            #elif gestureID == '3':
                #fileNamePose = "keyPointsPersonPoseShoot{0}.csv".format(trialTime)
                #labelsPose = "labelsPersonPoseShoot{0}.csv".format(trialTime)
            #elif gestureID == '4':
                #fileNamePose = "keyPointsPersonPoseStill{0}.csv".format(trialTime)
                #labelsPose = "labelsPersonPoseStill{0}.csv".format(trialTime)
                #fileNamePose = "keyPointsPoseOk{0}.csv".format(personIndex)
                #labelsPose = "labelsPoseOk{0}.csv".format(personIndex)

            #poseFile = open(pathNew+fileNamePose, 'a+')
            #poseLabels = open(pathNew + labelsPose, 'a+')

            #with poseFile and poseLabels:
                #writerPose = csv.writer(poseFile)
                #writerPoseLabels = csv.writer(poseLabels)
                #writerPoseLabels.writerow(gestureID)
                #writerPose.writerows(normalizedTrainPose)

            #GET DATA FOR THE RIGHT HAND
            if lstHandRight3dRealWorld and int(handNumber) == 1:
                if gestureID == '1':
                    fileNameRightHand = "keyPointsPersonRightHandPunch{0}.csv".format(trialTime)
                    labelsRightHand = "labelsPersonRightHandPunch{0}.csv".format(trialTime)
                elif gestureID == '2':
                    fileNameRightHand = "keyPointsPersonRightHandWave{0}.csv".format(trialTime)
                    labelsRightHand = "labelsPersonRightHandWave{0}.csv".format(trialTime)
                elif gestureID == '3':
                    fileNameRightHand = "keyPointsPersonRightHandShoot{0}.csv".format(trialTime)
                    labelsRightHand = "labelsPersonRightHandShoot{0}.csv".format(trialTime)
                elif gestureID == '4':
                    fileNameRightHand = "keyPointsPersonRightHandStill{0}.csv".format(trialTime)
                    labelsRightHand = "labelsPersonRightHandStill{0}.csv".format(trialTime)

                rightHandFile = open(pathNew+fileNameRightHand, 'a+')
                rightHandLabels = open(pathNew+labelsRightHand, 'a+')
                with rightHandFile and rightHandLabels:
                    writerRightHand= csv.writer(rightHandFile)
                    writerRightHandLabels = csv.writer(rightHandLabels)
                    writerRightHandLabels.writerow(gestureID)
                    writerRightHand.writerows(normalizedTrainRightHand)
            # GET DATA FOR THE LEFT HAND
            if lstHandLeft3dRealWorld and int(handNumber) == 2:
                if gestureID == '1':
                    fileNameLeftHand = "keyPointsPersonLeftHandPunch{0}.csv".format(trialTime)
                    labelsLeftHand = "labelsPersonLeftHandPunch{0}.csv".format(trialTime)
                elif gestureID == '2':
                    fileNameLeftHand = "keyPointsPersonLeftHandWave{0}.csv".format(trialTime)
                    labelsLeftHand = "labelsPersonLeftHandWave{0}.csv".format(trialTime)
                elif gestureID == '3':
                    fileNameLeftHand = "keyPointsPersonLeftHandShoot{0}.csv".format(trialTime)
                    labelsLeftHand = "labelsPersonLeftHandShoot{0}.csv".format(trialTime)
                elif gestureID == '4':
                    fileNameLeftHand = "keyPointsPersonLeftHandStill{0}.csv".format(trialTime)
                    labelsLeftHand = "labelsPersonLeftHandStill{0}.csv".format(trialTime)


                leftHandFile = open(pathNew+fileNameLeftHand, 'a+')
                leftHandLabels = open(pathNew+labelsLeftHand, 'a+')
                with leftHandFile and leftHandLabels:
                    writerLeftHand= csv.writer(leftHandFile)
                    writerLeftHandLabels = csv.writer(leftHandLabels)
                    writerLeftHandLabels.writerow(gestureID)
                    writerLeftHand.writerows(normalizedTrainLeftHand)

        timeLeft = timeToTrain - (time.time() - start)
        if timeLeft < 0:
            pipeline.stop()
            break


        actual_fps = 1.0 / (time.time() - start_time)
        cv2.imshow("OpenPose result", res)


        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            pipeline.stop()
            break
if __name__ == '__main__':
    run()