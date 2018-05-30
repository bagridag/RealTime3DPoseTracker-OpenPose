import pandas as pd

import os
import numpy as np

cwd = os.getcwd()

pathProcessed = cwd + "/newTrain/"

pathNew = cwd + "/trainData/"

def processData():

    trialTime = input("Enter the trial time for cleaning:")
    fileNamePunch = "keyPointsPersonRightHandPunch{0}.csv".format(trialTime)
    fileNameWave = "keyPointsPersonRightHandWave{0}.csv".format(trialTime)
    fileNameShoot = "keyPointsPersonRightHandShoot{0}.csv".format(trialTime)
    fileNameStill = "keyPointsPersonRightHandStill{0}.csv".format(trialTime)
    fileNamePunchLabels = "labelsPersonRightHandPunch{0}.csv".format(trialTime)
    fileNameWaveLabels = "labelsPersonRightHandWave{0}.csv".format(trialTime)
    fileNameShootLabels = "labelsPersonRightHandShoot{0}.csv".format(trialTime)
    fileNameStillLabels = "labelsPersonRightHandStill{0}.csv".format(trialTime)

    # READ THE LEFT HAND TRAINING DATA

    fileNamePunchL = "keyPointsPersonLeftHandPunch{0}.csv".format(trialTime)
    fileNameWaveL = "keyPointsPersonLeftHandWave{0}.csv".format(trialTime)
    fileNameShootL = "keyPointsPersonLeftHandShoot{0}.csv".format(trialTime)
    fileNameStillL = "keyPointsPersonLeftHandStill{0}.csv".format(trialTime)
    fileNamePunchLabelsL = "labelsPersonLeftHandPunch{0}.csv".format(trialTime)
    fileNameWaveLabelsL = "labelsPersonLeftHandWave{0}.csv".format(trialTime)
    fileNameShootLabelsL = "labelsPersonLeftHandShoot{0}.csv".format(trialTime)
    fileNameStillLabelsL = "labelsPersonLeftHandStill{0}.csv".format(trialTime)


    dataPunch = pd.read_csv(pathNew + fileNamePunch, header=None)
    colLength= dataPunch.shape[1]
    dataPunch= dataPunch.replace(0,np.nan)
    dataPunch = dataPunch.dropna(thresh=52, axis=0) #drop the row if more than 7 joint info is missing
    dataPunch= dataPunch.fillna(0)


    dataPunch.to_csv(pathProcessed + fileNamePunch, sep= ',', index= False, header = None)

    dataWave = pd.read_csv(pathNew + fileNameWave, header=None)
    dataWave= dataWave.replace(0,np.nan)
    dataWave = dataWave.dropna(thresh= 52, axis=0)
    dataWave= dataWave.fillna(0)


    dataWave.to_csv(pathProcessed + fileNameWave, sep= ',', index= False, header=None)

    dataShoot = pd.read_csv(pathNew + fileNameShoot, header= None)
    dataShoot= dataShoot.replace(0,np.nan)
    dataShoot = dataShoot.dropna(thresh = 52, axis=0)
    dataShoot= dataShoot.fillna(0)
    dataShoot.to_csv(pathProcessed + fileNameShoot, sep= ',', index= False, header=None)

    dataStill = pd.read_csv(pathNew + fileNameStill, header= None)
    #dataStill= dataStill.replace(0,np.nan)
    #dataStill = dataStill.dropna(thresh = 52, axis=0)
    #dataStill= dataStill.fillna(0)
    dataStill.to_csv(pathProcessed + fileNameStill, sep= ',', index= False, header=None)

    nRowsPunch = dataPunch.shape[0]
    nRowsWave = dataWave.shape[0]
    nRowsShoot = dataShoot.shape[0]
    nRowsStill = dataStill.shape[0]

    labelsPunch = pd.read_csv(pathNew + fileNamePunchLabels, header=None)
    labelsPunch = labelsPunch[0:nRowsPunch]
    labelsPunch.to_csv(pathProcessed + fileNamePunchLabels, sep= ',', index= False, header=None)

    labelsWave = pd.read_csv(pathNew + fileNameWaveLabels, header=None)
    labelsWave = labelsWave[0:nRowsWave]
    labelsWave.to_csv(pathProcessed + fileNameWaveLabels, sep= ',', index= False, header=None)


    labelsShoot = pd.read_csv(pathNew + fileNameShootLabels, header=None)
    labelsShoot = labelsShoot[0:nRowsShoot]
    labelsShoot.to_csv(pathProcessed + fileNameShootLabels, sep= ',', index= False, header=None)

    labelsStill = pd.read_csv(pathNew + fileNameStillLabels, header=None)
    labelsStill = labelsStill[0:nRowsStill]
    labelsStill.to_csv(pathProcessed + fileNameStillLabels, sep= ',', index= False, header=None)



    ###LEFT HAND

    dataPunchL = pd.read_csv(pathNew + fileNamePunchL, header=None)
    colLengthL= dataPunchL.shape[1]
    dataPunchL= dataPunchL.replace(0,np.nan)
    dataPunchL = dataPunchL.dropna(thresh=52, axis=0) #drop the row if more than 7 joint info is missing
    dataPunchL= dataPunchL.fillna(0)
    dataPunchL.to_csv(pathProcessed + fileNamePunchL, sep= ',', index= False, header = None)



    dataWaveL = pd.read_csv(pathNew + fileNameWaveL, header=None)
    dataWaveL= dataWaveL.replace(0,np.nan)
    dataWaveL = dataWaveL.dropna(thresh= 52, axis=0)
    dataWaveL= dataWaveL.fillna(0)
    dataWaveL.to_csv(pathProcessed + fileNameWaveL, sep= ',', index= False, header=None)

    dataShootL = pd.read_csv(pathNew + fileNameShootL, header= None)
    dataShootL= dataShootL.replace(0,np.nan)
    dataShootL = dataShootL.dropna(thresh = 52, axis=0)
    dataShootL= dataShootL.fillna(0)
    dataShootL.to_csv(pathProcessed + fileNameShootL, sep= ',', index= False, header=None)

    dataStillL = pd.read_csv(pathNew + fileNameStillL, header= None)
    #dataStill= dataStill.replace(0,np.nan)
    #dataStill = dataStill.dropna(thresh = 52, axis=0)
    #dataStill= dataStill.fillna(0)
    dataStillL.to_csv(pathProcessed + fileNameStillL, sep= ',', index= False, header=None)

    nRowsPunchL = dataPunchL.shape[0]
    nRowsWaveL = dataWaveL.shape[0]
    nRowsShootL = dataShootL.shape[0]
    nRowsStillL = dataStillL.shape[0]

    labelsPunchL = pd.read_csv(pathNew + fileNamePunchLabelsL, header=None)
    labelsPunchL = labelsPunchL[0:nRowsPunchL]
    labelsPunchL.to_csv(pathProcessed + fileNamePunchLabelsL, sep= ',', index= False, header=None)

    labelsWaveL = pd.read_csv(pathNew + fileNameWaveLabelsL, header=None)
    labelsWaveL = labelsWaveL[0:nRowsWaveL]
    labelsWaveL.to_csv(pathProcessed + fileNameWaveLabelsL, sep= ',', index= False, header=None)


    labelsShootL = pd.read_csv(pathNew + fileNameShootLabelsL, header=None)
    labelsShootL = labelsShootL[0:nRowsShootL]
    labelsShootL.to_csv(pathProcessed + fileNameShootLabelsL, sep= ',', index= False, header=None)

    labelsStillL = pd.read_csv(pathNew + fileNameStillLabelsL, header=None)
    labelsStillL = labelsStillL[0:nRowsStillL]
    labelsStillL.to_csv(pathProcessed + fileNameStillLabelsL, sep= ',', index= False, header=None)


