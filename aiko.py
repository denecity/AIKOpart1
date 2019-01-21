#aiko package
#Denis Titov
#16.01.2019

"""
aiko gives read/write functionality for a .hdf5 database.
so far it only works on that specific database :/
*path*/snapdir_000/snapshot_000.0.hdf5
"""

import h5py
import numpy as np
import sys
import os
import pandas as pd
import re

class arepo_reader():
    #defines vars
    def __init__(self, path, snapshotRange, particleTypes=["Gas", "DM", "Darkmatter", "Stars"], blockNames=["Coordinates", "Velocities", "Density"], groupBlockNames=["Velocities", "CenterOfMass", "Radius", "Coordinates"]):
        if path[-1] is not "/":
            path = path + "/"
        self.path = path
        self.snapshots = np.arange(snapshotRange[0], snapshotRange[1]+1, 1)
        self.particleTypes = particleTypes
        self.blockNames = blockNames
        self.groupBlockNames = groupBlockNames

    #gets list of dirnames for each selected snapshot
    def get_path_raw(self, snapshot):
        Snapdir = "snapdir_" + "%03d" % snapshot + "/"
        path = self.path + Snapdir
        return path

    #
    def get_path_gal(self, snapshot):
        groupdir = "groups_" + "%03d" % snapshot + "/"
        path = self.path + groupdir
        return path

    # returns galaxy data in df
    def read_arepo_gal(self):
        dataList = []
        for snapshot in self.snapshots:
            currentPath = self.get_path_gal(snapshot)
            currentInstance = snapshot_read_gal(currentPath, blockNames=self.groupBlockNames)
            currentData = currentInstance.read_arepo_gal()
            dataList.append(currentData)
        print("processing...")
        dfAll = pd.concat(dataList, ignore_index=True)
        #dfAll.set_index("ID", inplace=True)
        print("done!")
        return dfAll

    # returns full list in df
    def read_arepo_raw(self):
        dataList = []
        for snapshot in self.snapshots:
            currentPath = self.get_path_raw(snapshot)
            currentInstance = snapshot_read_raw(currentPath, particleTypes=self.particleTypes, blockNames=self.blockNames)
            currentData = currentInstance.read_arepo_snap()
            dataList.append(currentData)
        print("processing...")
        dfAll = pd.concat(dataList, ignore_index=True)
        dfAll.set_index("ID", inplace=True)
        print("done!")
        return dfAll

#class for reading snapshot directories, managed by aiko
class snapshot_read_raw():
    #sets path of snapshot dir (corrects for missing slash at end of path) and other vars
    def __init__(self, snapPath, blockNames, particleTypes):
        if snapPath[-1] is not "/":
            snapPath = snapPath + "/"
        self.snapPath = snapPath
        self.blockNames = blockNames
        self.particleTypes = particleTypes
        self.particleTypeIndexes = self.get_particleType_index()
        self.numSnapdirs = self.get_numSnapdirs()
        self.currentSnapshot = self.get_snapshot()

    #searches for a file with correct name, opens and gets number of snaps from header
    def get_numSnapdirs(self):
        for i in os.listdir(self.snapPath):
            if re.search("\.\d\.", i):
                fileName = i
                break
        filePath = self.snapPath + fileName
        file = h5py.File(filePath, 'r')
        attrs = file['/Header'].attrs
        numFilesPerSnapshot = attrs["NumFilesPerSnapshot"]
        return numFilesPerSnapshot

    #converts words of particleTypes to particleTypeIndexes
    def get_particleType_index(self):
        indexList = []
        index = {"Gas":0, "DM":1, "Darkmatter":1, "Stars":4}
        for particleType in self.particleTypes:
            converted = index[particleType]
            indexList.append(converted)
        return indexList

    #returns current snapshot
    def get_snapshot(self):
        for i in os.listdir(self.snapPath):
            if re.search("\_\d{3}\.", i):
                snapshot = re.findall("\_\d{3}\.", i)[0][1:4]
                return int(snapshot)

    #returns list of all filenames in dir
    def get_snapdirList(self):
        return os.listdir(self.snapPath)

    #returns dataframe of snap-particledata
    def read_arepo_snap(self):
        dfSnapList = []
        for filename in self.get_snapdirList():
            file = h5py.File(self.snapPath +  filename, "r")
            dfFileList = []
            typeList = list(file)
            print("reading " + filename)
            for particleTypeIndex in self.particleTypeIndexes:
                typeStructure = "PartType{}".format(particleTypeIndex)
                if typeStructure not in typeList:
                    break
                dfStructure = {}
                for blockName in self.blockNames:
                    h5Path = "/PartType{}/{}".format(particleTypeIndex, blockName)
                    if blockName == "Coordinates":
                        coords = np.float32(file[h5Path].value)
                        dfStructure["posX"] = coords[:,0]
                        dfStructure["posY"] = coords[:,1]
                        dfStructure["posZ"] = coords[:,2]
                    if blockName == "Velocities":
                        vels = np.float32(file[h5Path].value)
                        dfStructure["velX"] = vels[:,0]
                        dfStructure["velY"] = vels[:,1]
                        dfStructure["velZ"] = vels[:,2]
                    if blockName == "Density" and particleTypeIndex == 0:
                        dens = np.float32(file[h5Path].value)
                        dfStructure["Dens"] = dens
                ID = np.uint32(file["/PartType{}/{}".format(particleTypeIndex, "ParticleIDs")].value)
                dfStructure["ID"] = ID
                dfStructure["Snap"] = ID*0+self.currentSnapshot
                dfStructure["Type"] = ID*0+particleTypeIndex
                dfCurrent = pd.DataFrame(dfStructure)
                dfFileList.append(dfCurrent)
            dfFile = pd.concat(dfFileList, sort=True)
            dfSnapList.append(dfFile)
        dfSnap = pd.concat(dfSnapList, ignore_index=True)
        return dfSnap

#class for reading snapshot galaxy data, managed by aiko
class snapshot_read_gal():
    #sets path of snapshot dir (corrects for missing slash at end of path) and other vars
    def __init__(self, groupPath, blockNames):
        if groupPath[-1] is not "/":
            groupPath = groupPath + "/"
        self.groupPath = groupPath
        self.blockNames = self.convert_blockNames(blockNames)
        self.numGroupdirs = self.get_numGroupdirs()
        self.currentSnapshot = self.get_snapshot()

    #searches for a file with correct name, opens and gets number of snaps from header
    def get_numGroupdirs(self):
        return len(os.listdir(self.groupPath))

    #returns current snapshot
    def get_snapshot(self):
        for i in os.listdir(self.groupPath):
            if re.search("\_\d{3}\.", i):
                snapshot = re.findall("\_\d{3}\.", i)[0][1:4]
                return int(snapshot)

    #returns list of all filenames in dir
    def get_groupdirList(self):
        return os.listdir(self.groupPath)

    #returns converted list of grouppath
    def convert_blockNames(self, blockNames):
        newBlock = []
        if "Velocities" in blockNames:
            newBlock.append("GroupVel")
        if "CenterOfMass" in blockNames:
            newBlock.append("GroupCM")
        if "Radius" in blockNames:
            newBlock.append("Group_R_Crit200")
        if "Coordinates" in blockNames:
            newBlock.append("GroupPos")
        return newBlock

    #returns dataframe of snap-particledata
    def read_arepo_gal(self):
        dfFileList = []
        for filename in self.get_groupdirList():
            file = h5py.File(self.groupPath +  filename, "r")
            print("reading " + filename)
            dfStructure = {}
            for blockName in self.blockNames:
                h5Path = "/Group/{}".format(blockName)
                if blockName == "GroupPos":
                    coords = np.float32(file[h5Path].value)
                    dfStructure["posX"] = coords[:,0]
                    dfStructure["posY"] = coords[:,1]
                    dfStructure["posZ"] = coords[:,2]
                if blockName == "GroupVel":
                    vels = np.float32(file[h5Path].value)
                    dfStructure["velX"] = vels[:,0]
                    dfStructure["velY"] = vels[:,1]
                    dfStructure["velZ"] = vels[:,2]
                if blockName == "GroupCM":
                    CM = np.float32(file[h5Path].value)
                    dfStructure["CMX"] = CM[:,0]
                    dfStructure["CMY"] = CM[:,1]
                    dfStructure["CMZ"] = CM[:,2]
                if blockName == "Group_R_Crit200":
                    radius = np.float32(file[h5Path].value)
                    dfStructure["Radius"] = radius
            #ID = np.uint32(file["/Group/{}".format(particleTypeIndex, "ParticleIDs")].value)
            dfStructure["Snap"] = radius*0+self.currentSnapshot
            dfCurrent = pd.DataFrame(dfStructure)
            dfFileList.append(dfCurrent)
        dfSnap = pd.concat(dfFileList, ignore_index=True)
        return dfSnap

"""
TODO:
fix ID in galaxy read
implement interpolation
implement plotting
"""