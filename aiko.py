#aiko package
#Denis Titov
#16.01.2019

"""
aiko gives read/write functionality for a .hdf5 database.
it can also generate graphics
"""

import h5py
import numpy as np
import sys
import os
import pandas as pd
import re
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob

class arepo_reader():
    #defines vars
    def __init__(self, path, snapshotRange, particleTypes=["Gas", "DM", "Stars"], blockNames=["Coordinates", "Velocities", "Density"], groupBlockNames=["Velocities", "CenterOfMass", "Radius", "Coordinates", "Mass"]):
        if path[-1] is not "/":
            path = path + "/"
        if len(snapshotRange) == 1:
            snapshotRange = [snapshotRange[0], snapshotRange[0], 1]
        elif len(snapshotRange) == 2:
            snapshotRange.append(1)
        self.path = path
        midSnap = np.array(np.arange(snapshotRange[0], snapshotRange[1]+1, snapshotRange[2]))
        if 127 not in midSnap:
            midSnap = np.append(midSnap, 127)
        self.snapshots = midSnap
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

    # returns galaxy data in df (main function)
    def read_arepo_gal(self):
        dataList = []
        print("reading...")
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

    # returns full list in df (main function)
    def read_arepo_raw(self):
        dataList = []
        print("reading...")
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
        return glob.glob(self.snapPath + "*hdf5")

    #returns dataframe of snap-particledata
    def read_arepo_snap(self):
        dfSnapList = []
        for filename in self.get_snapdirList():
            file = h5py.File(filename, "r")
            dfFileList = []
            typeList = list(file)
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
        return glob.glob(self.groupPath + "*hdf5")

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
        if "Mass" in blockNames:
            newBlock.append("GroupMass")
        return newBlock

    #returns dataframe of snap-particledata
    def read_arepo_gal(self):
        dfFileList = []
        for filename in self.get_groupdirList():
            file = h5py.File(filename, "r")
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
                if blockName == "GroupMass":
                    groupMass = np.float32(file[h5Path].value)
                    dfStructure["Mass"] = groupMass
            dfStructure["Snap"] = radius*0+self.currentSnapshot
            dfCurrent = pd.DataFrame(dfStructure)
            dfFileList.append(dfCurrent)
        dfSnap = pd.concat(dfFileList, ignore_index=True)
        return dfSnap




#normalisaton algorythm
def trrot(x,y):
    nres=len(x)
    xs=x[0]
    ys=y[0]
    xe=x[nres-1]
    ye=y[nres-1]
    xt=x-xs
    yt=y-ys
    Ra=(xt*xt+yt*yt)**0.5
    phi=np.rad2deg(np.arctan2(yt,xt))
    phie=phi[nres-1]
    phinew=phi-phie
    xt=Ra*np.cos(np.deg2rad(phinew))
    yt=Ra*np.sin(np.deg2rad(phinew))
    return xt,yt,Ra


def norm_select(particles, groups, location, seed, radius=0.01, samples=10): #22669 for 122-127
    galSelect = groups.loc[int(location)]  #location
    galPosX = galSelect["posX"]
    galPosY = galSelect["posY"]
    galPosZ = galSelect["posZ"]
    galRad = radius             #radius
    galSnap = galSelect["Snap"]
    galFiltered = (particles.Snap == galSnap)&\
                  (particles.posX<(galPosX+galRad))&(particles.posX>(galPosX-galRad))&\
                  (particles.posY<(galPosY+galRad))&(particles.posY>(galPosY-galRad))&\
                  (particles.posZ<(galPosZ+galRad))&(particles.posZ>(galPosZ-galRad))
    ind = particles[galFiltered].sample(samples, random_state=seed).copy(deep=True)
    ind.posX = (ind.posX - galPosX)
    ind.posY = (ind.posY - galPosY)
    ind.posZ = (ind.posZ - galPosZ)
    return ind


def orbit(particles, groups, samples=10, interpol="cubic", steps=10, location=None, radius=0.01, seed=None):
    if location == None:
        collection = groups[groups["Mass"] > 100]
        np.random.seed(seed)
        location = np.random.choice(collection.index)
    ind = norm_select(particles, groups, location, seed, radius, samples)
    ivec=ind.index.drop_duplicates()
    normList = []
    for i in range(len(ivec)):
        tCords = particles.loc[ivec[i]]
        tX = np.array(tCords["posX"])
        tY = np.array(tCords["posY"])
        if tX.size < 5:
            continue
        tList = trrot(tX, tY)
        tdf = pd.DataFrame({"posX":tList[0], "posY":tList[1], "Ra":tList[2]})
        normList.append(tdf)

    if interpol == False:
        line = normList[0]*0
        ax =line.plot(x="posX",y="posY", legend=False)
        for i in normList:
            i.plot(x="posX",y="posY",ax=ax, legend=False, grid=True)

    if interpol == "cubic" or interpol == "quadratic":
        line = normList[0]*0
        ax =line.plot(x="posX",y="posY", legend=False)
        for i in normList:
            x = i["posX"]
            y = i["posY"]
            xSmooth = np.linspace(np.min(x), np.max(x), len(x)*steps+1)
            fspline = scipy.interpolate.interp1d(x, y, interpol)
            ySmooth = fspline(xSmooth)
            ism = pd.DataFrame({"Xsmooth":xSmooth, "Ysmooth":ySmooth})
            ism.plot(x="Xsmooth", y="Ysmooth", ax=ax, legend=False, grid=True)

def map(particles, groups, snap=None, location=None, seed=None, kind="hexbin", zoom=0.01, density=False, gridsize=None, mass=50, cmap="RbBu", give_location=False):
    if snap == None:
        snap = int(particles["Snap"].drop_duplicates().sample(1, random_state=seed).values)
    if location == None:
        possibles = groups[groups["Snap"] == snap]
        collection = possibles[possibles["Mass"] > mass]
        np.random.seed(seed)
        location = np.random.choice(collection.index)

    galPosX = groups.loc[location].posX
    galPosY = groups.loc[location].posY
    galPosZ = groups.loc[location].posZ

    sel = (particles.Snap == snap)&\
          (particles.posX<(galPosX+zoom))&(particles.posX>(galPosX-zoom))&\
          (particles.posY<(galPosY+zoom))&(particles.posY>(galPosY-zoom))&\
          (particles.posZ<(galPosZ+zoom))&(particles.posZ>(galPosZ-zoom))

    if density == True:
        galDens = particles[sel]["Dens"]

    if kind == "hexbin":
        if gridsize == None:
            gridsize = 100
        particles[sel].plot(kind='hexbin',x="posX",y="posY", gridsize=gridsize, colormap="jet",norm=colors.LogNorm(),vmin=1)

    if kind == "scatter" and density == True:
        particles[sel].plot(kind='scatter',x="posX",y="posY", c=galDens, cmap=plt.cm.get_cmap(cmap),s=0.4, marker=".", norm=colors.LogNorm(), vmin=1).set_facecolor('black')

    if kind == "scatter" and density == False:
        particles[sel].plot(kind='scatter',x="posX",y="posY", c="w",s=0.2, marker=".").set_facecolor('black')

    if give_location == True:
        return location

"""
TODO:
position in space
turn off normalisation
3D
"""