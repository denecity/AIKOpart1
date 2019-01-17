import h5py
import numpy as np
import sys
import os


def write_head(fp, header, size):
    fp.write(np.int32(8))
    fp.write(header.encode())
    fp.write(np.int32(size+8))
    fp.write(np.int32(8))


def write_header(fl, attr):
    fl.write(np.int32(256))
    fl.write(np.array(attr['NumPart_ThisFile'][:6], dtype=np.int32))
    fl.write(np.array(attr['MassTable'][:6], dtype=np.float64))
    fl.write(np.asarray([attr['Time'], attr['Redshift']], dtype=np.float64))
    fl.write(np.array([1, 1], dtype=np.int32))
    fl.write(np.array(attr['NumPart_Total'][:6], dtype=np.int32))
    fl.write(np.array([1, attr['NumFilesPerSnapshot']], dtype=np.int32))
    fl.write(np.asarray([attr['BoxSize'], attr['Omega0'], attr['OmegaLambda'], attr['HubbleParam']], dtype=np.float64))
    fl.write(np.zeros(np.int32(256/4)-6-12-4-2-6-2-8, dtype=np.int32))
    fl.write(np.int32(256))


if len(sys.argv) != 2:
    print('I require only one argument -- the folder name!')
    print('I exit, because you providing is not fit!', str(sys.argv))
    exit(0)
else:
    folder = str(sys.argv[1])+"/"
print(folder, sys.argv[1])
h5files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# loop to get all required information
for i in h5files:
    if i[-4:] == 'hdf5' or i[-4:] == 'HDF5':
        f = h5py.File(folder+i, "r")
        #        of = open(folder+i[:-4]+"G3", 'wb')
        of = open(folder+i[:-5], 'wb')
        attrs = f['/Header'].attrs
        TNp = attrs['NumPart_ThisFile'][:6]
        TNs = np.sum(TNp)
        print(folder+i[:-5])
        # HEAD
        print("writing HEAD")
        write_head(of, "HEAD", 256)
        write_header(of, attrs)

        # POS
        print("writing POS")
        write_head(of, "POS ", np.uint32(TNs*4*3))
        of.write(np.uint32(TNs*4*3))
        of.write(np.float32(f['/PartType0/Coordinates'].value))
        of.write(np.float32(f['/PartType1/Coordinates'].value))
        of.write(np.float32(f['/PartType2/Coordinates'].value))
        of.write(np.float32(f['/PartType3/Coordinates'].value))
        if 'PartType4' in f.keys():
            of.write(np.float32(f['/PartType4/Coordinates'].value))
        if 'PartType5' in f.keys():
            of.write(np.float32(f['/PartType5/Coordinates'].value))
        of.write(np.uint32(TNs*4*3))
        print("writing MASS size:",np.uint32(TNs*4*3))

        # VEL
        print("writing VEL")
        write_head(of, "VEL ", np.uint32(TNs*4*3))
        of.write(np.uint32(TNs*4*3))
        of.write(np.float32(f['/PartType0/Velocities'].value))
        of.write(np.float32(f['/PartType1/Velocities'].value))
        of.write(np.float32(f['/PartType2/Velocities'].value))
        of.write(np.float32(f['/PartType3/Velocities'].value))
        if 'PartType4' in f.keys():
            of.write(np.float32(f['/PartType4/Velocities'].value))
        if 'PartType5' in f.keys():
            of.write(np.float32(f['/PartType5/Velocities'].value))
        of.write(np.uint32(TNs*4*3))

        # ID
        print("writing ID")
        write_head(of, "ID  ", np.uint32(TNs*4))
        of.write(np.uint32(TNs*4))
        of.write(np.uint32(f['/PartType0/ParticleIDs'].value))
        of.write(np.uint32(f['/PartType1/ParticleIDs'].value))
        of.write(np.uint32(f['/PartType2/ParticleIDs'].value))
        of.write(np.uint32(f['/PartType3/ParticleIDs'].value))
        if 'PartType4' in f.keys():
            of.write(np.uint32(f['/PartType4/ParticleIDs'].value))
        if 'PartType5' in f.keys():
            of.write(np.uint32(f['/PartType5/ParticleIDs'].value))
        of.write(np.uint32(TNs*4))
        print("writing ID size:",np.uint32(TNs*4))

        # MASS
        print("writing MASS")
        write_head(of, "MASS", np.uint32(TNs*4))
        of.write(np.uint32(TNs*4))
        of.write(np.float32(f['/PartType0/Masses'].value))
        of.write(np.float32(f['/PartType1/Masses'].value))
        of.write(np.float32(f['/PartType2/Masses'].value))
        of.write(np.float32(f['/PartType3/Masses'].value))
        if 'PartType4' in f.keys():
#            print("writing boundary MASS PartType4 ",len(np.float32(f['/PartType4/Masses'].value)))
            of.write(np.float32(f['/PartType4/Masses'].value))
        if 'PartType5' in f.keys():
#            print("writing boundary MASS PartType5 ",len(np.float32(f['/PartType5/Masses'].value)))
            of.write(np.float32(f['/PartType5/Masses'].value))
        of.write(np.uint32(TNs*4))
        print("writing MASS size:",np.uint32(TNs*4))

        # U <only gas>
        print("writing U")
        write_head(of, "U   ", np.uint32(TNp[0]*4))
        of.write(np.uint32(TNp[0]*4))
        of.write(np.float32(f['/PartType0/InternalEnergy'].value))
        print("writing U size:",np.uint32(TNp[0]*4))
        of.write(np.uint32(TNp[0]*4))

        # Rho <only gas>
        print("writing RHO")
        write_head(of, "RHO ", np.uint32(TNp[0]*4))
        of.write(np.uint32(TNp[0]*4))
        of.write(np.float32(f['/PartType0/Density'].value))
        print("writing RHO size:",np.uint32(TNp[0]*4))
        of.write(np.uint32(TNp[0]*4))
        
        # NE <only gas>
        print("writing NE")
        write_head(of, "NE  ", np.uint32(TNp[0]*4))
        of.write(np.uint32(TNp[0]*4))
        of.write(np.float32(f['/PartType0/ElectronAbundance'].value))
        of.write(np.uint32(TNp[0]*4))

        # NH <only gas>
        print("writing NH")
        write_head(of, "NH  ", np.uint32(TNp[0]*4))
        of.write(np.uint32(TNp[0]*4))
        of.write(np.float32(f['/PartType0/NeutralHydrogenAbundance'].value))
        of.write(np.uint32(TNp[0]*4))

        # HSML <only gas>  !!Note no hsml for Arepo
        print("writing HSML")
        #Calculate 
        dens = np.array(f[u'PartType0/Density'])
        mass = np.array(f[u'PartType0/Masses'])
        volume = np.divide(mass, dens)
        hsml  = np.power(3.*volume/(4.*np.pi), 1./3.)

        write_head(of, "HSML", np.uint32(TNp[0]*4))
        of.write(np.uint32(TNp[0]*4))
        of.write(hsml)
        #of.write(np.float32(f['/PartType0/AllowRefinement'].value))
        of.write(np.uint32(TNp[0]*4))

        # SFR <only gas>
        print("writing SFR")
        write_head(of, "SFR ", np.uint32(TNp[0]*4))
        of.write(np.uint32(TNp[0]*4))
        of.write(np.float32(f['/PartType0/StarFormationRate'].value))
        of.write(np.uint32(TNp[0]*4))

        # Z <gas + Star>
        print("writing Z")
        if 'PartType4' in f.keys():
            Tbp = TNp[0]+TNp[4]
        else:
            Tbp = TNp[0]
        write_head(of, "Z   ", np.uint32(Tbp*4))
        of.write(np.uint32(Tbp*4))
        of.write(np.float32(f['/PartType0/GFM_Metallicity'].value))
        if 'PartType4' in f.keys():
            of.write(np.float32(f['/PartType4/GFM_Metallicity'].value))
        of.write(np.uint32(Tbp*4))

        if 'PartType4' in f.keys():
            # AGE <only Star> stellar formation time !!No BHs
            write_head(of, "AGE ", np.uint32(TNp[4]*4))
            of.write(np.uint32(TNp[4]*4))
            of.write(np.float32(f['/PartType4/GFM_StellarFormationTime'].value))
            of.write(np.uint32(TNp[4]*4))
