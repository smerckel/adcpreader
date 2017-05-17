import glob
import numpy as np

from rdi import rdi_reader, rdi_transforms


# look for many files 
filenames = glob.glob("/home/lucas/gliderdata/subex2016/adcp/PF*.PD0")
filenames.sort()
filenames = []
# failing that take just one, which is version controlled.
filenames = filenames or ["../data/PF230519.PD0"]


bindata = rdi_reader.PD0()
ensembles = bindata.ensemble_generator(filenames)

t1 = rdi_transforms.TransformENU_FSU()
t2 = rdi_transforms.TransformFSU_XYZ(alpha=0, beta=0.1919, gamma=0)
t3 = rdi_transforms.TransformXYZ_FSU(alpha=0, beta=0.2239, gamma=0.05)

# Set up the transformation pipeline. Note the order!
t4 = t3*t2*t1

ensembles = t4(ensembles)

# read and process the ensembles
_v = []
for i,ens in enumerate(ensembles):
    _v.append([ens['velocity']['Velocity%d'%(i+1)][0] for i in range(3)])
vx,vy,vz = np.array(_v).T
