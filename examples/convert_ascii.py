import glob
import numpy as np

from rdi import rdi_reader, rdi_transforms, rdi_writer


# look for many files 
filenames = glob.glob("/home/lucas/gliderdata/subex2016/adcp/PF*.PD0")
filenames.sort()
filenames = []
# failing that take just one, which is version controlled.
filenames = filenames or ["../data/PF230519.PD0"]


bindata = rdi_reader.PD0()

t1 = rdi_transforms.TransformENU_FSU()
t2 = rdi_transforms.TransformFSU_XYZ(hdg=0, pitch=0.1919, roll=0)
t3 = rdi_transforms.TransformXYZ_FSU(hdg=0, pitch=0.2239, roll=0.05)
# Set up the transformation pipeline. Note the order!
t4 = t3*t2*t1

writer = rdi_writer.AsciiWriter(filename = 'test.ascii')

ensembles = bindata.ensemble_generator(filenames)
ensembles = t4(ensembles)
writer.write_ensembles(ensembles)


# read and process the ensembles
_v = []
for i,ens in enumerate(ensembles):
    _v.append([ens['velocity']['Velocity%d'%(i+1)][0] for i in range(3)])
vx,vy,vz = np.array(_v).T
