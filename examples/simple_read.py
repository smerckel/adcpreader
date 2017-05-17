import glob
import numpy as np

import rdi

# look for many files 
filenames = glob.glob("/home/lucas/gliderdata/subex2016/adcp/PF*.PD0")
filenames.sort()
# failing that take just one, which is version controlled.
filenames = filenames or ["../data/PF230519.PD0"]


bindata = rdi.rdi_reader.PD0()
ensembles = bindata.ensemble_generator(filenames)

# read and process the ensembles
_v = []
for i,ens in enumerate(ensembles):
    _v.append([ens['velocity']['Velocity%d'%(i+1)][0] for i in range(3)])
ve,vn,vz = np.array(_v).T
