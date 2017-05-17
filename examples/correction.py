import numpy as np
import gsw

import dbdreader
import rdi.rdi_reader
import rdi.rdi_corrections

# read in a PD0 file and extract unmodified velocity vectors
filenames = ["../data/PF230519.PD0"]
pd0 = rdi.rdi_reader.PD0()
ensembles = pd0.ensemble_generator(filenames)
# read and process the ensembles
_v = []
for i,ens in enumerate(ensembles):
    _v.append([ens['velocity']['Velocity%d'%(i+1)][0] for i in range(3)])
ve0,vn0,vz0 = np.array(_v).T


# now read the accompanying glider data to compute the salinity
c = rdi.rdi_corrections.SpeedOfSoundCorrection()

dbdreader.CACHEDIR = '../data'
dbds = dbdreader.MultiDBD(pattern="../data/comet*.[de]bd")
tmp = dbds.get_sync("sci_ctd41cp_timestamp",
                    "sci_water_cond sci_water_temp sci_water_pressure m_lat m_lon".split())
t, tctd, C, T, P, lat , lon = np.compress(tmp[2]>0, tmp, axis=1)
SP = gsw.SP_from_C(C*10, T, P*10)
SA = gsw.SA_from_SP_Baltic(SP, lon, lat)

# the generator is exhausted so, we need to set it again.
ensembles = pd0.ensemble_generator(filenames)
# pipe it into the current correction method
ensembles = c.horizontal_current_from_salinity_pressure(ensembles,
                                                        tctd, SA, P*10)

# read and process the ensembles
_v = []
for i,ens in enumerate(ensembles):
    _v.append([ens['velocity']['Velocity%d'%(i+1)][0] for i in range(3)])
ve,vn,vz = np.array(_v).T
