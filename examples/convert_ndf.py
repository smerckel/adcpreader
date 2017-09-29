import numpy as np
import gsw

import rdi.rdi_writer as rdi_ndf
import dbdreader
import ndf



if 0:    
    # no correctionis applied:
    cnv = rdi_ndf.Pd0NDF()
    fns = cnv.read_files(pattern = "../data/PF*.PD0")
    config, data1d, data2d = cnv.read_data(fns, ctd_data=None)
    ndfdata = cnv.create_ndf(config, data1d, data2d)
    ndfdata.save("../data/test_dvl.ndf")
else:
    #corrections applied
    dbdreader.CACHEDIR = '../data/'
    dbds = dbdreader.MultiDBD(pattern="../data/comet*.[de]bd")
    tmp = dbds.get_sync("sci_ctd41cp_timestamp",
                        "sci_water_cond sci_water_temp sci_water_pressure m_lat m_lon".split())
    t, tctd, C, T, P, lat , lon = np.compress(tmp[2]>0, tmp, axis=1)

    SP = gsw.SP_from_C(C*10, T, P*10)
    SA = gsw.SA_from_SP_Baltic(SP, lon, lat)

    ctd_data = (tctd, SA, P*10)

    cnv = rdi_ndf.Pd0NDF()
    fns = cnv.read_files(pattern = "../data/PF*.PD0")
    config, data1d, data2d = cnv.read_data(fns, ctd_data)
    ndfdata = cnv.create_ndf(config, data1d, data2d)
    ndfdata.save("test_dvl_salinity_corrected.ndf")

