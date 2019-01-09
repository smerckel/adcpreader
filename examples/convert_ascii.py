import numpy as np

import rdi

filename = "../data/PF230519.PD0"

reader = rdi.rdi_reader.PD0()

enu_sfu = rdi.rdi_transforms.TransformENU_SFU()
sfu_xyz = rdi.rdi_transforms.TransformSFU_XYZ(hdg=0, pitch=0.1919, roll=0)
xyz_sfu = rdi.rdi_transforms.TransformXYZ_SFU(hdg=0, pitch=0.2239, roll=0)
transform = xyz_sfu * sfu_xyz * enu_sfu

with open("example_data.txt", "w") as fp:
    writer = rdi.rdi_writer.AsciiWriter(fp)

    # set up the pipeline
    reader.send_to(transform)
    transform.send_to(writer)

    # and process the data.
    reader.process(filename)
