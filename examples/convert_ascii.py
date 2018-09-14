import numpy as np

from rdi import rdi_reader, rdi_transforms, rdi_writer

pipeline = rdi_reader.Pipeline()

filenames = ["../data/PF230519.PD0"]


enu_fsu = rdi_transforms.TransformENU_FSU()
fsu_xyz = rdi_transforms.TransformFSU_XYZ(hdg=0, pitch=0.1919, roll=0)
xyz_fsu = rdi_transforms.TransformXYZ_FSU(hdg=0, pitch=0.2239, roll=0.05)

### now add the transforms to the pipeline:

pipeline.add(enu_fsu)
pipeline.add(fsu_xyz)
pipeline.add(xyz_fsu)

### or combine these transormations in one. Note the order!
# transform = xyz_fsu * fsu_xyz * enu_fsu
### and add transform to the pipeline operations:
# pipeline.add(transform)



### write to a file
### sink = rdi_writer.AsciiWriter(filename = 'test.ascii')
###
### or to stdout
###
sink = rdi_writer.AsciiWriter()

sink(pipeline(filenames))
