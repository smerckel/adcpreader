A working example
=================


The objective
-------------

Let's say we have adcp binary data of an instrument that outputs
currents in a earth referenced frame (east north up, ENU), and we know
that the pitch values that are used are biased as well as the speed of
sound is computed wrongly because of a improper setting of the
salinity.

We will create a generator generating ping-data, perform inverse
rotation transformations, representing the velocities relative to the platform,
applying a speed of sound correction, and perform a forward rotation
transformation using correct attitude data. Finally the data are
written to some convenient data format.


Starting the pipeline
----------------------


We start with importing some modules and creating a list of (one)
filename with binary data. ::

  import numpy as np

  import rdi
  filenames = ["../data/PF230519.PD0"]
  
Then, an reader object is created. ::

  reader = rdi.rdi_reader.PD0()

For a list of filenames, the ensemble_generator method returns a
generator object, yielding all the pings that are contained in the
list of binary data files. ::

  ensembles = reader.ensemble_generator(filenames)


We could now make a loop and do some per ping processing: ::

  for i,ens in enumerate(ensembles):
      do_some_processing_on_ensemble(ens)

Setting up the pipeline, from start to finish
---------------------------------------------

Let's fill in the do_some_processing_on_ensemble() function. The
objective was to correct for wrong attitude information provided to
the ADCP. To that end, the pipeline is extended with rotation
transformations.

The module rdi_transforms defines a number of transformation classes,
transforming between the various coordinate systems used by the
ADCP. These coordinate systems are:

* BEAM (untransformed, along beam velocities)
* XYZ (x, y, z coordinates), an ADCP referenced frame.
* FSU (forward starboard up), a platform referenced frame
* ENU (east north up), an earth referenced frame

The transformation classes defined,  transform between

* BEAM -> XYZ
* XYZ -> FSU
* FSU -> ENU

Besides these transformations their inverse counterparts are
defined. The definition of the angles used in the inverse
transformations are as in the forward transformations.
Direct transformations between other combinations of coordinate
systems are not defined, but can easily constructed from
concatenating the basic transformations above. 


To transform the velocity data back to the instruments coordinate
system XYZ, we (inverse) transform it first to ships coordinates
(FSU), and then (inverse again) to the XYZ coordinates. The first
transformation becomes ::

  enu_fsu = rdi_transforms.TransformENU_FSU()

This particular transformation uses the attitude angles available in
each ping to do the rotation. The angles used (not visible to the
user) are defined as rotation angles to transform from FSU to
ENU. 

The second transform is the transform from FSU to XYZ cooridnate
system. This transformation comprises of a single angle triplet, namely the
anlges at which the ADCP is mounted to the platform. For the glider
the would mean that the only the pitch angle is non-zero. Again, the
angles are defined for forward transforms, i.e. form XYZ to FSU. ::

  fsu_xyz = rdi_transforms.TransformFSU_XYZ(alpha=0, beta=0.1919, gamma=0)

If we know that the mounting pitch angle was not 0.1919 radians, but
0.2239, we can compute the velocity vectors relative to the platform,
but using the correct rotation ::
  
  xyz_fsu = rdi_transforms.TransformXYZ_FSU(alpha=0, beta=0.2239, gamma=0)

These successive rotations can be multiplied to get a resulting
transformation object ``t4``. Note that the multiplication has to be
performed in reversed order. ::
  
  # Set up the transformation pipeline. Note the order!
  transform = xyz_fsu * fsu*xyz * enu_fsu

Finally we would like to write the data into some format that we can
access easily. To that end, we create an object from the rdi_writer
module ::

  writer = rdi_writer.AsciiWriter()

(Invocation of the object AsciiWriter without arguments writes the
output to stdout.)

The final pipeline then becomes: ::


  ensembles = reader.ensemble_generator(filenames)
  ensembles = transform(ensembles)
  writer.write_ensembles(ensembles)

or more compact (and less readable): ::
  
  writer.write_ensembles(t4(reader.ensemble_generator(filenames)))


The full program listing then becomes (examples/convert_ascii.py)::

  import numpy as np

  from rdi import rdi_reader, rdi_transforms, rdi_writer

  filenames = ["../data/PF230519.PD0"]

  bindata = rdi_reader.PD0()

  enu_fsu = rdi_transforms.TransformENU_FSU()
  fsu_xyz = rdi_transforms.TransformFSU_XYZ(hdg=0, pitch=0.1919, roll=0)
  xyz_fsu = rdi_transforms.TransformXYZ_FSU(hdg=0, pitch=0.2239, roll=0.05)
  # Set up the transformation pipeline. Note the order!
  transform = xyz_fsu * fsu_xyz * enu_fsu

  # write to a file
  #writer = rdi_writer.AsciiWriter(filename = 'test.ascii')
  #
  #or to stdout
  #
  writer = rdi_writer.AsciiWriter()

  #now do the job...
  ensembles = bindata.ensemble_generator(filenames)
  ensembles = transform(ensembles)
  writer.write_ensembles(ensembles)


