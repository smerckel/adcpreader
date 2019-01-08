A working example
=================


The objective
-------------

Let's say we have adcp binary data of an instrument that outputs
currents in a earth referenced frame (east north up, ENU), and we know
that the pitch values that are used are biased as well as the speed of
sound is computed wrongly because of a improper setting of the
salinity.

We will create a pipeline, generating ping-data, perform inverse
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
  filename = "../data/PF230519.PD0"
  
Then, an reader object is created. ::

  reader = rdi.rdi_reader.PD0()

We can call the process() method, with a filename or list of
filenames as argument, to read the binary files, ensemble per ensemble ::
  
  reader.process(filename)

The reader instance feeds each ensemble into a pipe line of processes,
which, for now, is still empty and all information will disappear into
a void. So, to do something useful, we will have to define a line of
operations, with a /sink/ at the end to consume all the data. Most
often the sink will be a process that writes the processed pings to a
file.

Setting up the pipeline, from start to finish
---------------------------------------------

Let's set up a pipeline of operations that do something useful, rather
than just burning CPU cycles. The
objective was to correct for wrong attitude information provided to
the ADCP. To that end, the pipeline is extended with rotation
transformations.

The module rdi_transforms defines a number of transformation classes,
transforming between the various coordinate systems used by the
ADCP. These coordinate systems are:

* BEAM (untransformed, along beam velocities)
* XYZ (x, y, z coordinates), an ADCP referenced frame.
* SFU (starboard forward up), a platform referenced frame
* ENU (east north up), an earth referenced frame

The transformation classes defined,  transform between

* BEAM -> XYZ
* XYZ -> SFU
* SFU -> ENU

Besides these transformations their inverse counterparts are
defined. The definition of the angles used in the inverse
transformations are as in the forward transformations.
Direct transformations between other combinations of coordinate
systems are not defined, but can easily constructed from
concatenating the basic transformations above. 


To transform the velocity data back to the instruments coordinate
system XYZ, we (inverse) transform it first to ships coordinates
(SFU), and then (inverse again) to the XYZ coordinates. The first
transformation becomes ::

  enu_sfu = rdi_transforms.TransformENU_SFU()

This particular transformation uses the attitude angles available in
each ping to do the rotation. The angles used (not visible to the
user) are defined as rotation angles to transform from SFU to
ENU. 

The second transform is the transform from SFU to XYZ coordinate
system. This transformation comprises of a single angle triplet, namely the
anlges at which the ADCP is mounted to the platform. For the glider
the would mean that the only the pitch angle is non-zero. Again, the
angles are defined for forward transforms, i.e. form XYZ to SFU. ::

  sfu_xyz = rdi_transforms.TransformSFU_XYZ(hdg=0, pitch=0.1919, roll=0)

If we know that the mounting pitch angle was not 0.1919 radians, but
0.2239, we can compute the velocity vectors relative to the platform,
but using the correct rotation ::
  
  xyz_sfu = rdi_transforms.TransformXYZ_SFU(hdg=0, pitch=0.2239, roll=0)

These successive rotations can be multiplied to get a resulting
transformation object ``transform``. Note that the multiplication has to be
performed in reversed order. ::
  
  # Set up the transformation pipeline. Note the order!
  transform = xyz_sfu * sfu_xyz * enu_sfu

Finally we would like to write the data into some format that we can
access easily. To that end, we create an object from the rdi_writer
module ::

  writer = rdi_writer.AsciiWriter()

(Invocation of the object AsciiWriter without arguments writes the
output to stdout.)

Now, each operation or process, receives an ensemble, does some
operation on it, and then passes it on to the next operation. These
operations are implemented as coroutines.

The idiom used to create such a train of operations looks like ::

  reader.send_to(transform)
  transform.send_to(writer)

In this example, the ``reader`` instance is the source (does not
receive data), and the ``writer`` instance is the sink (does not pass
on data further).

To feed the data into the pipeline, the ``process()`` method of the
reader is called::

  reader.process(filename)


By default, if all ensembles in the given filename have been
processed, the pipeline is closed, and no more data can be fed into
it. If applicable, any open files handled by the sink can be
closed. This means that, if a second file is to be processed, the pipe line has
to be constructed again. If the pipe line is *not* to be closed, so
that the pipeline will keep accepting data, the positional argument
of the ``process()`` method /close_coroutine_at_exit/ should be set
to False.
  
The full program listing then becomes (examples/convert_ascii.py)

.. code-block:: python

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
