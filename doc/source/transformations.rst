Transformations
===============

The ADCP measures the water velocity in a number of beams (mostly
four), in the direction of each beam. Internally these settings are
transformed to a coordinate system as set by the user.

Transformation operations can be included in the pipeline to change
the coordinate system in which the velocity data are output by the
ADCP and/or apply corrections to the transformations that the ADCP has
done internally. For example, if the output coordinate system is set
to output velocities as a vector consisting of eastward, northward and
upward velocities, but an analysis requires the velocities expressed
relative to the platform that carries the ADCP, then the velocities
need to be transformed accordingly. Another instance where
transformations are required is when the ADCP is mounted with
different angles than prescribed in the setup file. For example, if
the ADCP is mounted with a slight roll angle, whereas the prescribed
roll angle is zero, this can be corrected for by transforming the
velocity data back to the coordinate system of the ADCP itself, using
the same values as the ADCP has used internally, followed by the
inverse transformation, using the corrected angles.

The rdi_transformation module defines the following coordinate
systems:

**BEAM**: velocities as measured by the instrument in along beam direction

**XYZ**:  velocities expressed in a coordinate system fixed to the ADCP

**FSU**:  velocities expressed in 'ship coordinates', that is relative to
the ship or platform (glider) that carries the ADCP. FSU is short for
Forward, Starboard and Up.

**ENU**:  velocities expressed in a earth referenced frame. ENU is short
for Eastward, Northward and Up.


Both the BEAM and XYZ coordinate systems are bound to the ADCP only
and require no external information to apply the transformation. To
convert to the FSU coordinate system the mounting angles of the ADCP
around the forward, starboard and up axis of the platform are
required. To transform to the ENU coordinate system, in addition the attitude of
the platform (pitch, heading and roll) is required.

The rdi_transforms module defines a number of classes to transform
between two coordinate systems. Each transformation is a single step
transformation. That is, only (bidrectional) transformations between BEAM and XYZ,
XYZ and FSU, and FSU and ENU are implemented. An instance that
transforms between other coordinate system combinations than those
listed above, can be simply obtained by left multiplying instances of
those single step transformations.

The transformation classes are all derived from an abstract
transformation class Transform() which provides the machinery common
to all transformation classes. The available transformation classes

TransformBEAM_XYZ()

TransformXYZ_FSU()

+--------+----------+------+
| nombre | apellido | edad |
+========+==========+======+
| pepe   | zarate   | 28   |
+--------+----------+------+
| toto   | garcia   | 29   |
+--------+----------+------+

.. math::

   \frac{ \sum_{t=0}^{N}f(t,k) }{N}


   
:math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}`

TransformFSU_ENU()


