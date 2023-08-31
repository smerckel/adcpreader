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

+--------+-----------------------------------------------------------------+
|BEAM    | velocities as measured by the instrument in along beam direction|
+--------+-----------------------------------------------------------------+
|XYZ     | velocities expressed in a coordinate system fixed to the ADCP   |
+--------+-----------------------------------------------------------------+
|SFU     | velocities expressed in 'ship coordinates', that is relative to |
|        | the ship or platform (glider) that carries the ADCP. SFU is     |
|        | short for Forward, Starboard and Up.                            |
+--------+-----------------------------------------------------------------+
|ENU     | velocities expressed in a earth referenced frame. ENU is short  |
|        | for Eastward, Northward and Up.                                 |
+--------+-----------------------------------------------------------------+

Both the BEAM and XYZ coordinate systems are bound to the ADCP only
and require no external information to apply the transformation. To
convert to the SFU coordinate system the mounting angles of the ADCP
around the forward, starboard and up axis of the platform are
required. To transform to the ENU coordinate system, in addition the attitude of
the platform (pitch, heading and roll) is required.

The rdi_transforms module defines a number of classes to transform
between two coordinate systems. Each transformation is a single step
transformation. That is, only (bidrectional) transformations between BEAM and XYZ,
XYZ and SFU, and SFU and ENU are implemented. An instance that
transforms between other coordinate system combinations than those
listed above, can be simply obtained by left multiplying instances of
those single step transformations.

The transformation classes are all derived from an abstract
transformation class Transform() which provides the machinery common
to all transformation classes. The available transformation classes

::

   TransformBEAM_XYZ()

::

   TransformXYZ_SFU()

::
   
   TransformSFU_ENU()


These transformation classes define transformations in the direction
the ADCP applies them. This means that same sign convention applies to
the angles of rotation used by
these transformations as is used internally by the ADCP.

**Inverse**
transformations can be defined by setting the optional parameter ``inverse =
True`` when calling the constructor, or using the predefined classes
(which are essentially short-hands for the transformation classes
above with ``inverse = True`` set):

::

   TransformXYZ_BEAM()

::

   TransformSFU_XYZ()

::
   
   TransformENU_SFU()


.. note::
   The angles of rotation applied to the inverse
   transformations still comply to the sign conventions used in the
   forward rotations.

Example
^^^^^^^

Let's assume the measurements are expressed in geodetic coordinates
(ENU), and the ADCP was mounted to the platform frame with angles
slightly deviating from what has been prescribed in the ADCP
configuration file. This means we have to undo the transformations
until the ADCP's frame of reference (XYZ) and the redo the
transformations using the correct mounting angles.

.. code-block:: python
  
   mounting_pitch_angle = 12*3.1415/180.
   configured_mounting_pitch_angle = 11*3.1415/180.
   # First undo last transformation step
   t1 = TransformENU_SFU()
   # then undo the second, using the mounting angle specified:
   t2 = TransformSFU_XYZ(hdg=0, pitch=configured_mounting_pitch_angle,
		         roll=0)
   # Now redo the transformation from XYZ to SFU using the correct
   # mounting angle
   t3 = TransformXYZ_SFU(hdg=0, pitch=mouting_pitch_angle, roll=0)
   #and redo the transformation to geodetic coordinates
   t4 = TransformSFU_ENU()
   
   # the resulting transformation is given by left-multiplying the
   #single transformation steps:
   t_result = t4 * t3 * t2 * t1
   
