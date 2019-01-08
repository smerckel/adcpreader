Correction algorithms
=====================

Speed of sound
--------------

Normally an ADCP has a preset value
for the salinity, and computes the speed of sound from the measured
temperature and preset salinity. In cases where the salinity has been
set wrongly, or where the salinity changes, it could be necessary to
include a correction of the measurements due to the effects of the
speed of sound in the pipe line.

A number of different options are available 

# HorizontalCurrentCorrectionFromSalinityPressure
# CurrentCorrectionFromSalinityPressure
# CurrentCorrectionFromSalinity

All these classes derive from SpeedOfSoundCorrection, and take an
option positional argument RTC_year_base (defaults to 2000).  The year information contained in the
binary data are stored as a two-digit decimal number. By default the
year base is set to 2000, interpreting the year 17 as 2017. Should
data be processed that were collected in the previous century, then
the year base should be set accordingly.

The class ``CurrentCorrectionFromSalinity`` is simplest, and scales
all velocities with a correction factor for the speed of sound, based
on a constant value for salinity. If both salinity and pressure are
given as time series, the classes ``HorizontalCurrentCorrectionFromSalinityPressure``
and ``CurrentCorrectionFromSalinityPressure`` can be used which
correct the horizontal are all three velocity components,
respectively. The former correction method requires the coordinate
frame to be ``Earth``.


Echo intensities can be scaled, per beam, using the
`ScaleEchoIntensities` class. An instant of this class can be inserted
in the pipeline to scale the echo intensities of each beam according
to factors given to the constructor.


A number of ensembles can be averaged into a single ensemble, using
the class `Aggregator`. The parameters averaged are:

# Roll
# Pitch
# Heading
# Soundspeed
# Salin
# Temp
# Press
# Time
# Timestamp
# Velocity1
# Velocity2
# Velocity3
# Velocity4
# Echo1
# Echo2
# Echo3
# Echo4
# BTVel1
# BTVel2
# BTVel3
# BTVel4


The classes  `AttitudeCorrectionLinear` and
`AttitudeCorrectionTiltCorrection` allow pitch, and heading, pitch and
roll, to be corrected respectively. This may be necessary for data
collected with an attitude sensor that reports biased tilt angles
(pitch and roll) as happened to glider Comet due to a leaking tilt
sensor. The simple correction method requires the linear coefficients
of a and b of a correction relation for the pitch. The more complex
AttitudeCorrectTiltCorrection class corrects for the errors in pitch
and roll, and corrects for the errors introduced into the heading
measurement.


