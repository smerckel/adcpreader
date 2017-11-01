Correction algorithms
=====================

Speed of sound
--------------

Currently the only correction class implemented applies to the
correction of the speed of sound. Normally an ADCP has a preset value
for the salinity, and computes the speed of sound from the measured
temperature and preset salinity. In cases where the salinity has been
set wrongly, or where the salinity changes, it could be necessary to
include a correction of the measurements due to the effects of the
speed of sound in the pipe line.

The class that gives access to the correction algorithms is
SpeedOfSoundCorrection(). The constructor of this class takes an
optional argument RTC_year_base. The year information contained in the
binary data are stored as a two-digit decimal number. By default the
year base is set to 2000, interpreting the year 17 as 2017. Should
data be processed that were collected in the previous century, then
the year base should be set accordingly.

The class SpeedOfSoundCorrection implements a method
horizontal_current_from_salinity_pressure(), which returns a generator
that modifies the horizontal components of the measured currents of
each ping. This method requires that the coordinate frame used to
output the data is set to 'Earth' so that the data actually contain
horizontal data. If the coordinate frame is set to something else than
'Earth' a ValueError is raised, aborting the pipeline. Calling this
method takes as argument an ensemble generator, a time vector (s), a
absolute salinity vector and a pressure vector (dbar). The time,
salinity and pressure vectors should be of equal length.

Another method that is implemented in this class is the 
current_correction_at_transducer_from_salinity_pressure() method,
which corrects the measured velocities in all bins by a factor
computing from the observed salinity in relation to the preset value
of salinity. The call signature of this method is identical to the
horiztonal_current_from_salinity_pressure() method.


