Quality control
===============

Data that are collected by the ADCP cannot always be accepted as valid
data. Reasons for this may be computations based on signals that are
too noisy, bins with attributed physical positions that are not in the
water column, etc.

Two classes are defined that can be configured and concatenated in the
pipeline. One class flags data based on values that exceed some limit,
whereas the other class flags data bad based on a signal to noise
ratio being below a given threshold. Data that are flag bad are not
removed from the ensemble, but are masked. This means that when any of
these filters are applied in the pipeline, data arrays can be masked
arrays instead of normal arrays. For certain operations this can have
a significant impact on performance.

ValueLimit Class
----------------

The class ValueLimit is a multipurpose filter. A typical application
is to require that velocities observed should be smaller than some
physical realistic value.

This filter is used by creating an instance of the ValueLimit class
and set the condition based on which data should be masked, for example: ::

  discard_velocities = ValueLimit()
  discard_velocities.set_discard_condition("velocity",
                                           "Velocity1",
					   "||>",
					   10)
  discard_velocities.set_discard_condition("velocity",
                                           "Velocity2",
					   "||>",
					   10)

The set_discard_condition() method defines how to discriminate the
data. The method takes four arguments. The first argument defines the
data block, such as "velocity", "echo", or "bottom_track". The second
argument sets the variable in this block, which is compared to a
certain value given as fourth argument according to an operator given
as third argument. Repeated calling of this method appends the
discrimination criterion.


HOW TO MAKE A TABLE???? ::
  
   > variable is discarded if value is greater than threshold
   >= variable is discarded if value is greater than or equal to threshold
   < variable is discarded if value is less than or equal to threshold
   <= variable is discarded if value is less than or equal to threshold
   ||> variable is discarded if absolute value is greater than threshold
   ||>= variable is discarded if absolute value is greater than or equal to threshold
   ||<  variable is discarded if absolute value is less than or equal to threshold
   ||<=  variable is discarded if absolute value is less than or equal to threshold



SNRLimit
--------

To mask values that have an echo intensity that is close to the noise
floor, the class SNRLimit can be used. It is setup in a single call ::

  snr_limit = SNRLimit(SNR_limit=10, noise_floor_db=26.1)

The signal-to-noise ratio (SNR) is computed from::
  
  SNR = 10**((echointensity-noise_floor_db)/10)

The noise floor is the is set in db. It can be estimated from finding
the lowest measured echo intensity by the instrument. Values for which
the SNR is lower than the given SNR_limit, are masked.  

   
Example
-------

An example of the use a ValueLimit filter and a SNRLimit filter is shown below ::

  pd0 = PD0()
  :
  :
  discard_velocities = ValueLimit()
  discard_velocities.set_discard_condition("velocity",
                                           "Velocity1",
					   "||>",
					   10)
  discard_noisy_returns = SNRLimit(10, 26.1)
  
  :
  # start of pipeline
  ensembles = pd0.ensemble_generator(filenames)
  :
  :
  ensembles = discard_velocities(ensembles)
  ensembles = discard_noisy_returns(ensembles)
  :
  :
  # further processing
  :
  :

					 
