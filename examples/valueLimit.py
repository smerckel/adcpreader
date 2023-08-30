import numpy as np

import adcpreader

filename = "../data/PF230519.PD0"

reader = adcpreader.rdi_reader.PD0()

# mask velocities if the absolute value is larger than 10 m/s
mask_velocities = adcpreader.rdi_qc.ValueLimit()
mask_velocities.mask_parameter("velocity",
                                  "Velocity1",
				  "||>",
				  10)
mask_velocities.mask_parameter("velocity",
                                  "Velocity2",
				  "||>",
				  10)
mask_velocities.mask_parameter("velocity",
                                  "Velocity3",
				  "||>",
				  10)
mask_velocities.mask_parameter("velocity",
                                  "Velocity4",
				  "||>",
				  10)

# drops ensembles if bottom track velocity is larger than 0.5 m/s
# use a regelar expression to apply to all velocity vectors
drop_ensembles = adcpreader.rdi_qc.ValueLimit(drop_masked_ensembles=True)
drop_ensembles.mask_parameter_regex("bottom_track",
                                    "BTVel.*",
				    "||>",
				    0.5)

# mask bins that where the SNR is less than 10:
mask_noisy_returns = adcpreader.rdi_qc.SNRLimit(10, 26.1)

# count the ensembles processed at the beginning and end of the pipeline:
counter_front = adcpreader.rdi_qc.Counter()
counter_end = adcpreader.rdi_qc.Counter()

data = adcpreader.rdi_writer.DataStructure()

reader | counter_front | mask_velocities | drop_ensembles | mask_noisy_returns | counter_end |data


reader.process('../data/PF230519.PD0')

number_of_dropped_profiles = counter_front.counts - counter_end.counts
