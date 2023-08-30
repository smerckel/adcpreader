import numpy as np
import adcpreader


mounted_pitch=11*np.pi/180

# first transform
t0 = adcpreader.rdi_transforms.TransformENU_SFU()
t1 = adcpreader.rdi_transforms.TransformSFU_XYZ(0, mounted_pitch, 0)
t2 = adcpreader.rdi_transforms.TransformXYZ_BEAM()
transform_enu_to_beam = t2 * t1 * t0

# second transform
t3 = adcpreader.rdi_transforms.TransformBEAM_XYZ()
t4 = adcpreader.rdi_transforms.TransformXYZ_SFU(0, mounted_pitch, 0)
transform_beam_to_sfu = t4 * t3

# third transform
transform_sfu_to_enu = adcpreader.rdi_transforms.TransformSFU_ENU()

# some data filtering:
max_velocity = 0.75
qc_u_limit = adcpreader.rdi_qc.ValueLimit(drop_masked_ensembles=False)
qc_u_limit.mask_parameter('velocity','Velocity1','||>',max_velocity)
qc_u_limit.mask_parameter('velocity','Velocity2','||>',max_velocity)
qc_u_limit.mask_parameter('velocity','Velocity3','||>',max_velocity)
qc_u_limit.mask_parameter('velocity','Velocity4','||>',max_velocity)


max_bt_velocity = 1.5
qc_bt_limit = adcpreader.rdi_qc.ValueLimit(drop_masked_ensembles=True)
qc_bt_limit.mask_parameter_regex('bottom_track','BTVel.*','||>',max_bt_velocity)


qc_snr_limit = adcpreader.rdi_qc.SNRLimit(3)

qc_amplitude_limit = adcpreader.rdi_qc.AcousticAmplitudeLimit(75)

# and a writer (sink)
writer = adcpreader.rdi_writer.NetCDFWriter('output.nc')

# and reader (source)
reader = adcpreader.rdi_reader.PD0()

pipeline = reader | transform_enu_to_beam | qc_u_limit | qc_bt_limit | qc_snr_limit 
pipeline |= qc_amplitude_limit | transform_beam_to_sfu | transform_sfu_to_enu | writer


with writer:
    pipeline.process('../data/PF230519.PD0')

