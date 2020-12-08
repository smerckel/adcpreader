import sys
sys.path.insert(0, "..")

from HZGnetCDF import ncHZG
from slocum.dvl_processing import DVLProcessor

    
dvl_data_file = "/home/lucas/gliderdata/nsb3_201907/process/dvl_3beam_nsb3_201907.nc"
gld_flght_data_file = "/home/lucas/gliderdata/nsb3_201907/process/comet_nsb3_201907.nc"
dbd_pattern = "/home/lucas/gliderdata/nsb3_201907/hd/comet*.dbd"


dvlp = DVLProcessor(dvl_data_file, gld_flght_data_file, dbd_pattern,
                    rho_reference=1025, buoyancy_sensor_name='m_ballast_pumped',
                    filter_period=15)


ladcp_configuration = dict(use_bottom_track=True,
                           use_barotropic_velocity_constraint=False,
                           use_surface_velocity_constraint=False,
                           use_glider_flight=False,
                           z_min=5,
                           z_max=45,
                           dz=0.5)

dvlp.configure_ladcp(**ladcp_configuration)

sv, bv = dvlp.compute_surface_and_barotropic_water_velocities()

velocity_data = dvlp.compute_velocity_matrix()

conf = ncHZG.get_default_conf()
conf['title'] = 'DVL data for comet nsb3_201907'
conf['source'] = 'data.py'
for k, v in ladcp_configuration.items():
    conf[k] = v
t = velocity_data[0]
z = velocity_data[1]
u = velocity_data[2].T
v = velocity_data[3].T
with ncHZG('comet_dvl_nsb3_201908.nc', mode='w', **conf) as nc:
    nc.add_parameter('u', 'm/s', t, z, u, standard_name='eastward_velocity')
    nc.add_parameter('v', 'm/s', t, z, v, standard_name='northward_velocity')
    nc.add_parameter('surface_velocity/u', 'm/s', sv['time'], sv['u'], standard_name="eastward surface velocity", time_dimension="Ts")
    nc.add_parameter('surface_velocity/v', 'm/s', sv['time'], sv['v'], standard_name="northward surface velocity", time_dimension="Ts")
    nc.add_parameter('averaged_velocity/u', 'm/s', bv['time'], bv['u'], standard_name="eastward depth-averaged velocity", time_dimension="Tb")
    nc.add_parameter('averaged_velocity/v', 'm/s', bv['time'], bv['v'], standard_name="northward depth-averaged velocity", time_dimension="Tb")
    
