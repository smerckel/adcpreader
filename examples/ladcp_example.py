import dbdreader
import ndf

if 0:
    dbd = dbdreader.MultiDBD(pattern = "/home/lucas/gliderdata/subex2016/hd/comet*.dbd")
    ndfdata = ndf.NDF("adcp_subex/glider_flight.ndf", open_mode='open')
    dvldata = ndf.MultiNDF("adcp_subex/comet*.ndf")
else:
    dbd = dbdreader.MultiDBD(pattern = "/home/lucas/gliderdata/latvia201711/hd/comet*.dbd")
    ndfdata = ndf.NDF("adcp_latvia/glider_flight.ndf", open_mode='open')
    dvldata = ndf.MultiNDF("adcp_latvia/comet*.ndf")


t_gps, lat_gps, lon_gps, hdg, ballast = dbd.get_sync("m_gps_lat", ["m_gps_lon", "m_heading", "m_ballast_pumped"])
m_depth = dbd.get("m_depth")
m_ballast_pumped = dbd.get("m_ballast_pumped")
m_water_vx = dbd.get("m_water_vx")
m_water_vy = dbd.get("m_water_vy")



try:
    gf_t, gf_u, gf_v, gf_z, gf_U = ndfdata.get_sync("u", ["v", "z", 'U'])
except:
    gf_t, gf_u, gf_v, gf_z, gf_w = ndfdata.get_sync("u", ["v", "z", "w"])
    gf_U = (gf_u**2 + gf_v**2 + gf_w**2)**0.5



data_dict = dict(time = gf_t, pressure = gf_z/10, gf_u = gf_u, gf_v=gf_v, gf_U = gf_U)

dvl_t, dvl_gf_z, dvl_gf_u, dvl_gf_v, dvl_gf_w = dvldata.get_sync("glider_flight depth", ["glider_flight velocity_east",
                                                                                         "glider_flight velocity_north",
                                                                                         "glider_flight velocity_up",])
_, dvl_r, dvl_u = dvldata.get("velocity east")
_, _, dvl_v = dvldata.get("velocity north")
_, _, dvl_w = dvldata.get("velocity up")

_, dvl_bt_u, dvl_bt_v, dvl_bt_w = dvldata.get_sync("bottom_track east", ["bottom_track north",
                                                                         "bottom_track up"])

dvldata_dict = dict(time = dvl_t, pressure = dvl_gf_z/10., dvl_gf_z = dvl_gf_z,
                    dvl_gf_u = dvl_gf_u, dvl_gf_v = dvl_gf_v, dvl_gf_w = dvl_gf_w,
                    dvl_u = dvl_u.T, dvl_v = dvl_v.T, dvl_w = dvl_w.T,
                    dvl_bt_u = dvl_bt_u, dvl_bt_v = dvl_bt_v, dvl_bt_w = dvl_bt_w)
                    
print("Data read...")










        
# set the data    
surface_data = Surface_Data(data.t_gps, data.lat_gps, data.lon_gps, data.ballast)
        
wv = WaterVelocities(data = data.data_dict,
                     surface_data = surface_data,
                     magnetic_variation = 3)

wv.integrate_glider_velocities()

surface_velocities = wv.compute_surface_velocities(min_gps_fixes=3)
barotropic_velocities = wv.compute_barotropic_velocities(max_subsurface_time=3*3600)

#data.dvldata_dict['dvl_bt_u'] = np.ma.masked_where(np.abs(data.dvldata_dict['dvl_bt_u'])>0.75, data.dvldata_dict['dvl_bt_u'])


ps = iterprofiles.ProfileSplitter(data = data.dvldata_dict)
ps.split_profiles()

ladcp = LoweredADCP()
ladcp.method_settings(use_bottom_track = True, use_glider_flight = True,
                      use_barotropic_velocity_constraint = True,
                      use_surface_velocity_constraint = True)
ladcp.initialise(zi_min=0, zi_max=100, dz=1, dvl_z = data.dvl_r)


ladcp_data = dict()
for velocity_component in 'u v'.split():
    #velocity_component = 'v'
    us_fun = interp1d(surface_velocities['time'], surface_velocities[velocity_component],
                      bounds_error=False, fill_value = (np.nan, np.nan))
    ub_fun = interp1d(barotropic_velocities['time'], barotropic_velocities[velocity_component],
                      bounds_error=False, fill_value = (np.nan, np.nan))
    ug = []
    uw = []

    for i, p in enumerate(ps):
        print("Profile %d"%(i))
        dvl_t, dvl_gf_z = p.get_cast("dvl_gf_z")
        _, dvl_u = p.get_cast("dvl_%s"%(velocity_component))
        _, dvl_bt_u = p.get_cast("dvl_bt_%s"%(velocity_component))
        _, dvl_gf_u = p.get_cast("dvl_gf_%s"%(velocity_component))
        u_barotropic = ub_fun(p.t_down)
        u_surface = us_fun(p.t_down)
        dvl_bt_u = np.ma.masked_where(np.isnan(dvl_bt_u), dvl_bt_u)

        _ug, _uw = ladcp.compute_velocity_profile(dvl_u, dvl_bt_u, dvl_gf_u, dvl_gf_z,
                                                  u_barotropic = float(u_barotropic),
                                                  u_surface=float(u_surface))
        ug.append(_ug)
        uw.append(_uw)
    t = np.array([p.t_down for p in ps])
    ug = np.ma.hstack(ug)
    uw = np.ma.vstack(uw)
    ladcp_data['time'] = t
    ladcp_data[velocity_component] = uw


f, ax = subplots(4,1, sharex=True)

pcm_u = ax[0].pcolormesh(epoch2num(ladcp_data['time']), ladcp.zi, ladcp_data['u'].T, vmin=-0.5, vmax=0.5)
colorbar(pcm_u, ax=ax[0], orientation='horizontal')

pcm_v = ax[1].pcolormesh(epoch2num(ladcp_data['time']), ladcp.zi, ladcp_data['v'].T, vmin=-0.5, vmax=0.5)
colorbar(pcm_u, ax=ax[1],orientation='horizontal')

wd_t, wd = data.dbd.get("m_water_depth")
wd_t, wd = np.compress(wd>40, (wd_t, wd), axis=1)

ax[0].plot(epoch2num(wd_t), wd)
ax[1].plot(epoch2num(wd_t), wd)

ax[0].set_ylabel('Depth (m)')
ax[1].set_ylabel('Depth (m)')
ax[0].set_ylim(110,0)
ax[1].set_ylim(110,0)
ax[2].plot(epoch2num(surface_velocities['time']), surface_velocities['u'], label='Surface velocity')
ax[3].plot(epoch2num(surface_velocities['time']), surface_velocities['v'], label='Surface velocity')
ax[2].plot(epoch2num(barotropic_velocities['time']), barotropic_velocities['u'], label='Barotropic velocity')
ax[3].plot(epoch2num(barotropic_velocities['time']), barotropic_velocities['v'], label='Barotropic velocity')
ax[2].set_ylabel('Eastward\nvelocity (m s$^{-1}$)')
ax[3].set_ylabel('Northward\nvelocity (m s$^{-1}$)')
ax[3].xaxis.set_major_formatter(DateFormatter("%d/%b\n%H:%M"))
ax[2].set_ylim(-0.75, 0.75)
ax[3].set_ylim(-0.75, 0.75)
ax[2].legend(loc='lower left', ncol=2)
ax[3].legend(loc='lower left', ncol=2)





p = ps[100]
dvl_t, dvl_gf_z = p.get_cast("dvl_gf_z")
_, dvl_u = p.get_cast("dvl_%s"%(velocity_component))
_, dvl_bt_u = p.get_cast("dvl_bt_%s"%(velocity_component))
_, dvl_gf_u = p.get_cast("dvl_gf_%s"%(velocity_component))
u_barotropic = ub_fun(p.t_down)
u_surface = us_fun(p.t_down)
dvl_bt_u = np.ma.masked_where(np.isnan(dvl_bt_u), dvl_bt_u)

_ug, _uw = ladcp.compute_velocity_profile(dvl_u, dvl_bt_u, dvl_gf_u, dvl_gf_z,
                                          u_barotropic = float(u_barotropic),
                                          u_surface=float(u_surface))
