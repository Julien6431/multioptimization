# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openap.casadi import aero as aero_casadi
from openap import aero, nav, top, prop
from tqdm import tqdm
from traffic.data import navaids, airports
pd.set_option("display.max_rows",15)
# %%
def obj_pop_exposure_day(x, u, dt, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    tas = aero_casadi.mach2tas(mach, h)
    thrust = optimizer.drag.clean(m, tas, h / aero.ft, vs / aero.fpm)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=True, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.001*cost * thrust + fuel

def obj_pop_exposure_night(x, u, dt, **kwargs):
    xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
    mach, vs, psi = u[0], u[1], u[2]
    tas = aero_casadi.mach2tas(mach, h)
    thrust = optimizer.drag.clean(m, tas, h / aero.ft, vs / aero.fpm)
    cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=True, **kwargs)
    fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
    return 0.008 * cost * thrust + fuel
    
#%%
map_type = "DN"
eham = nav.airport("EHAM")
actype = "a320"
start = (eham["lat"], eham["lon"])
nodes = 39
c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")

flights0 = None
flights = None
for i in tqdm(c_ends.index[:]):
    fid = c_ends.fid.values[i]
    start = "EHAM"
    m0 = c_ends.tow.values[i] / prop.aircraft(actype)["mtow"]
    end = (c_ends.latitude.values[i], c_ends.longitude.values[i])
    h_end=c_ends.altitude.values[i] * aero.ft

    #for now the runways are not used
    rwy = None  #airports["EHAM"].runways.data.query(f"name=='{c_ends.runway.values[i]}'")
    if rwy is None or len(rwy)==0:
        trk_start = None
    else:
        trk_start=rwy.bearing.values[0]

    # generate fuel optimal trajectory
    optimizer = top.Climb(actype, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=1000)
    flight_fuel = optimizer.trajectory(
        objective="fuel",
        h_end=h_end,
        trk_start=trk_start,
    ).assign(fid=fid)

    # generate optimalpopulation exposure trajectory
    optimizer = top.Climb(actype, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=1000)

    interpolant = top.tools.interpolant_from_dataframe(df_cost)
    if map_type == "N":
        flight_pop = optimizer.trajectory(
            objective=obj_pop_exposure_night,
            interpolant=interpolant,
            h_end=h_end,
            trk_start=trk_start,
        ).assign(fid=fid)
    else:
        flight_pop = optimizer.trajectory(
            objective=obj_pop_exposure_day,
            interpolant=interpolant,
            h_end=h_end,
            trk_start=trk_start,
        ).assign(fid=fid)

    # calculate cost from grid cost
    cost = interpolant(
        np.array([flight_pop.longitude.values, flight_pop.latitude.values, flight_pop.h.values])
    )
    cost0 = interpolant(
        np.array([flight_fuel.longitude.values, flight_fuel.latitude.values, flight_fuel.h.values])
    )
    flight_pop = flight_pop.assign(cost_grid=cost.full()[0])
    flight_fuel = flight_fuel.assign(cost_grid=cost0.full()[0])

    if flights0 is None:
        flights = flight_pop
        flights0 = flight_fuel
    else:
        flights = pd.concat([flights, flight_pop])
        flights0 = pd.concat([flights0, flight_fuel])

flights.to_csv(f"data_generated/flights_noise_opt_{map_type}.csv", index=False)
flights0.to_csv(f"data_generated/flights_fuel_opt_{map_type}.csv", index=False)
# %%
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.colors as mcolors

# map_type = "DN"
colors = list(mcolors.TABLEAU_COLORS.keys())
colors.extend(["b", "g", "y", "m", "c"])
flights = pd.read_csv(f"data_generated/flights_noise_opt_{map_type}.csv")
flights0 = pd.read_csv(f"data_generated/flights_fuel_opt_{map_type}.csv")
df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
df_real = pd.read_parquet(f"data_generated/opensky2024_centroids_{map_type}.parquet")
c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
cost_grid = df_cost.cost.values.reshape(30, 20, 20)

norm = plt.Normalize(vmin=0.00001, vmax=0.0008)
proj = ccrs.TransverseMercator(
    central_longitude=eham["lon"], central_latitude=eham["lat"]
)
trans = ccrs.PlateCarree()
fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    subplot_kw=dict(projection=proj),
)

ax.add_feature(BORDERS, linestyle="dotted", alpha=0.4)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.4)
ax.set_extent(
    [
        min(flights.longitude.values) - 0.2,
        max(flights.longitude.values) + 0.2,
        min(flights.latitude.values) - 0.2,
        max(flights.latitude.values) + 0.2,
    ]
)

norm = plt.Normalize(vmin=0.0001, vmax=0.04, clip=True)
cntr = ax.contourf(
    df_cost.longitude.values.reshape(30, 20, 20)[:, :, 5],
    df_cost.latitude.values.reshape(30, 20, 20)[:, :, 5],
    cost_grid[:, :, 3],
    cmap="binary",
    transform=trans,
    levels=12,
    norm=norm,
    alpha=0.5,
)

for i, fid in enumerate(flights0.fid.unique()[:]):

    flightr = df_real.query(f"flight_id=='{fid}'")
    ax.plot(
        flightr.longitude,
        flightr.latitude,
        color="tab:blue",
        lw=2,
        transform=trans,
        label="Real flights centroids" if i == 0 else None,
    )
    flight = flights.query("fid==@fid")
    flight0 = flights0.query("fid==@fid")
    ax.plot(
        flight.query("cost_grid>0").longitude,
        flight.query("cost_grid>0").latitude,
        color="k",
        lw=2,
        transform=trans,
        label="Noise-optimal" if i == 0 else None,
    )
    ax.plot(
        flight.longitude,
        flight.latitude,
        color="k",
        lw=1,
        linestyle="dashed",
        transform=trans,
    )
    ax.plot(
        flight0.longitude,
        flight0.latitude,
        color="r",
        lw=1,
        linestyle="dashed",
        transform=trans,
        label=f"Fuel-optimal" if i == 0 else None,
    )

plt.legend()
plt.tight_layout()
plt.savefig(f"figs/frn_{map_type}.png", bbox_inches="tight")
plt.show()
# %%
# $\mu_{\text{Fuel}}$ & \SI{954.0}{\kilogram} & \SI{956.2}{\kilogram} & \SI{1484.1}{\kilogram} \\
# $\mu_{\text{Cost}}$ &\num{8.189e-3} & \num{7.903e-3} & \num{9.688e-3} \\
# $\Sigma_{\text{Fuel}}$ & \SI{12402}{\kilogram} & \SI{12430}{\kilogram} & \SI{19293}{\kilogram} \\
# $\Sigma_{\text{Cost}}$ & \num{106.5e-3} & \num{102.7e-3} & \num{125.9e-3} \\
# %%
map_type = "D"
flights = pd.read_csv(f"data_generated/flights_noise_opt_{map_type}.csv")
flights0 = pd.read_csv(f"data_generated/flights_fuel_opt_{map_type}.csv")
for i, fid in enumerate(flights0.fid.unique()[:]):
    flight = flights.query("fid==@fid")
    flight0 = flights0.query("fid==@fid")
    print("cost", flight.cost_grid.sum() / flight0.cost_grid.sum())
    print("fuel", flight.fuel.sum() / flight0.fuel.sum())
print("cost_tot", flights.cost_grid.sum() / flights0.cost_grid.sum())
print("fuel_tot", flights.fuel.sum() / flights0.fuel.sum())
# %%
import matplotlib.colors as mcolors
from traffic.core import Traffic, Flight


colors = list(mcolors.TABLEAU_COLORS.keys())
colors.extend(["r", "b", "g", "y", "m", "c"])
flights = pd.read_csv(f"data_generated/opsk_flights_noise_realistic_{map_type}.csv")
flights0 = pd.read_csv(f"data_generated/opsk_flights0_fuel_realistic_{map_type}.csv")
df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}_2024.csv")
df_real = (
    pd.read_parquet(f"data_generated/opensky2024_centroids_{map_type}.parquet")
    # .query("flight_id=='VLG12BR_219'")
    # .reset_index(drop=True)
)
t_real = Traffic(df_real)


def fuel_assign(flight):
    df = (
        flight.fuelflow()
        .data.assign(fuel=lambda x: x.fuelflow * x.dt)
        .assign(fuel_sum=lambda x: x.fuel.sum())
    )
    return Flight(df)


t_real = t_real.pipe(fuel_assign).eval(6)
df_real = t_real.data
c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
df_all = pd.read_parquet(f"data_generated/opensky2024_clustererd_flights_{map_type}.parquet")
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE

# %%
interpolant = top.tools.interpolant_from_dataframe(df_cost)
import numpy as np


df_costs = []
for fid in flights.fid.unique():

    flightn = flights.query(f"fid=='{fid}'")
    flight0 = flights0.query(f"fid=='{fid}'")
    flightr = df_real.query(f"flight_id=='{fid}'")
    flightr = flightr.assign(h=lambda x: x.altitude * aero.ft)
    costn = interpolant(
        np.stack([flightn.longitude.values, flightn.latitude.values, flightn.h.values])
    )
    costr = interpolant(
        np.stack([flightr.longitude.values, flightr.latitude.values, flightr.h.values])
    )
    cost0 = interpolant(
        np.stack([flight0.longitude.values, flight0.latitude.values, flight0.h.values])
    )
    flightn = (
        flightn.assign(cost=np.array(costn.full())[0])
        .assign(noise=lambda x: x.cost * x.thrust)
        .assign(noise_sum=lambda x: x.noise.sum())
        .assign(cost_sum=lambda x: x.cost.sum())
    )
    flightr = (
        flightr.assign(cost=np.array(costr.full())[0])
        .assign(noise=lambda x: x.cost * x.thrust)
        .assign(noise_sum=lambda x: x.noise.sum())
        .assign(cost_sum=lambda x: x.cost.sum())
    )
    flight0 = (
        flight0.assign(cost=np.array(cost0.full())[0])
        .assign(noise=lambda x: x.cost * x.thrust)
        .assign(noise_sum=lambda x: x.noise.sum())
        .assign(cost_sum=lambda x: x.cost.sum())
    )

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(10, 10))
    ax0.plot(flightr.ts, flightr.fuel, "r", label="Real")
    ax0.plot(flight0.ts, flight0.fuel, "k", label="fuel")
    ax0.plot(flightn.ts, flightn.fuel, "g", label="noiseN")
    ax0.set_ylabel("fuel")
    ax0.set_xlabel("ts")
    ax0.set_title("fuel")
    ax0.legend()
    ax1.plot(flightr.ts, flightr.altitude, "r", label="Real")
    ax1.plot(flight0.ts, flight0.altitude, "k", label="fuel")
    ax1.plot(flightn.ts, flightn.altitude, "g", label="noiseN")
    ax1.set_ylabel("alt")
    ax1.set_xlabel("ts")
    ax1.set_title("alt")
    ax1.legend()
    ax2.plot(flightr.ts, flightr.thrust, "r", label="Real")
    ax2.plot(flight0.ts, flight0.thrust, "k", label="fuel")
    ax2.plot(flightn.ts, flightn.thrust, "g", label="noiseN")
    ax2.set_ylabel("thrust")
    ax2.set_xlabel("ts")
    ax2.set_title("thrust")
    ax2.legend()
    ax3.plot(flightr.longitude, flightr.latitude, "r", label="Real")
    ax3.plot(flight0.longitude, flight0.latitude, "k", label="fuel")
    ax3.plot(flightn.longitude, flightn.latitude, "g", label="noiseN")
    ax3.set_ylabel("longitude")
    ax3.set_xlabel("latitude")
    ax3.set_title("posititon")
    contr = ax3.contourf(
        df_cost.longitude.values.reshape(30, 20, 20)[:, :, 0],
        df_cost.latitude.values.reshape(30, 20, 20)[:, :, 0],
        df_cost.cost.values.reshape(30, 20, 20)[:, :, 6],
        levels=15,
        alpha=0.7,
        cmap="binary",
        # label = "population"
    )
    ax3.legend()
    plt.tight_layout()
    plt.savefig(f"figs/compare_opt_n_f_r_{fid}_{map_type}.png", dpi=100)
    plt.show()
    print([fid] * 3)

    print("flightn", flightn.cost.sum(), flightn.noise.sum(), flightn.fuel.sum())
    print("flightr", flightr.cost.sum(), flightr.noise.sum(), flightr.fuel.sum())
    print("flight0", flight0.cost.sum(), flight0.noise.sum(), flight0.fuel.sum())
    df_costs.append(
        # np.array([fid,
        #     flightr.cost.sum(),
        #     flightn.cost.sum(),
        #     flight0.cost.sum(),
        #     flightr.noise.sum(),
        #     flightn.noise.sum(),
        #     flight0.noise.sum(),
        #     flightr.fuel.sum(),
        #     flightn.fuel.sum(),
        #     flight0.fuel.sum(),
        # ]).T,
        # columns=
        {
            "fid": fid,
            "costs_r": flightr.cost.sum(),
            "costs_n": flightn.cost.sum(),
            "costs_f": flight0.cost.sum(),
            "noises_r": flightr.noise.sum(),
            "noises_n": flightn.noise.sum(),
            "noises_f": flight0.noise.sum(),
            "fuels_r": flightr.fuel_sum.values[0],
            "fuels_n": flightn.fuel.sum(),
            "fuels_f": flight0.fuel.sum(),
        },
    )
# %%
df_costs = pd.DataFrame.from_dict(df_costs)
# %%
df_costs = df_costs.assign(
    noise_nr=lambda x: x.noises_n / x.noises_r,
    noise_nf=lambda x: x.noises_n / x.noises_f,
    fuel_nr=lambda x: x.fuels_n / x.fuels_r,
    fuel_nf=lambda x: x.fuels_n / x.fuels_f,
)
df_costs.to_csv(f"df_costs_{map_type}.csv", index=False)
# %%
from openap import FuelFlow, Thrust

ff = FuelFlow("a320")
thr = Thrust("a320")
fuel = ff.enroute(tow, tas, alt, vs) * ts.diff()
thr.enroute(tow, tas, alt, vs)
noise = cost * thr
# %%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pyproj import Transformer, CRS
from openap import aero, nav
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
from openap import aero, nav, top, prop
from tqdm import tqdm

from traffic.data import navaids, airports

eham = nav.airport("EHAM")
map_type = "N"
flights = pd.read_csv(f"data_generated/flights_noise_opt_{map_type}.csv")
flights0 = pd.read_csv(f"data_generated/flights_fuel_opt_{map_type}.csv")
df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
df_real = (
    pd.read_parquet(f"data_generated/opensky2024_centroids_{map_type}.parquet")
    # .query("flight_id=='VLG12BR_219'")
    # .reset_index(drop=True)
)
max_lola = (df_real.longitude.max() + 0.3, df_real.latitude.max() + 0.3)
min_lola = (df_real.longitude.min() - 0.3, df_real.latitude.min() - 0.3)
pop_map_st = (
    pd.read_parquet(f"data_raw/pop_static.parquet")
    .query("@min_lola[1]<lat<@max_lola[1] and @min_lola[0]<lon<@max_lola[0]")
    .fillna(0)
)
npd = (
    pd.read_csv("data_generated/npds_new_airbus.csv")
    .query("engine_name=='CFM56-5-A1'")
    .drop(columns=["engine_name"])
    .reset_index(drop=True)
)
thrusts = npd.thrust_n.to_numpy()
dists = (
    np.array(npd.rename(columns=lambda x: x[:-3]).columns[:-1].astype(int)) * aero.ft
)
noise_levels = npd.iloc[:, :-1].values

interp_npd = RegularGridInterpolator(
    (thrusts, dists), noise_levels, bounds_error=False, fill_value=None
)
pp_dict = []
for fid in tqdm(flights.fid.unique()):
    flight = flights.query(f"fid=='{fid}'")
    flight = flight.reset_index(drop=True)
    # def distmat(pop_map,flight):
    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)

    pop_lat = pop_map_st.lat.unique()
    pop_lon = pop_map_st.lon.unique()
    Lat2d, Lon2d, Alt2d = np.meshgrid(pop_lat, pop_lon, [0])
    X2d, Y2d = transformer_xy.transform(Lon2d, Lat2d)
    point2d = np.array([X2d, Y2d, Alt2d]).reshape(3, -1).T

    xf, yf = transformer_xy.transform(flight.longitude, flight.latitude)
    point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
    dist = distance_matrix(point3d, point2d)
    dist_0 = np.where(dist > 45000 * aero.ft, -100, dist)

    noise = np.zeros(dist_0.shape)
    for i in range(40):
        thr = flight.thrust.values[i]
        ns = interp_npd(np.array([np.array([thr] * len(dist_0[i])), dist_0[i]]).T)
        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    aggregated_noise = np.sum(noise.reshape(40, len(pop_lon), -1), axis=0)
    # plt.contour(pop_lon,pop_lat,aggregated_noise.T, cmap = "Reds")
    # plt.plot(flight.longitude,flight.latitude)
    population = pop_map_st.pp.values.reshape(len(pop_lat), len(pop_lon))
    affected_pop = np.zeros(dist_0.shape)
    for i in range(40):
        affected_pop[i] = np.where(noise[i] > 50, population.flatten(), 0)

    agg_pop = np.sum(affected_pop.reshape(40, len(pop_lon), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append({"fid": fid, "people_affected": people_affected})

df_pp = pd.DataFrame.from_dict(pp_dict)
df_pp.to_csv(f"pp_affected_noise_opt_{map_type}.csv", index=False)
# %%
pp_dict = []
for fid in tqdm(flights0.fid.unique()):
    flight = flights0.query(f"fid=='{fid}'")
    flight = flight.reset_index(drop=True)
    # def distmat(pop_map,flight):
    crs_3035 = CRS.from_epsg(3035)
    crs_4326 = CRS.from_epsg(4326)
    transformer_xy = Transformer.from_crs(crs_4326, crs_3035, always_xy=True)

    pop_lat = pop_map_st.lat.unique()
    pop_lon = pop_map_st.lon.unique()
    Lat2d, Lon2d, Alt2d = np.meshgrid(pop_lat, pop_lon, [0])
    X2d, Y2d = transformer_xy.transform(Lon2d, Lat2d)
    point2d = np.array([X2d, Y2d, Alt2d]).reshape(3, -1).T

    xf, yf = transformer_xy.transform(flight.longitude, flight.latitude)
    point3d = np.array([xf, yf, flight.h.values]).reshape(3, -1).T
    dist = distance_matrix(point3d, point2d)
    dist_0 = np.where(dist > 45000 * aero.ft, -100, dist)

    noise = np.zeros(dist_0.shape)
    for i in range(40):
        thr = flight.thrust.values[i]
        ns = interp_npd(np.array([np.array([thr] * len(dist_0[i])), dist_0[i]]).T)
        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    aggregated_noise = np.sum(noise.reshape(40, len(pop_lon), -1), axis=0)
    # plt.contour(pop_lon,pop_lat,aggregated_noise.T, cmap = "Reds")
    # plt.plot(flight.longitude,flight.latitude)
    population = pop_map_st.pp.values.reshape(len(pop_lat), len(pop_lon))
    affected_pop = np.zeros(dist_0.shape)
    for i in range(40):
        affected_pop[i] = np.where(noise[i] > 50, population.flatten(), 0)

    agg_pop = np.sum(affected_pop.reshape(40, len(pop_lon), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append({"fid": fid, "people_affected": people_affected})

df_pp = pd.DataFrame.from_dict(pp_dict)
df_pp.to_csv(f"pp_affected_fuel_opt_{map_type}.csv", index=False)
# %%
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE

# flight = flights.query("fid=='VLG49VA_328'")
proj = ccrs.TransverseMercator(
    central_longitude=eham["lon"], central_latitude=eham["lat"]
)

trans = ccrs.PlateCarree()

fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    subplot_kw=dict(projection=proj),
)

ax.add_feature(BORDERS, linestyle="dotted", alpha=0.2)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.2)
# ax.set_title("day/night - " + map_type)
# ax.plot(f_real.longitude, f_real.latitude, transform=trans, c="cyan")
ax.scatter(
    pop_map_st.query("10<pp<2000").lon,
    pop_map_st.query("10<pp<2000").lat,
    c=pop_map_st.query("10<pp<2000").pp,
    cmap="binary",
    s=5,
    transform=trans,
    # levels=15,
    # norm=norm,
    alpha=0.2,
)
norm = plt.Normalize(vmin=50, vmax=100, clip=True)
ax.contour(
    pop_lon,
    pop_lat,
    aggregated_noise.T,
    norm=norm,
    cmap="Reds",
    transform=trans,
    alpha=0.4,
)
ax.plot(flight.longitude, flight.latitude, lw=2, transform=trans, alpha=0.4)
norm = plt.Normalize(vmin=0, vmax=8000, clip=True)
ax.contour(
    pop_lon, pop_lat, agg_pop.T, norm=norm, transform=trans, levels=5, cmap="viridis"
)

plt.show()
# %%
