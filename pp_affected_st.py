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
idd = "DN"
treshold = 75  # db
flights = pd.read_csv(f"data_generated/opsk_flights_noise_realistic_{idd}.csv")
flights0 = pd.read_csv(f"data_generated/opsk_flights0_fuel_realistic_{idd}.csv")
df_cost = pd.read_csv(f"data_generated/df_cost_{idd}_2024.csv")
df_real = (
    pd.read_parquet(f"data_generated/opensky2024_centroids_{idd}.parquet")
    # .query("flight_id=='VLG12BR_219'")
    # .reset_index(drop=True)
)
max_lola = (df_real.longitude.max() + 0.3, df_real.latitude.max() + 0.3)
min_lola = (df_real.longitude.min() - 0.3, df_real.latitude.min() - 0.3)
pop_map_st = (
    pd.read_parquet(f"data_raw/pop_st.parquet")
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

# %%
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
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

    mask = affected_pop != 0

    # Get a single representative layer with unique values at each position
    unique_values = np.where(mask.any(axis=0), affected_pop.max(axis=0), 0)

    # Sum the unique values
    result = unique_values.sum()

    agg_pop = np.sum(affected_pop.reshape(40, len(pop_lon), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append(
        {
            "fid": fid,
            "fuel": flight.fuel.sum(),
            "cost_grid": flight.cost_grid.sum(),
            "people_affected_sum_all": people_affected,
            "pp_unique_sum": result,
        }
    )


df_pp = pd.DataFrame.from_dict(pp_dict)
df_pp.to_csv(
    f"data_generated/pp_affected_{treshold}/pp_affected_noise_opt_{idd}.csv",
    index=False,
)
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
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

    mask = affected_pop != 0

    # Get a single representative layer with unique values at each position
    unique_values = np.where(mask.any(axis=0), affected_pop.max(axis=0), 0)

    # Sum the unique values
    result = unique_values.sum()

    agg_pop = np.sum(affected_pop.reshape(40, len(pop_lon), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append(
        {
            "fid": fid,
            "fuel": flight.fuel.sum(),
            "cost_grid": flight.cost_grid.sum(),
            "people_affected_sum_all": people_affected,
            "pp_unique_sum": result,
        }
    )

df_pp = pd.DataFrame.from_dict(pp_dict)
df_pp.to_csv(
    f"data_generated/pp_affected_{treshold}/pp_affected_fuel_opt_{idd}.csv", index=False
)
# %%
pp_dict = []
for fid in tqdm(df_real.flight_id.unique()):
    flight = df_real.query(f"flight_id=='{fid}'")
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
    point3d = np.array([xf, yf, flight.altitude.values * aero.ft]).reshape(3, -1).T
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
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)

    mask = affected_pop != 0

    # Get a single representative layer with unique values at each position
    unique_values = np.where(mask.any(axis=0), affected_pop.max(axis=0), 0)

    # Sum the unique values
    result = unique_values.sum()

    agg_pop = np.sum(affected_pop.reshape(40, len(pop_lon), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append(
        {
            "fid": fid,
            "fuel": flight.fuel.sum(),
            # "cost_grid": flight.cost_grid.sum(),
            "people_affected_sum_all": people_affected,
            "pp_unique_sum": result,
        }
    )

df_pp = pd.DataFrame.from_dict(pp_dict)
df_pp.to_csv(
    f"data_generated/pp_affected_{treshold}/pp_affected_real_{idd}.csv", index=False
)
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
# ax.set_title("day/night - " + idd)
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
    alpha=0.6,
)
ax.plot(flight.longitude, flight.latitude, lw=2, transform=trans, alpha=0.5)
norm = plt.Normalize(vmin=0, vmax=8000, clip=True)
ax.contour(
    pop_lon, pop_lat, agg_pop.T, norm=norm, transform=trans, levels=5, cmap="viridis"
)

plt.show()
# %%
flights = pd.read_csv(f"data_generated/flights_spectr.csv")
flights0 = pd.read_csv(f"data_generated/flights0_spectr.csv").query("fid==0")
flights0 = flights0.assign(fid="fuel")
flights = pd.concat([flights, flights0])
flights["fid"] = flights["fid"].astype(str)
# %%
treshold = 50
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
    for i in range(31):
        thr = flight.thrust.values[i]
        ns = interp_npd(np.array([np.array([thr] * len(dist_0[i])), dist_0[i]]).T)
        noise[i] = np.where(dist_0[i] < 0, 0, ns)
    aggregated_noise = np.sum(noise.reshape(31, len(pop_lon), -1), axis=0)
    # plt.contour(pop_lon,pop_lat,aggregated_noise.T, cmap = "Reds")
    # plt.plot(flight.longitude,flight.latitude)
    population = pop_map_st.pp.values.reshape(len(pop_lat), len(pop_lon))
    affected_pop = np.zeros(dist_0.shape)
    for i in range(31):
        affected_pop[i] = np.where(noise[i] > treshold, population.flatten(), 0)
    mask = affected_pop != 0

    # Get a single representative layer with unique values at each position
    unique_values = np.where(mask.any(axis=0), affected_pop.max(axis=0), 0)

    # Sum the unique values
    result = unique_values.sum()

    agg_pop = np.sum(affected_pop.reshape(31, len(pop_lon), -1), axis=0)
    people_affected = sum(sum(agg_pop))
    pp_dict.append(
        {
            "fid": fid,
            "fuel": flight.fuel.sum(),
            "cost_grid": flight.cost_grid.sum(),
            "people_affected_sum_all": people_affected,
            "pp_unique_sum": result,
        }
    )

df_pp = pd.DataFrame.from_dict(pp_dict)

df_pp
# %%
df_pp = df_pp.assign(
    fuel_r=lambda x: x.fuel / df_pp.fuel.values[-1],
    cost_grid_r=lambda x: x.cost_grid / df_pp.cost_grid.values[-1],
    pp_all_r=lambda x: x.people_affected_sum_all
    / df_pp.people_affected_sum_all.values[-1],
    pp_uni_r=lambda x: x.pp_unique_sum / df_pp.pp_unique_sum.values[-1],
)
df_pp.to_csv("data_generated/df_pp_spectr.csv", index=False)
# %%
print(df_pp[["fuel_r", "cost_grid_r", "pp_uni_r"]].to_latex())
# \begin{tabular}{lrr}
# \toprule
#     id & Fuel spent (kg) & People affected \
#     \midrule
# id0 & 1.137 & 0.813 & 1.177 \\
# id1 & 1.074 & 0.820 & 1.194 \\
# id2 & 1.030 & 0.853 & 1.174 \\
# id3 & 1.005 & 0.903 & 1.225 \\
# id4 & 1.000 & 1.000 & 1.000 \\
#     \bottomrule
# \end{tabular}

# \begin{tabular}{llrrrrr}
# \toprule
#     obj & Fuel spent (kg) & $vs_{mean}$ (ft/min) & $Thrust_{mean}$ (N) & $TAS{mean}$ (kts) & People Affected \\
#     \midrule
#     noise   & 813.66 & 2,483.8 & 72,482 & 471.65 & 3343508 \\
#     cdn = 5 & 816.94 & 2,483.8 & 72,828 & 467.96 & 3370702 \\
#     cdn = 15 & 820.58 & 2,500.0 & 74,050 & 457.58 & 3470045 \\
#     cdn = 50 & 835.00 & 2,500.0 & 75,654 & 441.35 & 3774080 \\
#     fuel    & 844.72 & 2,500.0 & 76,850 & 433.18 & 4644536 \\
#     \bottomrule
# \end{tabular}

# %%
