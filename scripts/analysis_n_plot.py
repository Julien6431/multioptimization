#%%
import numpy as np
import pandas as pd
from pyproj import Transformer, CRS
from openap import aero, nav
from scipy.interpolate import RegularGridInterpolator
from openap import aero, nav, top, prop
from tqdm import tqdm
from traffic.data import navaids, airports
from traffic.core import Flight, Traffic
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import sys
sys.path.append(os.path.abspath("../src/"))
from pp_affected import pp_affected
## Plotting df_cost and pop_maps static
#%%
### Plotting 2 different altitude layers of df_cost 
map_type = "DN"
df_cost = pd.read_csv(f"../data_generated/df_cost_{map_type}.csv")
if map_type=="D" or map_type=="N":
    pop_map = pd.read_parquet(f"../data_raw/{map_type}012011_1K_cropped.parquet").rename(columns = {"popul":"pp"}).fillna(0)
else:
    pop_map = pd.read_parquet("../data_raw/pop_static.parquet")

proj = ccrs.PlateCarree()
trans = ccrs.PlateCarree()
altp = np.linspace(0, 22000, 20)
fig, (ax0, ax1) = plt.subplots(
    1,
    2,
    figsize=(8, 6),
    subplot_kw=dict(projection=proj),
)
for ax in [ax0, ax1]:
    ax.set_extent([3, 6.6, 51, 54])
    ax.add_feature(BORDERS, color="k", alpha=0.3)
    ax.add_feature(COASTLINE, color="k", alpha=0.3)

    norm2 = plt.Normalize(vmin=10, vmax=10000)
    pop_vis = ax.scatter(
        pop_map.query("pp>0").lon.values,
        pop_map.query("pp>0").lat.values,
        c=pop_map.query("pp>0").pp.values,
        s=2,
        transform=trans,
        norm=norm2,
        alpha=1,
        cmap="Reds",
    )
    ax.set_facecolor("linen")

ialt1 = 5
ialt2 = 13
norm = plt.Normalize(vmin=0.0001, vmax=0.084, clip=True)
cost_contour = ax0.contour(
    df_cost.longitude.values.reshape(30, 20, 20)[:, :, 0],
    df_cost.latitude.values.reshape(30, 20, 20)[:, :, 0],
    df_cost.cost.values.reshape(30, 20, 20)[:, :, ialt1],
    levels=20,
    alpha=0.5,
    norm=norm,
    transform=trans,
    cmap="cool_r",
    label="grid_cost",
)

cost_contour2 = ax1.contour(
    df_cost.longitude.values.reshape(30, 20, 20)[:, :, 0],
    df_cost.latitude.values.reshape(30, 20, 20)[:, :, 0],
    df_cost.cost.values.reshape(30, 20, 20)[:, :, ialt2],
    levels=20,
    alpha=0.5,
    norm=norm,
    transform=trans,
    cmap="cool_r",
    label="grid_cost",
)
ax0.set_title(f"a) Altitude: {int(round(altp[ialt1],-2))} ft", fontsize=11)
ax1.set_title(f"b) Altitude: {int(round(altp[ialt2],-2))} ft", fontsize=11)


cbar = plt.colorbar(
    pop_vis,
    ax=ax1,
    orientation="horizontal",
    shrink=0.7,
    aspect=15,
    pad=0.03,
)
cbar.set_ticklabels(["0", "2000", "4000", "6000", "8000", ">10000"])
cbar.ax.set_xlabel("Population density, people/km$^2$", rotation=0, labelpad=5)


cbar2 = plt.colorbar(
    cost_contour,
    ax=ax0,
    orientation="horizontal",
    shrink=0.7,
    aspect=15,
    pad=0.03,
)
cbar2.set_ticks([0, 0.09])
cbar2.set_ticklabels(["Low", "High"])

cbar2.ax.set_xlabel("Grid cost", rotation=0, labelpad=5)

plt.tight_layout()
plt.savefig("../figs/cost_vs_pop.png", bbox_inches="tight", dpi=300)
plt.show()
# %%
### Plotting 4 layers of df_cost
proj = ccrs.PlateCarree()
trans = ccrs.PlateCarree()
altp = np.linspace(0, 22000, 20)
fig = plt.figure(figsize=(8,8.5))
gs = matplotlib.gridspec.GridSpec(90, 80)
ax0 = fig.add_subplot(gs[0:33, 0:37],projection=proj)
ax1 = fig.add_subplot(gs[0:33, 43:80],projection=proj)
ax2 = fig.add_subplot(gs[30:80:, 0:37],projection=proj)
ax3 = fig.add_subplot(gs[30:80:, 43:80],projection=proj)
# fig, ([ax0, ax1],[ax2,ax3]) = plt.subplots(
#     2,
#     2,
#     figsize=(8, 6),
#     subplot_kw=dict(projection=proj),
# )
for ax in [ax0, ax1,ax2,ax3]:
    ax.set_extent([3, 6.6, 50.7, 53.7])
    ax.add_feature(BORDERS, color="k", alpha=0.3)
    ax.add_feature(COASTLINE, color="k", alpha=0.3)

    norm2 = plt.Normalize(vmin=10, vmax=10000)
    pop_vis = ax.scatter(
        pop_map.query("pp>0").lon.values,
        pop_map.query("pp>0").lat.values,
        c=pop_map.query("pp>0").pp.values,
        s=2,
        transform=trans,
        norm=norm2,
        alpha=1,
        cmap="Reds",
    )
    ax.set_facecolor("linen")

for ax,ialt,abcd in zip([ax0, ax1,ax2,ax3],[16,10,5,2],["a","b","c","d"]):
    norm = plt.Normalize(vmin=0.0001, vmax=0.12, clip=True)
    if ialt==2:
        cost_contour = ax.contour(
            df_cost.longitude.values.reshape(30, 20, 20)[:, :, 0],
            df_cost.latitude.values.reshape(30, 20, 20)[:, :, 0],
            df_cost.cost.values.reshape(30, 20, 20)[:, :, ialt],
            levels=25,
            alpha=0.5,
            norm=norm,
            transform=trans,
            cmap="cool_r",
            label="grid_cost",
        )
    else:
        ax.contour(
            df_cost.longitude.values.reshape(30, 20, 20)[:, :, 0],
            df_cost.latitude.values.reshape(30, 20, 20)[:, :, 0],
            df_cost.cost.values.reshape(30, 20, 20)[:, :, ialt],
            levels=20,
            alpha=0.5,
            norm=norm,
            transform=trans,
            cmap="cool_r",
            label="grid_cost",
        )
    ax.set_title(f"{abcd}) Altitude: {int(round(altp[ialt],-2))} ft", fontsize=11)

cbar = plt.colorbar(
    pop_vis,
    ax=ax3,
    orientation="horizontal",
    shrink=0.7,
    aspect=15,
    pad=0.03,
)
cbar.set_ticklabels(["0", "2000", "4000", "6000", ">8000", ">10000"])
cbar.ax.set_xlabel("Population density, people/km$^2$", rotation=0, labelpad=5)


cbar2 = plt.colorbar(
    cost_contour,
    ax=ax2,
    orientation="horizontal",
    shrink=0.7,
    aspect=15,
    pad=0.03,
)
cbar2.set_ticks([0, 0.12])
cbar2.set_ticklabels(["Low", "High"])

cbar2.ax.set_xlabel("Grid cost", rotation=0, labelpad=5)

plt.tight_layout()
plt.savefig("../figs/cost_vs_pop4.png", bbox_inches="tight", dpi=300)
plt.show()
# %%
## Plot clusters and centoirds 
cmap = plt.get_cmap("plasma")
colors = cmap(np.linspace(0, 1, 13))  # Assign unique colors
map_type = "DN"
df_all = pd.read_parquet(f"../data_generated/opensky2024_clustered_flights_{map_type}.parquet")
df_centroids = pd.read_parquet(f"../data_generated/opensky2024_centroids_{map_type}.parquet")
if map_type=="D" or map_type=="N":
    pop_map = pd.read_parquet(f"../data_raw/{map_type}012011_1K_cropped.parquet").rename(columns = {"popul":"pp"}).fillna(0)
else:
    pop_map = pd.read_parquet("../data_raw/pop_static.parquet")
proj = ccrs.PlateCarree()
trans = ccrs.PlateCarree()

fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    subplot_kw=dict(projection=proj),
)
ax.set_extent([3, 6.6, 50.7, 53.7])
ax.add_feature(BORDERS, linestyle="dotted", alpha=0.8)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=0.8)

# norm2 = plt.Normalize(vmin=10, vmax=10000)
# pop_vis = ax.scatter(
#     pop_map.query("pp>0").lon.values,
#     pop_map.query("pp>0").lat.values,
#     c=pop_map.query("pp>0").pp.values,
#     s=2,
#     transform=trans,
#     norm=norm2,
#     alpha=1,
#     cmap="binary",
# )
ax.set_facecolor("lightgrey")
for fid in df_all.flight_id.unique():
    flight = df_all.query("flight_id==@fid")
    ax.plot(
        flight.longitude,
        flight.latitude,
        color=colors[flight.cluster.values[0]],
        lw=2,
        linestyle="dashed",
        alpha=0.2,
        transform=trans,
    )
for fid in df_centroids.flight_id.unique():
    flight = df_centroids.query("flight_id==@fid")
    ax.plot(
        flight.longitude,
        flight.latitude,
        color="w",
        lw=3.5,
        transform=trans,
    )
    ax.plot(
        flight.longitude,
        flight.latitude,
        color="k",
        lw=3.2,
        transform=trans,
    )
    ax.plot(
        flight.longitude,
        flight.latitude,
        color="w",
        lw=2.9,
        transform=trans,
    )
    ax.plot(
        flight.longitude,
        flight.latitude,
        color=colors[flight.cluster.values[0]],
        lw=2.6,
        transform=trans,
    )
plt.tight_layout()
plt.savefig("../figs/clusters_n_centroids.png", bbox_inches="tight", dpi=200)
plt.show()
# %%
## Fuel vs Noise
from openap.casadi import aero as aero_casadi
from openap import top

ac = "a320"
m0 = 0.8
eham =nav.airport("EHAM")
start = (eham["lat"], eham["lon"])
# end = (navaids["KEKIX"].latitude + 0.3, navaids["KEKIX"].longitude + 0.75)
end = (51.25555556, 4.066388889)
nodes = 30
h_end = 22_000 * aero.ft
flights = None
df_cost = pd.read_csv("../data_generated/df_cost_DN.csv")
optimizer = top.Climb(ac, start, end, m0=m0)
optimizer.setup(nodes=nodes, max_iteration=1000)
flights = optimizer.trajectory(
    objective="fuel",
    h_end=h_end,
).assign(fid = "id4").assign(num=0)
# for i,number in tqdm(enumerate([10,5,1, 0.004, 0.001, 0.0004])):
for i,number in tqdm(enumerate([5,1, 0.004, 0.001])):
    def obj_noise(x, u, dt, **kwargs):
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]
        tas = aero_casadi.mach2tas(mach, h)
        thrust = optimizer.drag.clean(m, tas, h / aero.ft, vs / aero.fpm)
        cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=True, **kwargs)
        fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
        return number * cost * thrust + fuel

    optimizer = top.Climb(ac, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=1000, debug=False)

    interpolant = top.tools.interpolant_from_dataframe(df_cost)
    flight = optimizer.trajectory(
        objective=obj_noise,
        interpolant=interpolant,
        h_end=h_end,
    ).assign(fid=f"id{i}").assign(num=number)
    flights = pd.concat([flights, flight])

cost = interpolant(
    np.array([flights.longitude.values, flights.latitude.values, flights.h.values])
)
flights = flights.assign(cost_grid=cost.full()[0]).sort_values(by=["num","ts"])
flights.to_csv("../data_generated/fligths_spectrum.csv",index=False)
# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.cm as cm
flights = pd.read_csv("../data_generated/fligths_spectrum.csv")
df_cost = pd.read_csv("../data_generated/df_cost_DN.csv")
eham = nav.airport("EHAM")
proj = ccrs.PlateCarree()  # TransverseMercator(
#     central_longitude=eham["lon"], central_latitude=eham["lat"]
# )
trans = ccrs.PlateCarree()
plot_extent = [
    min(flights.longitude.values) - 0.3 - 0.15,
    max(flights.longitude.values) + 0.3 + 0.15,
    min(flights.latitude.values) - 0.3 - 0.15,
    max(flights.latitude.values) + 0.3 + 0.15,
]
fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    subplot_kw=dict(projection=proj),
)
ax.set_extent(plot_extent)
ax.add_feature(BORDERS, linestyle="dotted", alpha=1)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=1)


df_c = df_cost  # .assign(cost=lambda x: np.where(x.cost > 0.8, 0.8, x.cost)).assign(cost=lambda x: np.where(x.cost < 0.01, 0.01, x.cost))


norm = plt.Normalize(vmin=0.0001, vmax=0.06)
contr = ax.contourf(
    df_cost.longitude.values.reshape(30, 20, 20)[:, :, 0],
    df_cost.latitude.values.reshape(30, 20, 20)[:, :, 0],
    df_cost.cost.values.reshape(30, 20, 20)[:, :, 6],
    levels=25,
    alpha=0.4,
    norm=norm,
    transform=trans,
    cmap="binary",
    # label = "population"
)
props = dict(boxstyle="round", facecolor="w")
ax.text(
    eham["lon"] + 0.06,
    eham["lat"] + 0.04,
    "EHAM",
    transform=trans,
    verticalalignment="top",
    bbox=props,
    alpha=0.8,
)


cmap = plt.get_cmap("viridis")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "narrow_viridis", cmap(np.linspace(0.3, 0.8, 256))
)

colors_v = cmap(np.linspace(0, 1, len(flights.num.unique())))  # Assign unique colors
norm_fid = matplotlib.colors.Normalize(vmin=0, vmax=5)  # Normalize for color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm_fid)  # For colorbar
sm.set_array([])

for i, num in enumerate(flights.num.unique()):
    flight = flights.query("num==@num")
    # flight0= flights0.query("num==@num")

    ax.plot(
        flight.longitude,
        flight.latitude,
        color=colors_v[(i)],
        lw=2,
        linestyle="dashed",
        transform=trans,
        label=f"Noise optimal{num}",
    )
    ax.plot(
        flight.query("cost_grid>0").longitude,
        flight.query("cost_grid>0").latitude,
        color=colors_v[(i)],
        lw=2,
        # linestyle="dashed",
        transform=trans,
        # label=f"Noise optimal{num}",
    )



cbar = fig.colorbar(
    sm, ax=ax, orientation="horizontal", shrink=0.5, fraction=0.03, pad=0.02
)
cbar.set_ticks([0, 5])
cbar.set_ticklabels(["Fuel optimal","Noise optimal"])
# cbar.set_label("Flight ID")
# ax.legend()


plt.tight_layout()
plt.savefig("../figs/noise_fuel_spectrum_flights.png", bbox_inches="tight")
plt.show()
# %%

# %%
ac = "a320"
pop_map = pd.read_parquet("../data_raw/pop_static.parquet")
grid_type ='ll'
treshold=45
table = []
for num in flights.num.unique():
    flight = flights.query("num==@num")
    flight0 = flights.query("num==0")
    pp_all,pp_uni, f = pp_affected(ac, pop_map,grid_type, flight,treshold)
    pp_all0,pp_uni0, f0 = pp_affected(ac, pop_map,grid_type, flight0,treshold)
    flight = flight.assign(noise = lambda x: x.cost_grid*x.thrust)
    flight0 = flight0.assign(noise = lambda x: x.cost_grid*x.thrust)
    table.append({
        "id": flight.fid.unique()[0],
        "num": flight.num.unique()[0],
        "fuel":f.fuel.sum()/f0.fuel.sum(),
        "cost":flight.cost_grid.sum()/flight0.cost_grid.sum(),
        "noise":flight.noise.sum()/flight0.noise.sum(),
        "pp_all":pp_all/pp_all0,
        "pp_uni": pp_uni/pp_uni0,
    })
print(treshold)
table =pd.DataFrame.from_dict(table)
table
# %%
# %%
## Fuel vs Noise - Experimenting
#####################################################################

#####################################################################

#####################################################################
import numpy as np
import pandas as pd
from pyproj import Transformer, CRS
from openap import aero, nav
from scipy.interpolate import RegularGridInterpolator
from openap import aero, nav, top, prop
from tqdm import tqdm
from traffic.data import navaids, airports
from traffic.core import Flight, Traffic
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs

from cartopy.feature import BORDERS, COASTLINE
import sys
sys.path.append(os.path.abspath("../src/"))
from pp_affected import pp_affected

#####################################################################


from openap.casadi import aero as aero_casadi
from openap import top

ac = "a320"
m0 = 0.8
eham =nav.airport("EHAM")
start = (eham["lat"], eham["lon"])
# end = (navaids["KEKIX"].latitude + 0.3, navaids["KEKIX"].longitude + 0.75)
end = (51.05555556, 4.066388889)
nodes = 30
h_end = 25_000 * aero.ft
flights = None
df_cost = pd.read_csv("../data_generated/df_cost_DN.csv")
# df_cost["cost"] = df_cost["cost"]/1000
optimizer = top.Climb(ac, start, end, m0=m0)
optimizer.setup(nodes=nodes, max_iteration=1000)
flights = optimizer.trajectory(
    objective="fuel",
    h_end=h_end,
).assign(fid = "id8").assign(num=0)
# for i,number in tqdm(enumerate([10,5,1, 0.004, 0.001, 0.0004])):
for i,number in tqdm(enumerate([10, 5,1, 0.009, 0.004, 0.001,0.0004,0.0001])):
    def obj_noise(x, u, dt, **kwargs):
        xp, yp, h, m, ts = x[0], x[1], x[2], x[3], x[4]
        mach, vs, psi = u[0], u[1], u[2]
        tas = aero_casadi.mach2tas(mach, h)
        thrust = optimizer.drag.clean(m, tas, h / aero.ft, vs / aero.fpm)
        cost = optimizer.obj_grid_cost(x, u, dt, n_dim=3, time_dependent=True, **kwargs)
        fuel = optimizer.obj_fuel(x, u, dt, **kwargs)
        return number * cost * thrust + fuel

    optimizer = top.Climb(ac, start, end, m0=m0)
    optimizer.setup(nodes=nodes, max_iteration=1000, debug=False)

    interpolant = top.tools.interpolant_from_dataframe(df_cost)
    flight = optimizer.trajectory(
        objective=obj_noise,
        interpolant=interpolant,
        h_end=h_end,
    ).assign(fid=f"id{i}").assign(num=number)
    flights = pd.concat([flights, flight])

cost = interpolant(
    np.array([flights.longitude.values, flights.latitude.values, flights.h.values])
)
flights = flights.assign(cost_grid=cost.full()[0]).sort_values(by=["num","ts"])
flights.to_csv("../data_generated/fligths_spectrum_exp.csv",index=False)
# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE
import matplotlib.cm as cm
flights = pd.read_csv("../data_generated/fligths_spectrum_exp.csv")
df_cost = pd.read_csv("../data_generated/df_cost_DN.csv")
# df_cost["cost"] = df_cost["cost"]/1000
eham = nav.airport("EHAM")
proj = ccrs.PlateCarree()  # TransverseMercator(
#     central_longitude=eham["lon"], central_latitude=eham["lat"]
# )
trans = ccrs.PlateCarree()
plot_extent = [
    min(flights.longitude.values) - 0.3 - 0.15,
    max(flights.longitude.values) + 0.3 + 0.15,
    min(flights.latitude.values) - 0.3 - 0.15,
    max(flights.latitude.values) + 0.3 + 0.15,
]
fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6),
    subplot_kw=dict(projection=proj),
)
ax.set_extent(plot_extent)
ax.add_feature(BORDERS, linestyle="dotted", alpha=1)
ax.add_feature(COASTLINE, linestyle="dotted", alpha=1)


df_c = df_cost  # .assign(cost=lambda x: np.where(x.cost > 0.8, 0.8, x.cost)).assign(cost=lambda x: np.where(x.cost < 0.01, 0.01, x.cost))


norm = plt.Normalize(vmin=0.0001, vmax=0.06)
contr = ax.contourf(
    df_cost.longitude.values.reshape(30, 20, 20)[:, :, 0],
    df_cost.latitude.values.reshape(30, 20, 20)[:, :, 0],
    df_cost.cost.values.reshape(30, 20, 20)[:, :, 8],
    levels=25,
    alpha=0.4,
    norm=norm,
    transform=trans,
    cmap="binary",
    # label = "population"
)
props = dict(boxstyle="round", facecolor="w")
ax.text(
    eham["lon"] + 0.06,
    eham["lat"] + 0.04,
    "EHAM",
    transform=trans,
    verticalalignment="top",
    bbox=props,
    alpha=0.8,
)


cmap = plt.get_cmap("viridis")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "narrow_viridis", cmap(np.linspace(0.01, 0.99, 256))
)

colors_v = cmap(np.linspace(0, 1, len(flights.num.unique())))  # Assign unique colors
norm_fid = matplotlib.colors.Normalize(vmin=0, vmax=8)  # Normalize for color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm_fid)  # For colorbar
sm.set_array([])
# ax.plot(
#     flight0.longitude,
#     flight0.latitude,
#     color=colors_v[4],
#     lw=2,
#     linestyle="dashed",
#     transform=trans,
#     label=f"Fuel optimal",
#     # alpha=0.8,
# )
for i, num in enumerate(flights.num.unique()[:]):
    flight = flights.query("num==@num")
    # flight0= flights0.query("num==@num")

    ax.plot(
        flight.longitude,
        flight.latitude,
        color=colors_v[(i)],
        lw=2,
        linestyle="dashed",
        transform=trans,
        label=f"Noise optimal{num}",
    )
    ax.plot(
        flight.query("cost_grid>0").longitude,
        flight.query("cost_grid>0").latitude,
        color=colors_v[(i)],
        lw=2,
        # linestyle="dashed",
        transform=trans,
        # label=f"Noise optimal{num}",
    )



cbar = fig.colorbar(
    sm, ax=ax, orientation="horizontal", shrink=0.5, fraction=0.03, pad=0.02
)
cbar.set_ticks([0, 5])
cbar.set_ticklabels(["Fuel optimal","Noise optimal"])
# cbar.set_label("Flight ID")
# ax.legend()


plt.tight_layout()
# plt.savefig("../figs/noise_fuel_spectrum_flights.png", bbox_inches="tight")
plt.show()
# %%
ac = "a320"
pop_map = pd.read_parquet("../data_raw/pop_static.parquet")
grid_type ='ll'
table = []
treshold=45  
print(treshold)
for i,num in enumerate(flights.num.unique()):
    flight = flights.query("num==@num")
    flight0 = flights.query("num==0")
    pp_all,pp_uni, f = pp_affected(ac, pop_map,grid_type, flight,treshold)
    pp_all0,pp_uni0, f0 = pp_affected(ac, pop_map,grid_type, flight0,treshold)
    table.append({
        "i":i,
        "id": flight.fid.unique()[0],
        "num": flight.num.unique()[0],
        "fuel":f.fuel.sum()/f0.fuel.sum(),
        "cost":flight.cost_grid.sum()/flight0.cost_grid.sum(),
        "pp_all":pp_all/pp_all0,
        "pp_uni": pp_uni/pp_uni0,
    })
table =pd.DataFrame.from_dict(table)
table
# %%

cmap = plt.get_cmap("viridis")
colors_v = cmap(np.linspace(0, 1, 101))  # Assign unique colors
norm_fid = matplotlib.colors.Normalize(vmin=0, vmax=100)  # Normalize for color mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm_fid)  # For colorbar
sm.set_array([])

l = []
fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 6)
)
for treshold in np.linspace(30,75,10):
    print("treshold:",treshold)
    flights = flights.sort_values(by=["fid","ts"])
    table = []

    for i,num in enumerate(flights.num.unique()):
        flight = flights.query("num==@num")
        flight0 = flights.query("num==0")
        pp_all,pp_uni, f = pp_affected(ac, pop_map,grid_type, flight,treshold)
        pp_all0,pp_uni0, f0 = pp_affected(ac, pop_map,grid_type, flight0,treshold)
        table.append({
            "i":i,
            "id": flight.fid.unique()[0],
            "num": flight.num.unique()[0],
            "fuel":f.fuel.sum()/f0.fuel.sum(),
            "cost":flight.cost_grid.sum()/flight0.cost_grid.sum(),
            "pp_all":pp_all/pp_all0,
            "pp_uni": pp_uni/pp_uni0,
        })
    table =pd.DataFrame.from_dict(table)

    pl =ax.plot(table.i,table.pp_all,c=colors_v[(int(treshold))],label=str(treshold)+"dB(A)")
    # print(table)
cbar = fig.colorbar(
    sm, ax=ax, orientation="horizontal", shrink=0.5, fraction=0.03, pad=0.02
)
plt.show()
# plt.legend()

# %%
