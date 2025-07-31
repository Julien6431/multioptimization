# %%
from traffic.core import Traffic, Flight
from traffic.data import airports
from openap import Thrust, FuelFlow, aero, prop, top
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

import sys

sys.path.append(os.path.abspath("src/"))
from pp_affected import pp_affected

idd = "DN"
t_clustered = Traffic(
    pd.read_parquet(
        f"data_generated/opensky2024_clustered_flights_{idd}.parquet"
    ).rename(columns={"heading": "track", "groundspeed": "tas", "flight_id": "fid"})
)
t_cnts = Traffic(
    pd.read_parquet(f"data_generated/opensky2024_centroids_{idd}.parquet").rename(
        columns={"heading": "track", "groundspeed": "tas", "flight_id": "fid"}
    )
)
n_clusters = t_clustered.data.cluster.max() + 1

# %%

from itertools import islice, cycle
from cartes.utils.features import countries
from cartes.crs import Amersfoort

color_cycle = cycle(
    "#a6cee3 #1f78b4 #b2df8a #33a02c #fb9a99 #e31a1c "
    "#fdbf6f #ff7f00 #cab2d6 #6a3d9a #ffff99 #b15928".split()
)
colors = list(islice(color_cycle, n_clusters))

with plt.style.context("traffic"):
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(projection=Amersfoort()))
    ax.add_feature(countries())

    for cluster in range(n_clusters):
        # if a := t_clustered.query(f"cluster == {cluster}"):
        # a.plot(ax, ls="--", color=colors[cluster], alpha=0.4)
        if a := t_cnts.query(f"cluster == {cluster}"):
            a.plot(ax, lw=3.5, color="w")
            a.plot(ax, lw=3, color="k")
            a.plot(ax, lw=2, color=colors[cluster], label=f"{cluster}")
    ax.set_global()
    ax.set_title(f"Traffic per cluster")
    ax.legend()
plt.show()

# %%

import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE

map_type = "DN"
df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
if map_type == "D" or map_type == "N":
    pop_map = (
        pd.read_parquet(f"data_raw/{map_type}012011_1K_cropped.parquet")
        .rename(columns={"popul": "pp"})
        .fillna(0)
    )
else:
    pop_map = pd.read_parquet("data_raw/pop_static.parquet")

proj = ccrs.PlateCarree()
trans = ccrs.PlateCarree()

# %%

df_cost = pd.read_csv(f"data_generated/df_cost_{idd}.csv")
interpolant = top.tools.interpolant_from_dataframe(df_cost)

fuel_opt = pd.read_csv(f"data_generated/flights_fuel_opt_{idd}.csv")
noise_opt = pd.read_csv(f"data_generated/flights_noise_opt_{idd}.csv")

cost = interpolant(
    np.array(
        [
            t_clustered.data.longitude.values,
            t_clustered.data.latitude.values,
            t_clustered.data.h.values,
        ]
    )
)
t_clustered = t_clustered.assign(cost_grid=cost.full()[0])

cost = interpolant(
    np.array(
        [
            t_cnts.data.longitude.values,
            t_cnts.data.latitude.values,
            t_cnts.data.h.values,
        ]
    )
)
t_cnts = t_cnts.assign(cost_grid=cost.full()[0])

# %%

for f in t_cnts:
    plt.plot(f.data.cost_grid.to_numpy().flatten())
plt.show()


# %%

from matplotlib.collections import LineCollection
from cartopy.crs import PlateCarree


def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, transform=PlateCarree(), **lc_kwargs)
    lc.set_array(c)
    return ax.add_collection(lc)


def colored_traffic(traff, ax, **lc_kwargs):
    cost = np.empty(0)
    segments = np.empty((0, 2, 2))
    for fid in traff.fid.unique():
        if fid not in ["EIN84X_1515", "EJU42KM_1219", "SAS62L_1257", "VLG12BR_219"]:
            f = traff.query("fid==@fid").reset_index(drop=True)
            x = f.longitude.to_numpy().flatten()
            y = f.latitude.to_numpy().flatten()
            c = f.cost_grid.to_numpy().flatten()
            _, _, _, c = pp_affected(ac, pop_map, "ll", f, treshold)
            c = np.sum(c, axis=1)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            new_segments = np.concatenate([points[:-1], points[1:]], axis=1)
            segments = np.concatenate((segments, new_segments), axis=0)
            cost = np.concatenate((cost, c[:-1]))

    lc = LineCollection(segments, transform=PlateCarree(), **lc_kwargs)
    lc.set_array(cost)

    return ax.add_collection(lc)


# %%
with plt.style.context("traffic"):
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(projection=Amersfoort()))

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
    ax.add_feature(countries())

    for f in t_cnts:
        x = f.data.longitude.to_numpy().flatten()
        y = f.data.latitude.to_numpy().flatten()
        c = (
            f.data.cost_grid.to_numpy().flatten()
        )  # * f.data.thrust.to_numpy().flatten()

        line = colored_line_between_pts(x, y, c, ax, lw=2, cmap="viridis")
    # line = colored_traffic(t_cnts.data, ax, lw=2, cmap="viridis")
    fig.colorbar(line, ax=ax, label="cost")
    ax.set_title("Grid cost")
    ax.set_global()

plt.show()

# %%

ac = "a320"
treshold = 50

with plt.style.context("traffic"):
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(projection=Amersfoort()))

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
    ax.add_feature(countries())

    line = colored_traffic(t_cnts.data, ax, ls="--", lw=2, cmap="hot")
    # line = colored_traffic(noise_opt, ax, ls="--", lw=2, cmap="hot")

    fig.colorbar(line, ax=ax, label="Number of poeple")
    ax.set_title(f"Number of poeple affected by a noise greater than {treshold}dB")
    ax.set_global()

plt.show()

# %%

fig, ax = plt.subplots(2, 6, figsize=(15, 6))
cost = np.empty(0)
for i in range(len(noise_opt.fid.unique()) - 1):
    fid = noise_opt.fid.unique()[i]
    # if fid not in ["EJU42KM_1219", "SAS62L_1257"]:
    f = noise_opt.query("fid==@fid").reset_index(drop=True)
    f_cnts = t_cnts.data.query("fid==@fid").reset_index(drop=True)
    cluster = f_cnts.cluster.values[0]
    # c = f.cost_grid.values
    _, _, _, c = pp_affected(ac, pop_map, "ll", f, treshold)
    c = np.sum(c, axis=1)
    ax[i // 6, i % 6].plot(f.ts, c, color="orange")
    _, _, _, c = pp_affected(ac, pop_map, "ll", f_cnts, treshold)
    c = np.sum(c, axis=1)
    # c = f_cnts.cost_grid.values
    ax[i // 6, i % 6].plot(f_cnts.ts, c, color="blue")
    ax[i // 6, i % 6].set_title(f"{cluster}")

plt.show()

# %% Plot metrics per cluster

threshold = 50
pp_affected_pd = pd.read_csv(f"data_generated/pp_affected_{threshold}.csv")

pp_affected_DN = pp_affected_pd.query("map_type=='DN'")
fids = pp_affected_DN.fid.unique()
groups = t_cnts.data.groupby("fid")

clusters = []

for _, df in groups:
    clusters.append(str(df.cluster.values[0]))
clusters = np.array(clusters)

fig, ax = plt.subplots(3, 1, figsize=(15, 20))

x = np.arange(len(pp_affected_DN.fid.unique()))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for i, col in enumerate(["cost_grid", "people_affected_sum_all", "pp_unique_sum"]):
    multiplier = 0
    for frn in ["Fuel", "Noise"]:
        f_type = frn[0].lower()
        measurement = (
            pp_affected_DN.query("frn== @f_type")[col].values
            / pp_affected_DN.query("frn== 'r'")[col].values
        )
        offset = width * multiplier
        rects = ax[i].bar(x + offset, measurement, width, label=frn)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    xlims = ax[i].get_xlim()
    ax[i].plot(xlims, [1, 1], color="r", lw=3)

    ax[i].set_ylabel("ratio")
    ax[i].set_title(col)
    ax[i].set_xticks(x + width, fids + "\n Cluster " + clusters, fontsize=7)
    ax[i].legend(loc="upper left", ncols=3)
    ax[i].grid()
    ax[i].set_xlim(xlims)
fig.savefig(f"figs/ratios{threshold}.png", bbox_inches="tight")
plt.show()

# %% Plot time spend in critical zone

fig, ax = plt.subplots(figsize=(15, 6))

x = np.arange(len(fuel_opt.fid.unique()))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
for df, frn in zip([fuel_opt, noise_opt, t_cnts.data], ["Fuel", "Noise", "Real"]):
    measurement = df.query("cost_grid>0").groupby("fid").ts.max()
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=frn)
    multiplier += 1

ax.set_ylabel("Time (s)")
ax.set_title("Time spend in critical zone per cluster")
ax.set_xticks(x + width, fids + "\n Cluster " + clusters, fontsize=7)
ax.legend(loc="upper left", ncols=3)
ax.grid()
fig.savefig(f"figs/time_spend_costgrid.png", bbox_inches="tight")
plt.show()

# %% Plot altitude

fig, ax = plt.subplots(figsize=(15, 6))

x = np.arange(len(fuel_opt.fid.unique()))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
for fid in fuel_opt.fid.unique():
    df = noise_opt.query("fid == @fid")
    ax.plot(df.ts, df.thrust, color="C1")
    df = fuel_opt.query("fid == @fid")
    ax.plot(df.ts, df.thrust, color="C0")
    df = t_cnts.data.query("fid == @fid")
    ax.plot(df.ts, df.thrust, color="C2")

ax.set_ylabel("Thrust")
ax.set_title("Thrust per trajectory")
ax.grid()
plt.show()


# %%
from traffic.data import navaids
from ipywidgets import Layout

longitude = t_cnts.data.longitude
latitude = t_cnts.data.latitude
lon, LON = 4, 5.3  # longitude.min(),longitude.max()
lat, LAT = 51.9, 52.5  # latitude.min(),latitude.max()
navaids_EHAM = navaids.data.query("@lon<longitude<@LON and @lat<latitude<@LAT")

m = t_cnts.map_leaflet(zoom=10, layout=Layout(height="1000px", max_width="1000px"))
m.add(navaids["EH029"])
m.add(navaids["EH073"])
m.add(navaids["EH050"])
m.add(navaids["LOPIK"])
m.add(navaids["LEKKO"])
m.add(navaids["EH005"])
m
# points = navaids_EHAM[["longitude", "latitude"]].values

# %%

color_cycle = cycle(
    "#a6cee3 #1f78b4 #b2df8a #33a02c #fb9a99 #e31a1c "
    "#fdbf6f #ff7f00 #cab2d6 #6a3d9a #ffff99 #b15928".split()
)
colors = list(islice(color_cycle, n_clusters))

with plt.style.context("traffic"):
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(projection=Amersfoort()))
    ax.add_feature(countries())

    for cluster in range(n_clusters):
        # if a := t_clustered.query(f"cluster == {cluster}"):
        # a.plot(ax, ls="--", color=colors[cluster], alpha=0.4)
        if a := t_cnts.query(f"cluster == {cluster}"):
            a.plot(ax, lw=3.5, color="w")
            a.plot(ax, lw=3, color="k")
            a.plot(ax, lw=2, color=colors[cluster], label=f"{cluster}")
    ax.scatter(*points.T, transform=PlateCarree(), s=2)
    ax.set_global()
    ax.set_title(f"Traffic per cluster")
    ax.legend()
plt.show()
# %%
