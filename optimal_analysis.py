# %%
import matplotlib.pyplot as plt
from cartes.utils.features import countries
from cartes.crs import Amersfoort
from cartopy.crs import PlateCarree
import numpy as np
import pandas as pd
import json
from pitot import geodesy as geo
from traffic.data import navaids, airports
from optimisation_waypoints import (
    plot_map,
    get_trajectory,
    obj_function,
    get_fid_params,
)
from traffic.core import Traffic

pd.set_option("display.max_rows", 15)

# %%

map_type = "DN"

if map_type == "D" or map_type == "N":
    pop_map = (
        pd.read_parquet(f"data_raw/{map_type}012011_1K_cropped.parquet")
        .rename(columns={"popul": "pp"})
        .fillna(0)
    )
else:
    pop_map = pd.read_parquet("data_raw/pop_static.parquet")


fuel_opt = pd.read_csv(f"data_generated/flights_fuel_opt_{map_type}.csv")
noise_opt = pd.read_csv(f"data_generated/flights_noise_opt_{map_type}.csv")
flights_waypoints = pd.read_csv(
    f"data_generated/waypoints/flights_existing_procedure_{map_type}.csv"
)
noise_opt_waypoints = pd.read_csv(
    f"data_generated/waypoints/flights_noise_opt_{map_type}.csv"
)

with open("data_generated/waypoints/current_waypoints.json", "r") as fp:
    fid_waypoints = json.load(fp)
with open("data_generated/waypoints/optimal_waypoints.json", "r") as fp:
    fid_waypoints_opt = json.load(fp)

# %%

with plt.style.context("traffic"):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection=Amersfoort()))
    ax.add_feature(countries())

    norm2 = plt.Normalize(vmin=10, vmax=10000)
    ax.scatter(
        pop_map.query("pp>0").lon.values,
        pop_map.query("pp>0").lat.values,
        c=pop_map.query("pp>0").pp.values,
        s=2,
        transform=PlateCarree(),
        norm=norm2,
        alpha=1,
        cmap="Reds",
    )

    for fid in fid_waypoints.keys():
        f0 = flights_waypoints.query("fid==@fid").reset_index(drop=True)
        f_opt_waypoints = noise_opt_waypoints.query("fid==@fid").reset_index(drop=True)
        f_opt_noise = noise_opt.query("fid==@fid").reset_index(drop=True)
        ax.plot(f0.longitude, f0.latitude, "--", color="black", transform=PlateCarree())
        ax.plot(
            f_opt_waypoints.longitude,
            f_opt_waypoints.latitude,
            color="red",
            transform=PlateCarree(),
        )
        ax.plot(
            f_opt_noise.longitude,
            f_opt_noise.latitude,
            color="blue",
            transform=PlateCarree(),
        )

        existing_waypoints = fid_waypoints[fid]
        optimal_waypoints = fid_waypoints_opt[fid]

        ax.scatter(
            np.array([p[1] for p in optimal_waypoints]),
            np.array([p[0] for p in optimal_waypoints]),
            color="green",
            transform=PlateCarree(),
            s=10,
        )

        ax.scatter(
            np.array([p[3] for p in existing_waypoints]),
            np.array([p[2] for p in existing_waypoints]),
            color="black",
            transform=PlateCarree(),
            s=10,
        )

    ax.set_facecolor("linen")
    ax.set_global()

plt.show()
fig.savefig("figs/optimal_waypoints_18L.png", bbox_inches="tight", dpi=700)
# %%


# waypoints = [start] + fid_waypoints[fid] + [c_ends.query("fid==@fid")]
# waypoints_coor = (
#     [[start.latitude, start.longitude]]
#     + [[p.latitude, p.longitude] for p in fid_waypoints[fid]]
#     + [end_coor]
# )
# trajectory = get_trajectory(waypoints_coor, m0)

# waypoints_coor_dr = [[start.latitude, start.longitude]] + [end_coor]
# trajectory_droite = get_trajectory(waypoints_coor_dr, m0)

# x_opt, val_opt = optim(fid)
# print(val_opt)
# waypoints_coor_opt = (
#     [[start.latitude, start.longitude]]
#     + [[i, j] for i, j in zip(x_opt[0::2], x_opt[1::2])]
#     + [end_coor]
# )
# trajectory_opt = get_trajectory(waypoints_coor_opt, m0=m0)

# track = geo.bearing(
#     trajectory_droite.latitude.to_numpy()[:-1],
#     trajectory_droite.longitude.to_numpy()[:-1],
#     trajectory_droite.latitude.to_numpy()[1:],
#     trajectory_droite.longitude.to_numpy()[1:],
# )

# fig = plot_map([trajectory, trajectory_droite, trajectory_opt], waypoints_coor_opt)
# plt.show()
