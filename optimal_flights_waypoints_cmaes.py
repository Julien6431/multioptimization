# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pitot import geodesy as geo
from openap import nav, prop
from traffic.data import navaids, airports
from optimisation_waypoints import (
    plot_map,
    get_trajectory,
    obj_function,
    get_fid_params,
)
from traffic.core import Traffic
from cmaes import CMA

pd.set_option("display.max_rows", 15)

# %%

actype = "a320"
map_type = "DN"
eham = nav.airport("EHAM")
c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
t_cnts = Traffic(
    pd.read_parquet(f"data_generated/opensky2024_centroids_{map_type}.parquet").rename(
        columns={"heading": "track", "groundspeed": "tas", "flight_id": "fid"}
    )
)
start = airports["EHAM"]
start_coor = [start.latitude, start.longitude]


t_cnts_L18 = t_cnts.query("runway=='18L'")
m = t_cnts_L18[0].map_leaflet()
for f in t_cnts_L18[1:]:
    m.add(f)
m.add(navaids["EH029"])
m.add(navaids["EH073"])
m.add(navaids["EH050"])
m.add(navaids["LOPIK"])
m.add(navaids["LEKKO"])
m.add(navaids["EH024"])
m.add(navaids["EH037"])
m.add(navaids["IVLUT"])
m.add(navaids["ANDIK"])
m.add(navaids["LUNIX"])
m.add(navaids["NYKER"])
m.add(navaids["WOODY"])
m.add(navaids["VELED"])
m

# %%

fid_waypoints = {
    t_cnts_L18[0].data.fid.values[0]: [navaids["EH073"], navaids["LEKKO"]],
    t_cnts_L18[2].data.fid.values[0]: [
        # navaids["EH037"],
        navaids["EH024"],
        navaids["IVLUT"],
        navaids["NYKER"],
    ],
    t_cnts_L18[4].data.fid.values[0]: [
        # navaids["EH037"],
        navaids["EH024"],
        navaids["IVLUT"],
        navaids["LUNIX"],
    ],
    t_cnts_L18[5].data.fid.values[0]: [
        # navaids["EH073"],
        navaids["EH029"],
        navaids["EH050"],
        navaids["LOPIK"],
    ],
}

with open("data_generated/waypoints/current_waypoints.json", "w") as fp:
    json.dump(fid_waypoints, fp)

# %%

trajectories_pd = pd.DataFrame()
trajectories = []
for fid in tqdm(fid_waypoints.keys()):
    end_coor, m0 = get_fid_params(fid)
    waypoints_coor = (
        [start_coor]
        + [[p.latitude, p.longitude] for p in fid_waypoints[fid]]
        + [end_coor]
    )
    trajectory = get_trajectory(waypoints_coor, m0)
    trajectory["fid"] = fid
    trajectories.append(trajectory)

    trajectory = trajectory[
        [
            "ts",
            "h",
            "latitude",
            "longitude",
            "altitude",
            "tas",
            "vertical_rate",
            "fuel",
            "mass",
            "thrust",
            "grid_cost_position",
            "grid_cost",
        ]
    ]
    trajectory["fid"] = fid

    trajectories_pd = pd.concat([trajectories_pd, trajectory])

trajectories_pd.to_csv(
    f"data_generated/waypoints/flights_existing_procedure_{map_type}.csv",
    index=False,
)

fig = plot_map(trajectories)
plt.show()

# %%


def optim(fid, c=0.001, tol_fun=6, eval_max=10**3, verbose=False):
    waypoints = fid_waypoints[fid]
    end_coor, m0 = get_fid_params(fid)
    obj_func = lambda x: obj_function(x, start_coor, end_coor, m0, c=c)

    x0 = [
        val for pair in [(p.latitude, p.longitude) for p in waypoints] for val in pair
    ]
    x0 = np.array(x0)

    lower_bound = (x0 - 0.2).reshape((-1, 1))
    upper_bound = (x0 + 0.2).reshape((-1, 1))
    bounds = np.hstack((lower_bound, upper_bound))

    optimizer = CMA(
        mean=x0, sigma=0.075, bounds=bounds, population_size=10, lr_adapt=True
    )

    if verbose:
        print(" evals    f(x)")
        print("======  ==========")

    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = obj_func(x)
            evals += 1
            solutions.append((x, value))
            if verbose:
                if evals % 100 == 0:
                    print(f"{evals:5d}  {value:10.5f}")
        optimizer.tell(solutions)

        solutions.sort(key=lambda s: s[1])
        if (solutions[-1][1] - solutions[0][1] < tol_fun) and solutions[0][1] < 10**5:
            break
        if evals >= eval_max:
            break

    solutions.sort(key=lambda s: s[1])
    x_opt, val_opt = solutions[0]

    return x_opt, val_opt


# %%

trajectories_opt_pd = pd.DataFrame()

trajectories_opt = []
fid_waypoints_opt = {}
for fid in tqdm(fid_waypoints.keys()):
    end_coor, m0 = get_fid_params(fid)
    x_opt, val_opt = optim(fid, verbose=True)
    waypoints_coor_opt = (
        [start_coor] + [[i, j] for i, j in zip(x_opt[0::2], x_opt[1::2])] + [end_coor]
    )
    trajectory_opt = get_trajectory(waypoints_coor_opt, m0=m0)
    trajectories_opt.append(trajectory_opt)

    fid_waypoints_opt[fid] = [[i, j] for i, j in zip(x_opt[0::2], x_opt[1::2])]

    trajectory_opt = trajectory_opt[
        [
            "ts",
            "h",
            "latitude",
            "longitude",
            "altitude",
            "tas",
            "vertical_rate",
            "fuel",
            "mass",
            "thrust",
            "grid_cost_position",
            "grid_cost",
        ]
    ]
    trajectory_opt["fid"] = fid

    trajectories_opt_pd = pd.concat([trajectories_opt_pd, trajectory_opt])
    trajectories_opt_pd.to_csv(
        f"data_generated/waypoints/flights_noise_opt_{map_type}.csv", index=False
    )

    with open("data_generated/waypoints/optimal_waypoints.json", "w") as fp:
        json.dump(fid_waypoints_opt, fp)

fig = plot_map(trajectories + trajectories_opt)
plt.show()
