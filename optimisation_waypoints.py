import matplotlib.pyplot as plt
from cartes.utils.features import countries
from cartes.crs import Amersfoort
from cartopy.crs import PlateCarree
from openap.gen import FlightGenerator
import matplotlib
import numpy as np
import pandas as pd
from openap import top, Thrust, FuelFlow, Drag
from pitot import geodesy as geo

pd.set_option("display.max_rows", 15)

map_type = "DN"
actype = "a320"
c_ends = pd.read_csv(f"data_generated/opensky_centroid_ends_{map_type}.csv")
df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
interpolant = top.tools.interpolant_from_dataframe(df_cost)

df_cost = pd.read_csv(f"data_generated/df_cost_{map_type}.csv")
if map_type == "D" or map_type == "N":
    pop_map = (
        pd.read_parquet(f"data_raw/{map_type}012011_1K_cropped.parquet")
        .rename(columns={"popul": "pp"})
        .fillna(0)
    )
else:
    pop_map = pd.read_parquet("data_raw/pop_static.parquet")


matplotlib.rc("font", size=11)
matplotlib.rc("font", family="Ubuntu")
matplotlib.rc("grid", color="darkgray", linestyle=":")


def plot_map(flights: pd.DataFrame, waypoints_coor=None):
    if isinstance(flights, pd.DataFrame):
        flights = [flights]
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw=dict(projection=Amersfoort())
        )
        ax.add_feature(countries())
        for f in flights:
            ax.plot(f.longitude, f.latitude, transform=PlateCarree())

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
        ax.set_facecolor("linen")

        if waypoints_coor:
            for w in waypoints_coor:
                ax.scatter([w[1]], [w[0]], color="r", transform=PlateCarree(), s=10)

        ax.set_global()
    return fig, ax


def get_fid_params(fid):
    end = (
        c_ends.query("fid==@fid")[["latitude", "longitude", "tow"]].to_numpy().flatten()
    )
    m0 = end[-1]
    end_coor = list(end[:2])
    return end_coor, m0


fuelflow = FuelFlow(actype)
thr = Thrust(actype)
dr = Drag(actype)


def compute_metrics(df, m0):
    mass_current = m0

    fuelflow_every_step = []
    fuel_every_step = []
    thrust_every_step = []
    drag_every_step = []
    grid_cost_position = []
    grid_cost = []
    mass = []

    for _, row in df.iterrows():
        ff = fuelflow.enroute(
            mass=mass_current,
            tas=row.tas,
            alt=row.altitude,
            vs=row.vertical_rate,
        )

        thrust = thr.climb(
            tas=row.tas,
            alt=row.altitude,
            roc=row.vertical_rate,
        )

        drag = dr.clean(
            mass=mass_current,
            tas=row.tas,
            alt=row.altitude,
            vs=row.vertical_rate,
        )

        fuel = ff * row["dt"]
        fuelflow_every_step.append(ff)
        fuel_every_step.append(fuel)
        thrust_every_step.append(thrust)
        drag_every_step.append(drag)
        mass.append(mass_current)
        mass_current -= fuel

        cost_position = interpolant([row.longitude, row.latitude, row.h])
        grid_cost_position.append(float(cost_position))
        grid_cost.append(float(cost_position * row["dt"]))

    df = df.assign(
        fuel_flow=fuelflow_every_step,
        fuel=fuel_every_step,
        mass=mass,
        # thrust=thrust_every_step,
        thrust=drag_every_step,
        grid_cost_position=grid_cost_position,
        grid_cost=grid_cost,
    )
    return df


def get_trajectory(waypoints_coor, m0, dt=1):
    fgen = FlightGenerator(ac="a320")
    flight_climb = fgen.climb(dt=dt)
    flight_climb.rename(columns={"t": "ts", "groundspeed": "tas"}, inplace=True)
    flight_climb = flight_climb.query("h>0").reset_index(drop=True)
    flight_climb["dt"] = dt
    flight_climb["ts"] = flight_climb["ts"] - flight_climb["ts"].min()
    flight_climb["s"] = flight_climb["s"] - flight_climb["s"].min()
    latitudes = []
    longitudes = []
    for sta, end in zip(waypoints_coor[:-1], waypoints_coor[1:]):
        x1, y1 = sta
        x2, y2 = end

        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        xx = np.linspace(x1, x2, 10**5)
        yy = a * xx + b

        latitudes += list(xx)
        longitudes += list(yy)

    latitudes = np.array(latitudes).reshape((-1, 1))
    longitudes = np.array(longitudes).reshape((-1, 1))
    coordinates = np.hstack((latitudes, longitudes))

    dist = geo.distance(
        coordinates[:, 0][:-1],
        coordinates[:, 1][:-1],
        coordinates[:, 0][1:],
        coordinates[:, 1][1:],
    )
    cumdist = np.cumsum(dist)
    cumdist = np.insert(cumdist, 0, 0)

    latitudes = []
    longitudes = []
    for s in flight_climb.s.to_numpy():
        if s <= cumdist[-1]:
            index = np.argmin(np.abs(s - cumdist))
            lat, lon = coordinates[index]
            latitudes.append(lat)
            longitudes.append(lon)

    l = len(latitudes)
    flight_climb = flight_climb.iloc[:l]

    flight_climb["latitude"] = latitudes
    flight_climb["longitude"] = longitudes

    flight_climb = compute_metrics(flight_climb, m0)

    return flight_climb


def obj_function(x, start, end, m0, c=0.001):
    waypoints_coor = [start]
    waypoints_coor += [[i, j] for i, j in zip(x[0::2], x[1::2])]
    waypoints_coor += [end]
    trajectory = get_trajectory(waypoints_coor, m0)
    latitudes = trajectory.latitude.to_numpy()
    longitudes = trajectory.longitude.to_numpy()
    if len(trajectory) == 0:
        return 10**5

    # track = geo.bearing(
    #     latitudes[:-1],
    #     longitudes[:-1],
    #     latitudes[1:],
    #     longitudes[1:],
    # )
    # if np.max(np.abs(np.diff(track))) > 60:
    #     return 10**5

    distance = geo.distance(
        latitudes[-1],
        longitudes[-1],
        end[0],
        end[1],
    )
    if distance > 10**3:
        return 10**5

    loss = (trajectory.fuel + c * trajectory.thrust * trajectory.grid_cost).sum()
    return loss
