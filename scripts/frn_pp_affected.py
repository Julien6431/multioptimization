# %%
from traffic.core import Traffic, Flight
import pandas as pd
import numpy as np
from openap import FuelFlow, Thrust, aero, top
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer, CRS
import itertools
from tqdm import tqdm
import sys
import os

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath("../src/"))
from pp_affected import pp_affected, cost_grid_cost, get_npd_interpolator

# %%
ac = "a320"
fuelflow = FuelFlow(ac)
thrust = Thrust(ac)

curr_path = os.path.dirname(os.path.realpath(__file__))
options = itertools.product(["N", "D", "DN"], [50, 60, 75], ["f", "r", "n"])
options = itertools.product(["N", "D", "DN"], ["f", "r", "n"])

for treshold in [50, 60, 75]:
    pp_dict = []
    print(treshold)
    options = itertools.product(["N", "D", "DN"], ["f", "r", "n"])
    for map_type, frn in options:
        print(map_type, treshold, frn)
        if map_type == "D" or map_type == "N":
            pop_map = (
                pd.read_parquet(
                    curr_path + f"/../data_raw/{map_type}012011_1K_cropped.parquet"
                )
                .rename(columns={"popul": "pp"})
                .fillna(0)
            )
        else:
            pop_map = pd.read_parquet(curr_path + f"/../data_raw/pop_static.parquet")

        if frn == "f":
            flights = pd.read_csv(
                curr_path + f"/../data_generated/flights_fuel_opt_{map_type}.csv"
            ).assign(frn="f")
        elif frn == "n":
            flights = pd.read_csv(
                curr_path + f"/../data_generated/flights_noise_opt_{map_type}.csv"
            ).assign(frn="n")
        else:
            flights = (
                pd.read_parquet(
                    curr_path
                    + f"/../data_generated/opensky2024_centroids_{map_type}.parquet"
                )
                .assign(frn="r")
                .rename(columns={"groundspeed": "tas", "flight_id": "fid"})[
                    [
                        "ts",
                        "fid",
                        "latitude",
                        "altitude",
                        "longitude",
                        "mass",
                        "vertical_rate",
                        "tas",
                        "h",
                        "frn",
                    ]
                ]
            )
        df_cost = pd.read_csv(curr_path + f"/../data_generated/df_cost_{map_type}.csv")

        for fid in flights.fid.unique():
            flight = flights.query("fid==@fid").reset_index(drop=True)
            if map_type == "DN":
                grid_type = "ll"
            else:
                grid_type = "xy"
            flight = cost_grid_cost(df_cost, flight)
            pp_aff_all, pp_aff_uni, flight, _ = pp_affected(
                ac, pop_map, grid_type, flight, treshold
            )
            pp_dict.append(
                {
                    "fid": fid,
                    "fuel": flight.fuel.sum(),
                    "cost_grid": flight.cost_grid.sum(),
                    "people_affected_sum_all": pp_aff_all,
                    "pp_unique_sum": pp_aff_uni,
                    "map_type": map_type,
                    "frn": frn,
                }
            )

    df = pd.DataFrame.from_dict(pp_dict)
    df.to_csv(
        curr_path + f"/../data_generated/pp_affected_{treshold}.csv",
        index=False,
    )


# for map_type, treshold, frn in options:
#     print(map_type, treshold, frn)
#     if map_type == "D" or map_type == "N":
#         pop_map = (
#             pd.read_parquet(
#                 curr_path + f"/../data_raw/{map_type}012011_1K_cropped.parquet"
#             )
#             .rename(columns={"popul": "pp"})
#             .fillna(0)
#         )
#     else:
#         pop_map = pd.read_parquet(curr_path + f"/../data_raw/pop_static.parquet")

#     if frn == "f":
#         flights = pd.read_csv(
#             curr_path + f"/../data_generated/flights_fuel_opt_{map_type}.csv"
#         ).assign(frn="f")
#     elif frn == "n":
#         flights = pd.read_csv(
#             curr_path + f"/../data_generated/flights_noise_opt_{map_type}.csv"
#         ).assign(frn="n")
#     else:
#         flights = (
#             pd.read_parquet(
#                 curr_path
#                 + f"/../data_generated/opensky2024_centroids_{map_type}.parquet"
#             )
#             .assign(frn="r")
#             .rename(columns={"groundspeed": "tas", "flight_id": "fid"})[
#                 [
#                     "ts",
#                     "fid",
#                     "latitude",
#                     "altitude",
#                     "longitude",
#                     "mass",
#                     "vertical_rate",
#                     "tas",
#                     "h",
#                     "frn",
#                 ]
#             ]
#         )
#     df_cost = pd.read_csv(curr_path + f"/../data_generated/df_cost_{map_type}.csv")

#     pp_dict = []

#     for fid in flights.fid.unique():
#         flight = flights.query("fid==@fid").reset_index(drop=True)
#         if map_type == "DN":
#             grid_type = "ll"
#         else:
#             grid_type = "xy"
#         flight = cost_grid_cost(df_cost, flight)
#         pp_aff_all, pp_aff_uni, flight, _ = pp_affected(
#             ac, pop_map, grid_type, flight, treshold
#         )
#         pp_dict.append(
#             {
#                 "fid": fid,
#                 "fuel": flight.fuel.sum(),
#                 "cost_grid": flight.cost_grid.sum(),
#                 "people_affected_sum_all": pp_aff_all,
#                 "pp_unique_sum": pp_aff_uni,
#                 "map_type": map_type,
#                 "frn": frn,
#             }
#         )

#     df = pd.DataFrame.from_dict(pp_dict)
#     df.to_csv(
#         curr_path
#         + f"/../data_generated/pp_affected_{treshold}/pp_affected_{frn}_{map_type}.csv",
#         index=False,
#     )

# # %%

# %%
