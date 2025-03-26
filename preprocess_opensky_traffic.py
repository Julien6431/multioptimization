# %%
import pandas as pd
import numpy as np
from traffic.core import Flight, Traffic
from glob import glob
from openap import aero, Thrust, prop, FuelFlow
from traffic.data import airports
# %%
def sort_timestamp(flight):
    return Flight(flight.data.sort_values(by=["timestamp"]))

def mean_cruise_alt(flight):
    if flight is None:
        return None
    if len(flight.data) == 0:
        return None
    if (
        flight.query("phase=='CRUISE'") is None
        and flight.query("phase=='CLIMB'") is not None
        and len(flight.query("phase=='CLIMB'").data) > 10
    ):
        mean_alt = np.percentile(
            flight.query("phase=='CLIMB'").data.altitude.values, 90
        )
        mean_speed = np.percentile(
            flight.query("phase=='CLIMB'").data.groundspeed.values[-5:], 50
        )
        return flight.assign(mean_alt=mean_alt, mean_speed=mean_speed)
    if flight.query("phase=='CRUISE'") is not None:
        mean_alt = flight.query("phase=='CRUISE'").data.altitude.mean()
        mean_speed = flight.query("phase=='CRUISE'").data.groundspeed.mean()
        return flight.assign(mean_alt=mean_alt, mean_speed=mean_speed)

def find_runway(flight):
    if flight.takeoff_from_runway("EHAM").all() is not None:
        rwy = flight.takeoff_from_runway("EHAM").all().data.runway.values[0]
    else:
        rwy = None
    return flight.assign(runway=rwy)


def ts_assign(flight):
    start_time = flight.data.timestamp.min()
    if 8 < start_time.hour < 20:
        flight = flight.assign(day_night="D")
    else:
        flight = flight.assign(day_night="N")
    return flight.assign(ts=lambda x: x.timestamp.diff().cumsum().dt.total_seconds()).fillna(
        0
    )
# %%%
for month in ["03", "06", "09", "12"]:
    files = glob(f"data_raw/opensky/*{month}*.parquet")
    df = pd.concat([pd.read_parquet(file) for file in files]).assign(
        vertical_rate=lambda x: x.vertrate / aero.fpm,
        altitude=lambda x: x.altitude / aero.ft,
        groundspeed=lambda x: x.velocity / aero.kts,
        fl=lambda x: x.altitude / aero.ft / 100,
    )
    t = Traffic(df).pipe(sort_timestamp).eval(6)
    t = (
        # t.filter()
        t.assign_id()
        .cumulative_distance()
        .phases()
        .pipe(mean_cruise_alt)
        .pipe(ts_assign)
        .pipe(find_runway)
        .eval(6)
    )
    df = t.data.assign(distance=lambda x: x.cumdist * aero.nm / 1000)
    df.to_parquet(f"data_raw/opensky/monthly/2024-{month}.parquet")

# %%
for month in ["03", "06", "09", "12"]:
    t = Traffic(pd.read_parquet(f"data_raw/opensky/monthly/2024-{month}.parquet"))
    t = t.resample("10s").resample(100).eval(6)
    t.data.to_parquet(f"data_raw/opensky/monthly/2024-{month}_resampled.parquet")

# %% 
t = Traffic(
    pd.concat(
        [
            pd.read_parquet(f"data_raw/opensky/monthly/2024-03_resampled.parquet"),
            pd.read_parquet(f"data_raw/opensky/monthly/2024-06_resampled.parquet"),
            pd.read_parquet(f"data_raw/opensky/monthly/2024-09_resampled.parquet"),
            pd.read_parquet(f"data_raw/opensky/monthly/2024-12_resampled.parquet"),
        ]
    )
    .query("mean_alt>28000")
    .fillna("idk") #for runway nans
)

# %%
def find_arrival(flight):
    flights_info = pd.read_csv("data_raw/flights_info.csv")#[["icao24","departure","arrival","callsign","day"]]
    flights_info["day"] = pd.to_datetime(flights_info["day"]).dt.tz_localize(None)
    flights_info["firstseen"] = pd.to_datetime(flights_info["firstseen"]).dt.tz_localize(None)
    df = flight.data
    month = df.timestamp.min().month
    day = df.timestamp.min().day
    finfo = flights_info.query(f"day.dt.month=={month} and day.dt.day=={day}")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    min_time = df["timestamp"].min()
    finfo = finfo.assign(dif_time=lambda x: (x["firstseen"] - min_time).abs()).query(
        "dif_time == dif_time.min()")
    df = df.merge(finfo[["icao24","departure","arrival","callsign"]], on=["icao24","callsign"], how="left")
    return Flight(df)


thrust = Thrust("a320")

mass_estim_coefs = np.array([1.45557, -2.11051, 187.53086,55622.0])
# Coefficients: TOW = (1.45557) * dist + (-2.11051) * h +(187.53086) * TAS + (55622.0)

def drop_dups(f):
    return f.drop_duplicates(subset=["timestamp"], keep="first").drop_duplicates(
        subset=["latitude", "longitude"], keep="first"
    )
def assign_tow(f):
    m_mtow = prop.aircraft("a320")["limits"]["MTOW"]
    oew = prop.aircraft("a320")["limits"]["OEW"]
    mass_estim_coefs = np.array([1.45557, -2.11051, 187.53086,55622.0])
    arrival = f.data.arrival.values[0]
    if type(arrival) is type("string") and arrival!="NaN":
        if arrival =="LDZK":
            distance = aero.distance(airports["EHAM"].latitude,airports["EHAM"].longitude,46.012778,15.860000)/1000
        else:
            distance = aero.distance(airports["EHAM"].latitude,airports["EHAM"].longitude,airports[arrival].latitude,airports[arrival].longitude)/1000
        # print("distance to "+arrival+" - "+str(distance)+" km" )
    else:
        distance = 1500
        # print(f.flight_id + "nan arrival")
    tow=(mass_estim_coefs[0] * distance
        + mass_estim_coefs[1] * f.data.mean_alt.values[0]
        + mass_estim_coefs[2] * f.data.mean_speed.values[0]
        + mass_estim_coefs[3])
    if tow > m_mtow:
        tow=m_mtow
    elif tow <oew:
        tow = oew*1.2
    return f.assign(tow=tow)



def assign_fuel(flight):
    df = (
        flight.fuelflow(initial_mass=flight.data.tow.values[0])
        .data.assign(fuel=lambda x: x.fuelflow * x.dt)
        .assign(fuel_sum=lambda x: x.fuel.sum())
    )
    return Flight(df)


def assign_mass(f):
    return f.assign(mass=lambda x: x.tow - x.fuel.cumsum())
    # fuelflow.enroute(f.tow, f.groundspeed, f.fl*100, f.vertical_rate)


def assign_thrust(f):
    return f.assign(
        thrust=lambda x: thrust.enroute(
            mass=x.mass, tas=x.groundspeed, alt=x.altitude, vs=x.vertical_rate
        )
    )

t = (
    t.pipe(drop_dups)
    .pipe(find_arrival)
    .pipe(assign_tow)
    .pipe(assign_fuel)#not needed
    .pipe(assign_mass)
    .pipe(assign_thrust)#not needed
    .eval(6)
)
t = t.query("distance<220").compute_xy()
t = t.assign(
    x=lambda df: df.x.astype(int),
    y=lambda df: df.y.astype(int),
)

t = t.pipe(drop_dups).resample(40).eval(6)
t.data.to_parquet("data_generated/opensky_flights_2024dn_ready_for_clustering.parquet")
# %%
