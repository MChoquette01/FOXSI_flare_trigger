from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import numpy as np
import netCDF4 as nc
import math
from sqlalchemy import create_engine, inspect, text, insert, delete, Column, INTEGER, FLOAT, VARCHAR, MetaData, Table as sqltable
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

"""Transfers data from a FITS file with extracted flares (from making_historical.py) to PostgreSQL.
Requires download of MongoDB (https://www.postgresql.org/download/). The desktop app pgAdmin helps (should be part of the same install)."""


flare_filepath = r"C:\Users\matth\Documents\Capstone\data\GOES_XRS_historical.fits"
xray_filepath = r"C:\Users\matth\Documents\Capstone\data\sci_xrsf-l2-avg1m_g16_s20170207_e20240906_v2-2-0.nc"

# open xray file
xray_data = nc.Dataset(xray_filepath)
xray_xrsa_fluxes = xray_data.variables["xrsa_flux"][:]
xray_xrsa_fluxes = xray_xrsa_fluxes.tolist(fill_value=np.nan)
xray_xrsb_fluxes = xray_data.variables["xrsb_flux"][:]
xray_xrsb_fluxes = xray_xrsb_fluxes.tolist(fill_value=np.nan)
xray_times = xray_data.variables["time"][:]
xray_times = xray_times.tolist(fill_value=np.nan)

# Open FITS file
flares = fits.open(flare_filepath)
data = Table(flares[1].data)[:]
all_data = []
for index in range(len(data)):
    this_data = {}
    for variable in data.colnames:
        this_data[variable] = data[variable][index]
    all_data.append(this_data)

# Open PostgreSQL
engine = create_engine('postgresql://postgres:Admin123.@localhost:5432/')  # Update with the postgreSQL credentials on YOUR local machine
connection = engine.connect()

Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()
metadata = MetaData()


class Flares(Base):
    __tablename__ = 'Flares'
    FlareID = Column(VARCHAR, primary_key=True)
    FlareClass = Column(VARCHAR)
    MinutesToPeak = Column(INTEGER)
    CurrentXRSA = Column(FLOAT)
    CurrentXRSB = Column(FLOAT)
    XRSA3MinuteDifference = Column(FLOAT)
    XRSB3MinuteDifference = Column(FLOAT)
    RemainingXRSA = Column(VARCHAR)
    RemainingXRSB = Column(VARCHAR)


# flares_table = sqltable(
#                'Flares',
#                metadata,
#                Column('FlareID', VARCHAR, primary_key=True),
#                Column('FlareClass', VARCHAR),
#                Column("MinutesToPeak", INTEGER),
#                Column("CurrentXRSA", FLOAT),
#                Column("CurrentXRSB", FLOAT),
#                Column("XRSA3MinuteDifference", FLOAT),
#                Column("XRSB3MinuteDifference", FLOAT),
#                Column('RemainingXRSA', VARCHAR),
#                Column('RemainingXRSB', VARCHAR))


if engine.dialect.has_table(connection, "Flares"):
    session.query(Base.metadata.tables[Flares.__tablename__]).filter().delete()
    session.commit()

# create DB
with engine.connect() as conn:
    Base.metadata.create_all(engine)
    conn.commit()


def assign_flare_class(xrsb_history):

    max_flux = np.max(xrsb_history)
    flux_value = str(max_flux)[:3]
    if max_flux < 10**-7:
        return f"A{flux_value}"
    elif 10**-7 <= max_flux < 10**-6:
        return f"B{flux_value}"
    elif 10**-6 <= max_flux < 10**-5:
        return f"C{flux_value}"
    elif 10**-5 <= max_flux < 10**-4:
        return f"M{flux_value}"
    elif max_flux > 10**-4:
        return f"X{flux_value}"

# Write FITS to DB
to_insert = []
for flare_data in tqdm(all_data, desc="Writing Flares Database..."):
    flare_id = flare_data["flare ID"]
    flare_class = flare_data["class"]
    if flare_class == "":
        flare_class = assign_flare_class(flare_data["xrsb"])
    peak_time = flare_data["peak time"]
    for idx, (xrsa_datum, xrsb_datum, time) in enumerate(zip(flare_data["xrsa"], flare_data["xrsb"], flare_data["time"])):
        minutes_to_peak = (peak_time - flare_data["time"][idx]) / 60
        xrsa_remaining_data = str(flare_data["xrsa"][idx:])
        # convert data to be a list-like string
        # '[one two three]' -> '[one, two, three]'
        xrsa_remaining_data = xrsa_remaining_data.replace("  ", " ")
        xrsa_remaining_data = xrsa_remaining_data.replace("[ ", "[")
        xrsa_remaining_data = xrsa_remaining_data.replace(" ", ", ")
        xrsb_remaining_data = str(flare_data["xrsb"][idx:])
        xrsb_remaining_data = xrsb_remaining_data.replace("  ", " ")
        xrsb_remaining_data = xrsb_remaining_data.replace("[ ", "[")
        xrsb_remaining_data = xrsb_remaining_data.replace(" ", ", ")
        # calculate difference in quantities over the last 3 datapoints...
        if idx >= 3:
            xrsa_3_minute_diff = float(flare_data["xrsa"][idx] - flare_data["xrsa"][idx - 3])
            xrsb_3_minute_diff = float(flare_data["xrsb"][idx] - flare_data["xrsb"][idx - 3])
        # ...unless we don't have that much data for a flare, then find the X-ray data from the source...
        else:
            xray_time_index = xray_times.index(time)
            xrsa_3_minute_diff = float(xray_xrsa_fluxes[xray_time_index] - xray_xrsa_fluxes[xray_time_index - 3])
            xrsb_3_minute_diff = float(xray_xrsb_fluxes[xray_time_index] - xray_xrsb_fluxes[xray_time_index - 3])
            # ...UNLESS it's still NaN (bad data)...write as null
            if math.isnan(xrsa_3_minute_diff):
                xrsa_3_minute_diff = None
            if math.isnan(xrsb_3_minute_diff):
                xrsb_3_minute_diff = None

        to_insert.append({
            "FlareID": f"{flare_id}_{idx}",
            "FlareClass": flare_class,
            "MinutesToPeak": minutes_to_peak,
            "CurrentXRSA": float(xrsa_datum),
            "CurrentXRSB": float(xrsb_datum),
            "XRSA3MinuteDifference": xrsa_3_minute_diff,
            "XRSB3MinuteDifference": xrsb_3_minute_diff,
            "RemainingXRSA": xrsa_remaining_data,
            "RemainingXRSB": xrsb_remaining_data
        })
with engine.connect() as conn:
    conn.execute(Base.metadata.tables[Flares.__tablename__].insert(), to_insert)
    conn.commit()