#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta

def save_pickle(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def datenum_to_datetime(datenum):
    """
    Source: https://gist.github.com/victorkristof/b9d794fe1ed12e708b9d#file-datenum-to-datetime-py 
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60

    return dt.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)


def convert_timestamps_to_datetime(schedule_df, t_parameters=["T_initial", "T_final"]):
    ''' 
    After schedule is completed across entire dataset, this column adds datetime conversions of the t timestamp
    '''
    schedule_df["Start datetime"] = [dt.fromtimestamp(t_0) for t_0 in schedule_df[t_parameters[0]].values]
    schedule_df["End datetime"] = [dt.fromtimestamp(t_f) for t_f in schedule_df[t_parameters[1]].values]

    # identify month, day, hour of timestamps
    schedule_df["Month"] = schedule_df["Start datetime"].dt.month
    schedule_df["Day"] = schedule_df["Start datetime"].dt.day
    schedule_df["Hour"] = schedule_df["Start datetime"].dt.hour
    schedule_df["Minute"] = schedule_df["Start datetime"].dt.minute
    # covnert datetime to season
    schedule_df["Season"] = schedule_df["Start datetime"].dt.month%12 // 3 + 1
    # covnert datetime to binary time of day - following India sunset and Sunrise time
    period_of_day_conditons = [(schedule_df["Start datetime"].dt.hour < 7) & (schedule_df["Start datetime"].dt.hour > 18), 
            (schedule_df["Start datetime"].dt.hour > 7) & (schedule_df["Start datetime"].dt.hour < 18)]
    period_of_day_values = [0, 1]
    schedule_df["Period of Day"] = np.select(period_of_day_conditons,period_of_day_values)
    
    return schedule_df


def open_field_data(turbine):
    field_data_filepath = "data/Field data/" + str(turbine) + ".csv"
    raw_field_data_df = pd.read_csv(str(field_data_filepath)) #.rename(columns={"Unnamed: 0": "t"})

    if "Lidar" in turbine:
        # converts matlab datetime to python datetime
        # raw_field_data_df["DateTime"] = raw_field_data_df["DateTime"].apply(datenum_to_datetime)
        # raw_field_data_df["t"] = (raw_field_data_df.DateTime - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        # raw_field_data_df = raw_field_data_df.dropna()
        # raw_field_data_df.to_csv(f"../data/Field data/{turbine}.csv")
        field_data_df = raw_field_data_df[["t", "WndDir"]].rename(columns={"WndDir": "wind_direction"})
        field_data_df["sine_wind_direction"] = np.sin(field_data_df["wind_direction"])

    else:
        field_data_df = raw_field_data_df[["t", "TrueWndDir"]].rename(columns={"TrueWndDir": "wind_direction"})
    field_data_df = convert_timestamps_to_datetime(field_data_df, t_parameters=["t", "t"])
    return field_data_df


def open_fig():
    '''
    TODO: implement this function
    '''
    return


def save_fig():
    '''
    TODO: implement this function
    '''
    return