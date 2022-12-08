import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from loguru import logger
import pickle
from utils import *

def get_delta_degrees(angle1, angle2, sine=True):
    '''
    Returns the difference in degrees between inputed angles, considering circularity of degrees
    param angle1 : float : [deg]
    param angle2 : float : [deg]
    '''
    if not sine:
        full_rotation = 360
        absolute_distance = abs(angle1 - angle2)
        if absolute_distance < abs(absolute_distance - full_rotation):
            return absolute_distance
        
        return abs(absolute_distance - full_rotation)
    # if inputs are sine transformations of wind direction angles
    return abs(angle1-angle2)


def standard_control(i, t_i, T, delta_degree, turbine_rot_speed, temporal_resolution, phi_T_i, mean_phi_T_i, t_interval_size, last_theta_t, sine=True):
    ''' 
    Function representing yaw-aligned controller methods for turbine reactions to changes in wind direction. Called by the perfect preview function.
    Returns the next row of turbine duty.
    '''
    update_nacelle = True
    # duration[s] it takes for turbine to change from theta_t -> mean_phi_T_i
    delta_t = math.ceil(delta_degree/turbine_rot_speed) 

    # if the duration to update nacelle position is less than the data resolution, then update instantaneously
    if delta_t < temporal_resolution:
        theta_t = [mean_phi_T_i]*(t_interval_size)
        delta_degrees = [get_delta_degrees(phi_T_i[i], mean_phi_T_i, sine) for i in range(t_interval_size)]
        # append current nacelle position after repositioning, order of row entries correspond to column names above ^
        row_results = [i, t_i, t_i+T, phi_T_i, mean_phi_T_i, theta_t, delta_degrees, update_nacelle, delta_t]

    else:
        # intitialize previous time interval nacelle position, and arrays to store nacelle position and yaw error
        theta_0 = last_theta_t[-1]
        theta_t = []
        delta_degrees = []
        direction = -1 if theta_0 > mean_phi_T_i else 1
        # update nacelle position over time, given the turbine rotation dynamics
        for i in range(t_interval_size):
            next_theta = (theta_0 + direction*turbine_rot_speed*i*temporal_resolution)
            # if low pass filter of forecast is passed/ignored, break from loop
            # if len(theta_t) == t_interval_size:
            #     break
            if (theta_0 > mean_phi_T_i and mean_phi_T_i > next_theta) or (next_theta > mean_phi_T_i and mean_phi_T_i > theta_0):
                next_theta = mean_phi_T_i
                theta_t.append(next_theta)
                # delta_degrees.append(get_delta_degrees(mean_phi_T_i, last_theta))
                if len(phi_T_i) == 0:
                    delta_degrees.append(get_delta_degrees(mean_phi_T_i, next_theta, sine))
                else:
                    delta_degrees.append(get_delta_degrees(phi_T_i[i], next_theta, sine))
                break
            
            #TODO : check this logic
            full_rotation = 360
            if next_theta > full_rotation: 
                next_theta = -(next_theta%full_rotation)
            theta_t.append(next_theta)
            # recalculate change in degrees between target and current nacelle pos.
            # delta_degrees.append(get_delta_degrees(mean_phi_T_i, last_theta))
            if len(phi_T_i) == 0:
                delta_degrees.append(get_delta_degrees(mean_phi_T_i, next_theta, sine))
            else:
                delta_degrees.append(get_delta_degrees(phi_T_i[i], next_theta, sine))

        # remaining nacelle position in the interval will equal the forecast wind direction (yaw aligned)
        remaining_time_interval = t_interval_size - len(theta_t)
        if remaining_time_interval > 0:
            theta_t += [mean_phi_T_i]*remaining_time_interval
            delta_degrees += [0.0]*remaining_time_interval
        # append current nacelle position after repositioning, order of row entries correspond to column names above ^
        row_results = [i, t_i, t_i+T, phi_T_i, mean_phi_T_i, theta_t, delta_degrees, update_nacelle, delta_t]

    return row_results


def perfect_preview_control(data_df, T, temporal_resolution=60,deg_threshold=7,turbine_rot_speed=0.3,control_type="standard", sine=True):
    ''' 
    This function executes the identified wind farm control system informed by perfect preview information 
    of wind direction T into the future. The nacelle turbine position is referred to as "theta" and the wind direction is
    referred to as "phi"

    param data_df : pd.Dataframe : stores the time-series wind direction data 
            must have column names ["t" [seconds], "wind_direction" [degrees]]
    param T : int : forecasting period horizon
    param deg_threshold : float : the wind direction [degree] threshold at which turbine nacelle position should change
    turbine_rot_speed : float : the speed at which the turbine nacelle position is able to rotate [°/s]
    control_type : str : {"standard", "wake steering"}
    '''
    ## intialization
    # the maximum update frequency
    t_interval_size = int(T/(temporal_resolution))
    N = int(data_df.shape[0]/t_interval_size)
    # time and nacelle position at t=0
    t0 = data_df.t.values[0]

    # define wind direction column
    wind_dir_col = "wind_direction"
    # if sine is true, wind dir column is sine of wind direction
    if sine:
        wind_dir_col = "sine_wind_direction"
        # convert degree threshold to sine of angle in radians
        rad_threshold = deg_threshold * math.pi/180
        deg_threshold = np.sin(rad_threshold)
        # turbine rotation speed to sine of angle in radians
        if turbine_rot_speed != math.inf:
            rad_turbine_rotation_speed = turbine_rot_speed*math.pi/180
            turbine_rot_speed = np.sin(rad_turbine_rotation_speed)
    
    # starting nacelle position is the average across perfect preview data
    phi_t0 = data_df[((data_df.t >= t0) & (data_df.t <= t0+T))][wind_dir_col].values
    t0_interval_size = len(phi_t0)
    mean_phi_t0 = np.mean(phi_t0)
    theta_t = [mean_phi_t0]*(t0_interval_size)
    # count of turbine nacelle direction updates
    duty = 0
    # store changes in nacelle position
    turbine_control_schedule_df = pd.DataFrame(columns=["Interval", "T_initial", "T_final", "Perfect Preview Wind Direction (PPWD) (phi(t))", 
        "Low-pass-filter of PPWD (mean phi)", "Nacelle position (theta(t))",  "Delta Degree", "Nacelle Position Update (T/F)", "Nacelle Repositioning Duration (s)"])

    # store initialization run
    delta_degrees_t0 = [get_delta_degrees(phi_t0[i], theta_t[i], sine) for i in range(len(phi_t0))]
    delta_t0 = 0.0
    update_nacelle = False
    first_row = [0, t0, t0+T, phi_t0, mean_phi_t0, theta_t, delta_degrees_t0, update_nacelle, delta_t0]
    turbine_control_schedule_df.loc[len(turbine_control_schedule_df.index)] = first_row

    ## iterative nacelle updating
    for i in range(1, N-1):
        # define starting interval t_i
        t_i = i*T + t0
       
        # get perfect preview data ("forecasted") along time horizon T
        ti_interval_condition = ((data_df.t >= t_i) & (data_df.t <= t_i+T))
        phi_T_i = data_df[ti_interval_condition][wind_dir_col].values #data_df.wind_direction.values[t_i:t_i+(delta_t_interval)]
        ti_interval_size = len(phi_T_i)
        if ti_interval_size > 0:
            # compute low-pass-filter of perfect preview data in window of size T
            mean_phi_T_i = np.mean(phi_T_i)
            
            # check whether update is needed comparing the mean forecasting wind direction and most recent nacelle position
            last_theta_t = theta_t[-1]
            if len(phi_T_i) == 0:
                delta_degree = get_delta_degrees(mean_phi_T_i, last_theta_t, sine) # circular variable, finds shortest distance between directions
            else:
                delta_degree = get_delta_degrees(phi_T_i[0], last_theta_t, sine)
            # turbine nacelle update occurs
            if delta_degree > deg_threshold:
                update_nacelle = True
                duty += 1
                if control_type == "standard":
                    row_results = standard_control(i, t_i, T, delta_degree, turbine_rot_speed, temporal_resolution,phi_T_i, mean_phi_T_i, ti_interval_size, last_theta_t=theta_t, sine=sine)
                    theta_t, delta_degrees = row_results[5],row_results[6] 
                    turbine_control_schedule_df.loc[len(turbine_control_schedule_df.index)] = row_results

                elif control_type == "wake steering":
                    # row_results = wakesteering_control()
                    # turbine_control_schedule_df.loc[len(turbine_control_schedule_df.index)] = row_results
                    continue

            # no turbine nacelle update occurs
            else:
                # no update to the last nacelle position is made
                theta_t = [last_theta_t]*(ti_interval_size)
                delta_degrees = [get_delta_degrees(phi_T_i[i], last_theta_t, sine) for i in range(ti_interval_size)]
                update_nacelle = False
                delta_t = 0
                # order of row entries correspond to column names above ^
                row_results = [i, t_i, t_i+T, phi_T_i, mean_phi_T_i, theta_t, delta_degrees, update_nacelle, delta_t]
                turbine_control_schedule_df.loc[len(turbine_control_schedule_df.index)] = row_results
    
    
    turbine_control_schedule_df = convert_timestamps_to_datetime(turbine_control_schedule_df,)

    return turbine_control_schedule_df, duty


def plot_maker(model_method, t_interval_index, x_axis_range, y_axis_ranges):
    """
    Using matplotlib, create plot with set figure size, legend location, axes labels, etc. 
    This is currently design to meet specific helper needs for the function plot_true_v_pred_each_step()
    Returns ax plt.subplot with designated features
    """
    # create figure to plot it
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios':[3, 1]})

    # top subplot demonstrates the turbine control
    box = axs[0].get_position()
    axs[0].set_position([box.x0,box.y0,box.width*0.8,box.height])
    axs[0].set_xlim(x_axis_range)
    # if y_axis_ranges[0][0] < 180 and y_axis_ranges[0][1] > 180: 
    #     y_axis_ranges[0] = [-180,180]
    # axs[0].set_ylim(y_axis_ranges[0][0] + y_axis_ranges[0][0]*0.3, y_axis_ranges[0][1] + y_axis_ranges[0][1]*0.3)
    axs[1].set_xlim(x_axis_range)
    axs[1].set_ylim(y_axis_ranges[1][0] + y_axis_ranges[1][0]*0.3, y_axis_ranges[1][1] + y_axis_ranges[1][1]*0.3)
    
    # bottom plot shows the yaw error
    # setting axes and titles
    axs[0].set_ylabel("True Wind direction [°]")
    axs[1].set_ylabel("Yaw Error [°]")
    # if sine:
    #    axs[0].set_ylabel("Sine(True Wind Direction)")
    axs[0].set_xlabel("Time [s]")
    axs[1].set_xlabel("Time [s]")
    axs[1].grid(True)
    
    # set title
    plt.suptitle(f"{model_method[0]}, run {t_interval_index}, {model_method[1]}")
    fig.tight_layout()

    return axs


##########################################
################ PLOTTING ################
##########################################

def plot_turbine_reactions(update_schedule_df, field_temporal_resolution, n=10, sine=True):
    
    for i in range(update_schedule_df.shape[0]- 2):
        # define rows 1, 2, 3 corresponding to intervals
        row1 = update_schedule_df.iloc[i]
        row2 = update_schedule_df.iloc[i+1]
        row3 = update_schedule_df.iloc[i+2]
        # interval run in dataset
        run_index = row1.Interval
    
        # # define the t range (t1, t2, t3)
        # t1 = range(int(row1.T_initial), int(row1.T_final) + field_temporal_resolution , field_temporal_resolution)
        # t2 = range(int(row2.T_initial), int(row2.T_final) + field_temporal_resolution , field_temporal_resolution)
        # t3 = range(int(row3.T_initial), int(row3.T_final) + field_temporal_resolution , field_temporal_resolution)
        # print(f"t : {len(t1)}, {len(t2)}, {len(t3)}")

        # perfect preview wind direction data ('forecasted' wind direction)
        ppwd_t1 = row1["Perfect Preview Wind Direction (PPWD) (phi(t))"]
        ppwd_t2 = row2["Perfect Preview Wind Direction (PPWD) (phi(t))"]
        ppwd_t3 = row3["Perfect Preview Wind Direction (PPWD) (phi(t))"]
        print(f"ppwd : {len(ppwd_t1)}, {len(ppwd_t2)}, {len(ppwd_t3)}")

        # define the t range (t1, t2, t3)
        t1 = [int(row1.T_initial) + field_temporal_resolution*i for i in range(len(ppwd_t1))]
        t2 = [int(row2.T_initial) + field_temporal_resolution*i for i in range(len(ppwd_t2))]
        t3 = [int(row3.T_initial) + field_temporal_resolution*i for i in range(len(ppwd_t3))]
        print(f"t : {len(t1)}, {len(t2)}, {len(t3)}")

        # low pass filter of forecasted wind direction
        lpf_t1 =  [row1["Low-pass-filter of PPWD (mean phi)"]]*len(t1) 
        lpf_t2 =  [row2["Low-pass-filter of PPWD (mean phi)"]]*len(t2)
        lpf_t3 =  [row3["Low-pass-filter of PPWD (mean phi)"]]*len(t3)
        # print(f"lpf of ppwd : {len(lpf_t1)}, {len(lpf_t2)}, {len(lpf_t3)}")

        # nacelle control in reaction to changes in wind direction
        turbine_pos_t1 = row1["Nacelle position (theta(t))"]
        turbine_pos_t2 = row2["Nacelle position (theta(t))"]
        turbine_pos_t3 = row3["Nacelle position (theta(t))"]
        # print(f"nacelle position: {len(turbine_pos_t1)}, {len(turbine_pos_t2)}, {len(turbine_pos_t3)}")

        # yaw error
        yaw_error_t1 = row1["Delta Degree"]
        yaw_error_t2 = row2["Delta Degree"]
        yaw_error_t3 = row3["Delta Degree"]
        # if sine is true, convert yaw error to degrees
        if sine:
            yaw_error_t1 = abs(np.arcsin(row1["Delta Degree"])*180/math.pi)
            yaw_error_t2 = abs(np.arcsin(row2["Delta Degree"])*180/math.pi)
            yaw_error_t3 = abs(np.arcsin(row3["Delta Degree"])*180/math.pi)

        print(f"yaw error: {len(yaw_error_t1)}, {len(yaw_error_t2)}, {len(yaw_error_t3)}")
        print([*yaw_error_t1, *yaw_error_t2])
        # define axes limits
        x_axis_range = [row1.T_initial, row3.T_final]
        turbine_control_y_limits = [min([*ppwd_t1, *ppwd_t2, *ppwd_t3, *lpf_t1, *lpf_t2, *lpf_t3, *turbine_pos_t1, *turbine_pos_t2, *turbine_pos_t3]), \
            max([*ppwd_t1, *ppwd_t2, *ppwd_t3, *lpf_t1, *lpf_t2, *lpf_t3, *turbine_pos_t1, *turbine_pos_t2, *turbine_pos_t3])]
        delta_degree_y_limits = [min([*yaw_error_t1, *yaw_error_t2, *yaw_error_t3]), max([*yaw_error_t1, *yaw_error_t2, *yaw_error_t3])]
        y_axis_ranges = [turbine_control_y_limits, delta_degree_y_limits]
        # print(y_axis_ranges)
        # define graph title
        start_dt = row1["Start datetime"]
        end_dt = row3["End datetime"]
        model_method = ("Perfect Preview Wind Direction (PPWD) Based Turbine Control",  f"[{start_dt}, {end_dt}]")
        # create base plots
        try:
            axs = plot_maker(model_method, run_index, x_axis_range, y_axis_ranges,sine)
            
            # plot turbine controller
            axs[0].plot(t1, ppwd_t1, label= "PPWD (t1)", color="blue")
            axs[0].plot(t2, ppwd_t2, label= "PPWD (t2)", color="orange")
            axs[0].plot(t3, ppwd_t3, label= "PPWD (t3)", color="green")
            axs[0].plot(t1, lpf_t1, label= "LPF of PPWD (t1)", linestyle='dashed', color="blue", linewidth=3)
            axs[0].plot(t2, lpf_t2, label= "LPF of PPWD (t2)", linestyle='dashed', color="orange", linewidth=3)
            axs[0].plot(t3, lpf_t3, label= "LPF of PPWD (t3)", linestyle='dashed', color="green", linewidth=3)
            axs[0].plot(list(t1) + list(t2) + list(t3), turbine_pos_t1 + turbine_pos_t2 + turbine_pos_t3, label= "Nacelle position", linestyle='dotted', color="red")
            
            # plot corresponding yaw error below
            axs[1].plot(list(t1) + list(t2) + list(t3), [*yaw_error_t1, *yaw_error_t2, *yaw_error_t3], label="Yaw Error", color="red")

            # axs[1].plot(t1, yaw_error_t1, label="Yaw Error at t1", color="blue")
            # axs[1].plot(t2, yaw_error_t2, label="Yaw Error at t2", color="orange")
            # axs[1].plot(t3, yaw_error_t3, label="Yaw Error at t3", color="green")

            # moving legend outside of plot
            axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            if i > n:
                break
        except:
            print(f"Skipping plots for run {i}")
            continue

##########################################
#####   EXPERIMENT FUNCTIONS  ############
##########################################

def get_mean_hourly_duty(schedule_df):
    '''
    Finds the duty (total number of activations) of the turbine under the control system, adds this as a column to the df
    Computes the average hourly duty (# of turbine activations) from total duty and adds these as a column

    param schedule_df : pd.Dataframe : dataframe of turbine control schedule resulting from perfect preview-based control
    
    returns mean hourly duty : int
    '''
    schedule_df = convert_timestamps_to_datetime(schedule_df)
    # identify month, day, hour of timestamps
    schedule_df["Month"] = schedule_df["Start datetime"].dt.month
    schedule_df["Day"] = schedule_df["Start datetime"].dt.day
    schedule_df["Hour"] = schedule_df["Start datetime"].dt.hour
    # add counter of whether duty occured (T = 1, F = 0)
    schedule_df["Duty"] = schedule_df["Nacelle Position Update (T/F)"].astype(int)
    # get mean hourly duty across entire schedule
    average_hourly_duty = schedule_df.groupby(by=["Month", "Day", "Hour"]).sum().mean()["Duty"]    
    return average_hourly_duty


def plot_grid_search_results(T_range, results, field_temporal_resolution, deg_threshold, turbine_rotation_speed, control_type, graph_title, results_folder_path="../figures/", sine=True):
    '''
    TODO
    '''
    # turbines = list(results.keys())
    colors = ["blue", "red", "green", "orange"]
    metrics = ["Yaw Duty", "Mean Hourly Yaw Duty", "Mean Yaw Error", "Std Dev Yaw Error"]
    for i in range(len(metrics)):
        metric = metrics[i]
        duties, schedules = results
        color = colors[i]

        plt.figure(figsize=(12,10))
        
        if metric == "Yaw Duty":
            plt.plot(T_range, duties, color=color)
        
        elif metric == "Mean Hourly Yaw Duty":
            hourly_duties_i = [get_mean_hourly_duty(schedule_df=sched) for sched in schedules]
            plt.plot(T_range, hourly_duties_i, color=color)

        else:
            mean_yaw_errors = []
            std_dev_yaw_errors = []
            
            for schedule in schedules:
                
                mean_yew_error_i = np.mean(schedule["Delta Degree"].values.mean())
                std_dev_yaw_error_i = np.std(schedule["Delta Degree"].values.mean())
                mean_yew_error_i = math.asin(mean_yew_error_i)*180/math.pi
                std_dev_yaw_error_i = math.asin(std_dev_yaw_error_i)*180/math.pi
                mean_yaw_errors.append(mean_yew_error_i)
                std_dev_yaw_errors.append(std_dev_yaw_error_i)
                
            if metric == "Mean Yaw Error": 
                plt.plot(T_range, mean_yaw_errors, color=color)
            
            elif metric == "Std Dev Yaw Error":
                plt.plot(T_range, std_dev_yaw_errors, color=color)

        plt.suptitle(f"Effect of Varying T Forecasting Horizons on the {metric}", y=1, fontsize=12)
        if turbine_rotation_speed != math.inf and sine:
            turbine_rotation_speed = np.arcsin(turbine_rotation_speed)*180/math.pi
        plt.title(f"Field Data (1 min resolution) [threshold[°]={deg_threshold}°, control={control_type}, rot.speed[°/s]={turbine_rotation_speed},]", fontsize=10)
        plt.xlabel("T time horizon (s)")
        plt.ylabel(f"{metric}")
        plt.savefig(f"{results_folder_path}{graph_title}_{control_type}_{metric.strip()}_T_threshold={deg_threshold}.png")

def duty_T_grid_search(data_df, T_range, field_temporal_resolution, deg_threshold, turbine_rotation_speed, control_type="standard",results_folder_path="../experiments/"):
    duties = []
    schedules = []
    for T in T_range:
        logger.info(f"Started experiment with T = {T}")
        perfect_preview_update_schedule_df_i, duty_i = perfect_preview_control(data_df=data_df, T=T, temporal_resolution=field_temporal_resolution, deg_threshold=deg_threshold,turbine_rot_speed=turbine_rotation_speed,control_type=control_type)
        duties.append(duty_i)
        schedules.append(perfect_preview_update_schedule_df_i)
        # save_pickle(perfect_preview_update_schedule_df_i, f"{results_folder_path}schedule_df_T={T}.pickle")
        logger.success(f"Completed experiment with T = {T}")
       
    return duties, schedules

    
def run_field_data_T_grid_search(data_df, T_range, field_temporal_resolution, deg_threshold=0.6, turbine_rotation_speed=0.3, control_type="standard", graph_title="LiDAR", results_folder_path="../../experiments/"):
    ''' 
    param turbines : [str] : list of the names/labels of turbines in the western india wind farm
    param deg_threshold : float : the threshold of delta degree the turbine should change nacelle position
    param turbine_rotation_speed : float : the speed at which the nacelle changes direction
    param control_type : str : {"standard", "wake steering}
    param graph_title : str : custome title to include in the final plot
    '''
    results = {}
    # iterate over turbines to open datasets
    try: 
        results = duty_T_grid_search(data_df, T_range, field_temporal_resolution, deg_threshold, turbine_rotation_speed, control_type, results_folder_path)
        duties, schedules = results
        save_pickle(schedules,f"{results_folder_path}all_schedules_df.pickle")
        logger.success("Completed T grid search")
    except Exception as e:
        print(e)
        logger.error(f"Run error: {e}")

    try:   
        plot_grid_search_results(T_range, results, field_temporal_resolution, deg_threshold, turbine_rotation_speed, control_type, graph_title=graph_title,results_folder_path=results_folder_path, sine=True)
        logger.success("Plotted and saved T grid search results")
    except Exception as e:
        print("Error plotting: ", e)
        logger.error(f"Plotting error: {e}")

    return results