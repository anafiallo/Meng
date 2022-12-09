from perfect_preview_alg import *
import os
from loguru import logger

if __name__ == "__main__":
    ## India Wind Farm data
    # turbine = "Lidar_H104m"
    # data_df = open_field_data(turbine)
    # temporal_resolution = 60 # [seconds]

    ## WHOI MVCO data
    data_df = unpickle("data/WHOI/2016_z=107m_(10, 11)-(12, 30).pickle")
    temporal_resolution = 1 # seconds
    deg_threshold = 0.0 # degrees
    rad_threshold = deg_threshold * math.pi/180
    sin_deg_threshold = np.sin(rad_threshold)
    
    deg_turbine_rotation_speed = math.inf
    if deg_turbine_rotation_speed == math.inf:
        turbine_rotation_speed = math.inf
    # else:
    #     rad_turbine_rotation_speed = deg_turbine_rotation_speed*math.pi/180
    #     turbine_rotation_speed = np.sin(rad_turbine_rotation_speed)
    
    # define range of temporal horizons to assess
    T_range = [temporal_resolution*i for i in range(1600, 1800, 5)]
    T_min = T_range[0]
    T_max = T_range[-1]

    # turbine_rotation_speed = np.sin(0.6) # math.inf
    control_type = "standard"
    results_folder_path = os.path.join(f"./experiments/pp_WHOI_lidar_h=104m_Tmin={T_min}_Tmax={T_max}_degThreshold={deg_threshold}_nacelleDegRotSpeed={deg_threshold}/")
    
    if not os.path.exists(results_folder_path):
        logger.info("Making directory path to store experiment results")
        os.makedirs(results_folder_path)

    logger.info("Starting run")

    lidar_pp_results= run_field_data_T_grid_search(data_df, T_range, temporal_resolution, deg_threshold, deg_turbine_rotation_speed, control_type, graph_title="Lidar H104m", results_folder_path=results_folder_path)