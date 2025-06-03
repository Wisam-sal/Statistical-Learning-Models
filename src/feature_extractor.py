import pandas as pd
import numpy as np
from src.entities.space import SmartSpace
from src.entities.unit import Unit
from typing import List

def _get_sensor_area(bounding_box) -> float:
    """
    Calculate the area of a bounding box.
    :param bounding_box: list of bounding box coordinates [xmin, ymin, xmax, ymax]
    :return: area of the bounding box
    """
    xmin, ymin, xmax, ymax = bounding_box
    return (xmax - xmin) * (ymax - ymin)

def get_warehouse_sensor_unit_areas(warehouse:SmartSpace) -> pd.DataFrame:
    """
    Get the area of each sensor's bounding box for each unit in the warehouse.
    :param warehouse: SmartSpace object representing the warehouse
    :return: DataFrame with sensor areas and unit locations
    """
    warehouse_sensors = [sensor.id for sensor in warehouse.sensors]

    df = pd.DataFrame(columns=warehouse_sensors)
    for unit in warehouse.units: 
        for bounding_box in unit.bounding_box_visible:
            row_num = len(df)
            for warehouse_sensor in warehouse_sensors:
                bounding_box_area = _get_sensor_area(bounding_box.get(warehouse_sensor, [0, 0, 0, 0]))
                df.loc[row_num, warehouse_sensor] = bounding_box_area

    x, y, z = [], [], []
    for unit in warehouse.units:
        for loc in unit.location:
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])

    df['x_position'] = x
    df['y_position'] = y
    df['z_position'] = z
    df['unit_distance_from_origin'] = [unit.distance_from_origin(loc) for unit in warehouse.units for loc in unit.location]
    df['unit_id'] = [unit.id for unit in warehouse.units for _ in range(len(unit.location))]

    return df

def get_unit_statistics(unit: Unit, warehouse_sensor_ids: List) -> pd.DataFrame:

    unit_location_x = np.array([location[0] for location in unit.location],dtype=float)
    unit_location_y = np.array([location[1] for location in unit.location],dtype=float)
    unit_distance_from_origin = np.array(np.sqrt(unit_location_x**2 + unit_location_y**2), dtype=float)
    sensor_unit_frames = np.array([int(frame) for frame in unit.frame_id],dtype=float)
    unit_direction = np.array([rotation[:-1] for rotation in unit.bounding_box_rotation],dtype=float)
    unit_direction_unique = np.array([list(x) for x in set(tuple(direction) for direction in unit_direction)],dtype=float)

    statistical_columns = ['MIN_LOC_X', 'MIN_LOC_Y', 'MIN_DIST_ORIGIN',
                           'MAX_LOC_X', 'MAX_LOC_Y', 'MAX_DIST_ORIGIN',
                           'MEAN_LOC_X', 'MEAN_LOC_Y', 'MEAN_DIST_ORIGIN',
                           'STD_LOC_X', 'STD_LOC_Y', 'STD_DIST_ORIGIN',
                           'MEDIAN_LOC_X', 'MEDIAN_LOC_Y', 'MEDIAN_DIST_ORIGIN',
                           'RANGE_LOC_X', 'RANGE_LOC_Y', 'RANGE_DIST_ORIGIN',
                           'TOTAL_DIST_TRAVELED_X', 'TOTAL_DIST_TRAVELED_Y', 'TOTAL_DIST_TRAVELED_EUCLIDEAN',
                           'TOTAL_DIRECTION_CHANGES', 'TIME_SPENT_WALKING_SEC', 'TIME_SPENT_STANDING_SEC',
                           'RATIO_OF_CAMERA_CAPTURES',
                           'UNIT_ID','UNIT_TYPE']

    min_location_unit_x = min(unit_location_x)
    min_location_unit_y = min(unit_location_y)
    min_distance_from_origin = min(unit_distance_from_origin)

    max_location_unit_x = max(unit_location_x)
    max_location_unit_y = max(unit_location_y)
    max_distance_from_origin = max(unit_distance_from_origin)

    mean_location_unit_x = np.mean(unit_location_x)
    mean_location_unit_y = np.mean(unit_location_y)
    mean_distance_from_origin = np.mean(unit_distance_from_origin)

    std_location_unit_x = np.std(unit_location_x)
    std_location_unit_y = np.std(unit_location_y)
    std_distance_from_origin = np.std(unit_distance_from_origin)

    median_location_unit_x = np.median(unit_location_x)
    median_location_unit_y = np.median(unit_location_y)
    median_distance_from_origin = np.median(unit_distance_from_origin)

    range_location_unit_x = max_location_unit_x - min_location_unit_x
    range_location_unit_y = max_location_unit_y - min_location_unit_y
    range_distance_from_origin = max_distance_from_origin - min_distance_from_origin

    total_distance_traveled_x = np.sum(np.abs(np.diff(unit_location_x)))
    total_distance_traveled_y = np.sum(np.abs(np.diff(unit_location_y)))
    total_distance_traveled = np.sum(np.sqrt(np.diff(unit_location_x)**2 + np.diff(unit_location_y)**2))

    total_direction_changes = len(unit_direction_unique)

    _frame_diffs = np.diff(sensor_unit_frames)

    time_spent_walking_sec = np.sum(_frame_diffs[_frame_diffs < 50]) / 30
    time_spent_standing_sec = 300 - time_spent_walking_sec

    ratio_of_camera_captures = len(set([key for frame in unit.bounding_box_visible for key in frame.keys()])) / len(warehouse_sensor_ids)

    data = {
        'MIN_LOC_X': [min_location_unit_x],
        'MIN_LOC_Y': [min_location_unit_y],
        'MIN_DIST_ORIGIN': [min_distance_from_origin],
        'MAX_LOC_X': [max_location_unit_x],
        'MAX_LOC_Y': [max_location_unit_y],
        'MAX_DIST_ORIGIN': [max_distance_from_origin],
        'MEAN_LOC_X': [mean_location_unit_x],
        'MEAN_LOC_Y': [mean_location_unit_y],
        'MEAN_DIST_ORIGIN': [mean_distance_from_origin],
        'STD_LOC_X': [std_location_unit_x],
        'STD_LOC_Y': [std_location_unit_y],
        'STD_DIST_ORIGIN': [std_distance_from_origin],
        'MEDIAN_LOC_X': [median_location_unit_x],
        'MEDIAN_LOC_Y': [median_location_unit_y],
        'MEDIAN_DIST_ORIGIN': [median_distance_from_origin],
        'RANGE_LOC_X': [range_location_unit_x],
        'RANGE_LOC_Y': [range_location_unit_y],
        'RANGE_DIST_ORIGIN': [range_distance_from_origin],
        'TOTAL_DIST_TRAVELED_X': [total_distance_traveled_x],
        'TOTAL_DIST_TRAVELED_Y': [total_distance_traveled_y],
        'TOTAL_DIST_TRAVELED_EUCLIDEAN': [total_distance_traveled],
        'TOTAL_DIRECTION_CHANGES': [total_direction_changes],
        'TIME_SPENT_WALKING_SEC': [time_spent_walking_sec],
        'TIME_SPENT_STANDING_SEC': [time_spent_standing_sec],
        'RATIO_OF_CAMERA_CAPTURES': [ratio_of_camera_captures],
        'UNIT_ID': [int(unit.id)],
        'UNIT_TYPE': [unit.category]
    }

    return pd.DataFrame(data, columns=statistical_columns)
