from src.utils import load_json, timer
from src.config import CAMERA_CALIBRATION_ROOT, GROUND_TRUTH_ROOT, SPACE_MAP_ROOT 
from pathlib import Path
from PIL import Image
from typing import List
from src.entities.sensor import Sensor
from src.entities.unit import Unit
from collections import Counter

class SmartSpace:

    def __init__(self):
        self._name : str = None
        self._path : Path = None
        self.calibration : dict = None
        self.ground_truth : dict = None
        self.map : Image = None
        self.sensors : List[Sensor] = []
        self.unit_names_in_map : List[tuple] = []
        self.units : List[Unit] = []

    def __repr__(self):
        return f"SmartSpace(name={self.name})"
        
    @property         
    def name(self):
        return self._name
    
    @property
    def path(self):
        return self._path
    
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError("SmartSpace Name should be a string.")
        self._name = value

    @path.setter
    def path(self, value):
        if not isinstance(value, Path):
            raise ValueError("Path must be a string or a Path object.")
        self._path = value

    @classmethod
    @timer
    def from_json(cls, path):

        instance = cls()
        path = Path(path)     
        instance.name = path.name 
        instance.path = path
        instance.calibration = load_json(path / CAMERA_CALIBRATION_ROOT)
        instance.ground_truth = load_json(path / GROUND_TRUTH_ROOT)
        instance._load_map(path / SPACE_MAP_ROOT)
        instance._load_sensors()
        instance._get_unit_names_in_map()

        return instance
    
    def _load_map(self, map_path):
        try:
            self.map = Image.open(map_path)
            self._crop_room_area()

        except FileNotFoundError:
            print(f"Error: Map file not found at {map_path} from {self.path}")
        except Exception as e:
            print(f"Error loading map: {e} from {self.path}")

    def _get_unit_names_in_map(self):

        for frame in self.ground_truth.items(): 
            for unit in frame[1]:
                obj_type = unit.get('object type')
                obj_id = unit.get('object id')
                if (obj_type, obj_id) not in self.unit_names_in_map:
                    self.unit_names_in_map.append((obj_type, obj_id))

        for unit_in_map in self.unit_names_in_map:
            unit = Unit.from_dict(unit_in_map, self.ground_truth)
            self.units.append(unit)
    
    def _load_sensors(self):
        self.sensors = [Sensor.from_dict(sensor_data) for sensor_data in self.calibration['sensors']]

    def analytics(self):
        """
        Returns basic analytics of the SmartSpace.
        """
        print(f'SmartSpace Name: {self.name}')
        print(f'Sensors: {len(self.sensors)}')
        print(f'Frames: {len(self.ground_truth)}')
        print("Units type counts-")
        for obj_type, count in Counter(obj[0] for obj in self.unit_names_in_map).items():
            print(f"    {obj_type}: {count}")
        print(f'    Total: {len(self.unit_names_in_map)}')

    def get_units(self, unit_id : List[str] = None) -> List[Unit]:
        """
        Returns a list of Unit objects based on the provided unit IDs.
        Args:
            unit_id: List of unit IDs to filter by. If None, returns all units.
        Returns:
            List of Unit objects matching the provided IDs.
        """
        if unit_id is None:
            return self.units
        else:
            return [unit for unit in self.units if unit.id in unit_id]
        
    def get_sensors(self, sensor_id : List[str] = None) -> List[Sensor]:
        """
        Returns a list of Sensor objects based on the provided sensor IDs.
        Args:
            sensor_id: List of sensor IDs to filter by. If None, returns all sensors.
        Returns:
            List of Sensor objects matching the provided IDs.
        """
        if sensor_id is None:
            return self.sensors
        else:
            return [sensor for sensor in self.sensors if sensor.id in sensor_id]

