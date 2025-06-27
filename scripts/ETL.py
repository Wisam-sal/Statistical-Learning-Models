from src.entities.space import SmartSpace
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import list_files_in_folder
from src.feature_extractor import get_warehouse_sensor_unit_areas, get_unit_statistics
import pandas as pd

def main():
    warehouse_paths = list_files_in_folder(RAW_DATA_DIR)

    all_units_statistics = []

    for warehouse_path in warehouse_paths:
        warehouse = SmartSpace.from_json(warehouse_path)
        df = get_warehouse_sensor_unit_areas(warehouse)
        print(f"DataFrame shape: {df.shape}")
        df.to_csv(PROCESSED_DATA_DIR / f"{warehouse.name}_bounding_box_areas.csv", index=False)
        print(f"Processed {warehouse.name} and saved the DataFrame.")

        warehouse_sensor_ids = [sensor.id for sensor in warehouse.sensors]
        for unit in warehouse.units:
            unit_stats = get_unit_statistics(unit, warehouse_sensor_ids)
            unit_stats['WAREHOUSE'] = warehouse.name
            all_units_statistics.append(unit_stats)

    all_units_statistics = pd.concat(all_units_statistics)
    all_units_statistics.reset_index(inplace = True, drop = True)
    all_units_statistics.to_csv(PROCESSED_DATA_DIR/"all_units_statistics.csv", index=False)

if __name__ == "__main__":
    main()
