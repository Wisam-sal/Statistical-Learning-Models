class Sensor:
    """
    Sensor class to represent a sensor in a smart space.
    """
    def __init__(self, 
                 type: str, 
                 id: str, 
                 position: tuple,
                 coordinates: dict,
                 glob_coordinates: dict,
                 scale_factor: float,
                 attributes: list,
                 intrinsic_matrix: list,
                 extrinsic_matrix: list,
                 camera_matrix: list,
                 homography: list):
        
        self.type : str = type
        self.id : str = id
        self.position : tuple = position
        self.coordinates : dict = coordinates
        self.glob_coordinates : dict = glob_coordinates
        self.scale_factor : float = scale_factor
        self.attributes : list = attributes
        self.intrinsic_matrix : list = intrinsic_matrix
        self.extrinsic_matrix : list = extrinsic_matrix
        self.camera_matrix : list = camera_matrix
        self.homography : list = homography

    def __repr__(self):
        return f"Sensor(type={self.type}, id={self.id}, position={self.position})"
    
    @classmethod
    def from_dict(cls, sensor_data: dict):
        """
        Create a Sensor object from a dictionary representation.
        """
        type = sensor_data.get('type')
        id = sensor_data.get('id')
        if id == 'Camera': id = 'Camera_00'
        id = id.replace('Camera_','Camera_00')
        coordinates = sensor_data.get('coordinates', {})
        position = (coordinates.get('x', 0), coordinates.get('y', 0))
        scale_factor = sensor_data.get('scaleFactor', 1.0)
        glob_coordinates = sensor_data.get('translationToGlobalCoordinates', {})

        attributes = sensor_data.get('attributes', [])
        intrinsic_matrix = sensor_data.get('intrinsicMatrix', [])
        extrinsic_matrix = sensor_data.get('extrinsicMatrix', [])
        camera_matrix = sensor_data.get('cameraMatrix', [])
        homography = sensor_data.get('homography', [])

        return cls(type=type, 
                   id=id, 
                   position=position, 
                   coordinates=coordinates,
                   scale_factor=scale_factor,
                   glob_coordinates=glob_coordinates,
                   attributes=attributes,
                   intrinsic_matrix=intrinsic_matrix,
                   extrinsic_matrix=extrinsic_matrix,
                   camera_matrix=camera_matrix,
                   homography=homography)
    
    def get_position_on_map(self) -> tuple:
        x = self.coordinates["x"] * self.scale_factor + self.glob_coordinates["x"]
        y = self.coordinates["y"] * self.scale_factor + self.glob_coordinates["y"]
        return (x, y)
