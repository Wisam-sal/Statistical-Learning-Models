"""
# Unit types = {'Forklift', 'NovaCarter', 'Person', 'Transporter'}
"""

class Unit:
    def __init__(self, 
                 category: str  = None, 
                 id : int = None, 
                 location: list = [],
                 bounding_box_scale: list = [],
                 bounding_box_rotation: list = [],
                 bounding_box_visible: list = [],
                 frame_id: list = [],
                 distance: list = []):
        self._id = id
        self._category = category
        self._location = location if location is not None else []
        self._bounding_box_scale = bounding_box_scale if bounding_box_scale is not None else []
        self._bounding_box_rotation = bounding_box_rotation if bounding_box_rotation is not None else []
        self._bounding_box_visible = bounding_box_visible if bounding_box_visible is not None else []
        self._frame_id = frame_id if frame_id is not None else []
        self._distance_from_origin = distance if distance is not None else []

    def __repr__(self):
        return f"Unit(category={self._category}, ID={self._id})"
    
    @property
    def id(self):
        return self._id
    @property
    def category(self):
        return self._category
    @property
    def location(self):
        return self._location
    @property
    def bounding_box_scale(self):
        return self._bounding_box_scale
    @property
    def bounding_box_rotation(self):
        return self._bounding_box_rotation
    @property
    def bounding_box_visible(self):
        return self._bounding_box_visible
    @property
    def frame_id(self):
        return self._frame_id
    @property
    def distance_from_origin(self):
        return self._distance_from_origin
    
    def distance_from_origin(self, loc):
        """
        2D Euclidean distance from the origin (0, 0) to the location. (center of map)
        """
        return int((loc[0]**2 + loc[1]**2)**0.5) 
        
    @classmethod
    def from_dict(cls, unit_data: tuple, ground_truth: dict = None):
        """
        Unit object from a dictionary.
        """
        location = []
        bounding_box_scale = []
        bounding_box_rotation = []
        bounding_box_visible = []
        category = unit_data[0]
        distance = []
        frame_id = []
        id = str(unit_data[1])
        for frame in ground_truth.items():
            for unit in frame[1]:
                if id == str(unit['object id']) and category == unit['object type']:
                    if cls.distance_from_origin(cls, unit.get('3d location')) not in distance:
                        location.append(unit.get('3d location'))
                        distance.append(cls.distance_from_origin(cls, unit.get('3d location')))
                        bounding_box_scale.append(unit.get('3d bounding box scale'))
                        bounding_box_rotation.append(unit.get('3d bounding box rotation'))

                        frame_box = {
                            (k.split('_')[0] + '_' + k.split('_')[1].zfill(4)) if '_' in k else k + '_0000': v
                            for k, v in unit.get('2d bounding box visible').items()
                        }

                        bounding_box_visible.append(frame_box)
                        frame_id.append(frame[0])

        return cls(category=category, 
                   id=id, 
                   location=location, 
                   bounding_box_scale=bounding_box_scale, 
                   bounding_box_rotation=bounding_box_rotation,
                   bounding_box_visible=bounding_box_visible,
                   distance=distance,
                   frame_id=frame_id)
    
