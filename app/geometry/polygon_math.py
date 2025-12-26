"""
Polygon-based geometric calculations using Shapely.
Optimized with prepared geometry for repeated containment checks.
"""

from typing import List, Tuple, Optional, Set
from shapely.geometry import Point, Polygon
from shapely.prepared import prep


class PolygonZone:
    """
    Wrapper around Shapely polygon with prepared geometry for fast PIP checks.
    """
    
    def __init__(self, vertices: List[tuple]):
        """
        Initialize polygon zone.
        
        Args:
            vertices: List of (x, y) tuples defining polygon vertices
        """
        self._vertices = vertices
        self._polygon = Polygon(vertices)
        self._prepared = prep(self._polygon)
    
    @property
    def vertices(self) -> List[tuple]:
        return self._vertices
    
    @property
    def polygon(self) -> Polygon:
        return self._polygon
    
    def contains(self, point: tuple) -> bool:
        """
        Check if point is inside polygon (fast, uses prepared geometry).
        
        Args:
            point: (x, y) tuple
        
        Returns:
            True if point is inside polygon
        """
        return self._prepared.contains(Point(point))
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if point is inside polygon (convenience method).
        
        Args:
            x: X coordinate
            y: Y coordinate
        
        Returns:
            True if point is inside polygon
        """
        return self._prepared.contains(Point(x, y))
    
    @property
    def area(self) -> float:
        """Get polygon area"""
        return self._polygon.area
    
    @property
    def bounds(self) -> tuple:
        """Get bounding box (minx, miny, maxx, maxy)"""
        return self._polygon.bounds
    
    def quick_reject(self, point: tuple) -> bool:
        """
        Quick check if point is outside bounding box.
        Use before contains() for potential speedup.
        
        Args:
            point: (x, y) tuple
        
        Returns:
            True if point is definitely outside (can skip contains check)
            False if point might be inside (need to check contains)
        """
        minx, miny, maxx, maxy = self.bounds
        x, y = point
        return x < minx or x > maxx or y < miny or y > maxy


def create_polygon_zone(vertices: List[tuple]) -> PolygonZone:
    """
    Factory function to create a PolygonZone.
    
    Args:
        vertices: List of (x, y) tuples
    
    Returns:
        PolygonZone instance
    """
    return PolygonZone(vertices)


def point_in_polygon_simple(vertices: List[tuple], point: tuple) -> bool:
    """
    Simple point-in-polygon using ray casting (without Shapely).
    Use for one-off checks where creating PolygonZone is overhead.
    
    Args:
        vertices: List of (x, y) tuples defining polygon
        point: (x, y) tuple to check
    
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(vertices)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


def detect_zone_transition(
    prev_inside: bool,
    curr_inside: bool
) -> Optional[str]:
    """
    Detect if there was a zone boundary crossing.
    
    Args:
        prev_inside: Was the object inside the zone previously?
        curr_inside: Is the object inside the zone now?
    
    Returns:
        "entered": Object moved from outside to inside
        "exited": Object moved from inside to outside
        None: No transition
    """
    if not prev_inside and curr_inside:
        return "entered"
    elif prev_inside and not curr_inside:
        return "exited"
    return None


def is_spawn(
    is_first_detection: bool,
    is_inside: bool
) -> bool:
    """
    Check if this is a spawn event (first detection inside zone).
    
    Args:
        is_first_detection: Is this the first time we see this track?
        is_inside: Is the current position inside the zone?
    
    Returns:
        True if this is a spawn event
    """
    return is_first_detection and is_inside


def count_objects_in_zone(
    zone: PolygonZone,
    positions: List[tuple]
) -> int:
    """
    Count how many positions are inside a zone.
    
    Args:
        zone: PolygonZone to check
        positions: List of (x, y) positions
    
    Returns:
        Number of positions inside the zone
    """
    count = 0
    for pos in positions:
        if not zone.quick_reject(pos) and zone.contains(pos):
            count += 1
    return count


def get_objects_in_zone(
    zone: PolygonZone,
    track_positions: dict
) -> Set[int]:
    """
    Get set of track IDs that are inside a zone.
    
    Args:
        zone: PolygonZone to check
        track_positions: Dict of {track_id: (x, y)}
    
    Returns:
        Set of track IDs inside the zone
    """
    inside_ids = set()
    for track_id, pos in track_positions.items():
        if not zone.quick_reject(pos) and zone.contains(pos):
            inside_ids.add(track_id)
    return inside_ids