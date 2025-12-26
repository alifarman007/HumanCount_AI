"""
Line-based geometric calculations for tripwire counting.
Uses vector cross product for side detection and line crossing.
"""

import math
from typing import Tuple, Optional


def get_line_side(line_p1: tuple, line_p2: tuple, point: tuple) -> float:
    """
    Calculate which side of a line a point is on using cross product.
    
    Args:
        line_p1: First point of line (x, y)
        line_p2: Second point of line (x, y)
        point: Point to check (x, y)
    
    Returns:
        > 0: Point is on LEFT side of line (when looking from p1 to p2)
        < 0: Point is on RIGHT side of line
        = 0: Point is exactly ON the line
    """
    x1, y1 = line_p1
    x2, y2 = line_p2
    x, y = point
    
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def get_line_side_discrete(
    line_p1: tuple, 
    line_p2: tuple, 
    point: tuple, 
    hysteresis: float = 0.0
) -> int:
    """
    Get discrete side of line with hysteresis threshold.
    
    Args:
        line_p1: First point of line (x, y)
        line_p2: Second point of line (x, y)
        point: Point to check (x, y)
        hysteresis: Minimum distance from line to register a side
    
    Returns:
        1: Point is on LEFT side (past hysteresis)
        -1: Point is on RIGHT side (past hysteresis)
        0: Point is within hysteresis zone (undetermined)
    """
    cross = get_line_side(line_p1, line_p2, point)
    
    # Normalize cross product by line length to get actual distance
    line_length = get_line_length(line_p1, line_p2)
    if line_length > 0:
        distance = abs(cross) / line_length
    else:
        distance = abs(cross)
    
    if distance < hysteresis:
        return 0
    
    return 1 if cross > 0 else -1


def get_distance_to_line(line_p1: tuple, line_p2: tuple, point: tuple) -> float:
    """
    Calculate perpendicular distance from point to line.
    
    Args:
        line_p1: First point of line (x, y)
        line_p2: Second point of line (x, y)
        point: Point to check (x, y)
    
    Returns:
        Perpendicular distance from point to line
    """
    x1, y1 = line_p1
    x2, y2 = line_p2
    x, y = point
    
    line_length = get_line_length(line_p1, line_p2)
    
    if line_length == 0:
        # Line is a point, return distance to that point
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    
    # Distance formula using cross product
    cross = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))
    return cross / line_length


def get_line_length(p1: tuple, p2: tuple) -> float:
    """Calculate length of line segment."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def detect_line_crossing(
    line_p1: tuple,
    line_p2: tuple,
    prev_position: tuple,
    curr_position: tuple,
    hysteresis: float = 0.0
) -> Optional[str]:
    """
    Detect if movement from prev_position to curr_position crossed the line.
    
    Args:
        line_p1: First point of line
        line_p2: Second point of line
        prev_position: Previous position of object
        curr_position: Current position of object
        hysteresis: Minimum distance past line to trigger crossing
    
    Returns:
        "left_to_right": Crossed from left side to right side
        "right_to_left": Crossed from right side to left side
        None: No crossing detected
    """
    prev_side = get_line_side_discrete(line_p1, line_p2, prev_position, hysteresis)
    curr_side = get_line_side_discrete(line_p1, line_p2, curr_position, hysteresis)
    
    # No crossing if either position is in hysteresis zone
    if prev_side == 0 or curr_side == 0:
        return None
    
    # No crossing if on same side
    if prev_side == curr_side:
        return None
    
    # Crossing detected!
    if prev_side > 0 and curr_side < 0:
        return "left_to_right"
    else:
        return "right_to_left"


def get_crossing_direction_as_count(
    crossing: str,
    entry_direction: str
) -> Optional[str]:
    """
    Convert crossing direction to entry/exit based on configuration.
    
    Args:
        crossing: "left_to_right" or "right_to_left"
        entry_direction: Configured entry direction
    
    Returns:
        "entry" or "exit"
    """
    if crossing is None:
        return None
    
    # Map entry_direction to expected crossing for entry
    entry_crossings = {
        "left_to_right": "left_to_right",
        "right_to_left": "right_to_left",
        "top_to_bottom": "left_to_right",  # Top is treated as left
        "bottom_to_top": "right_to_left"   # Bottom is treated as right
    }
    
    expected_entry = entry_crossings.get(entry_direction, "left_to_right")
    
    if crossing == expected_entry:
        return "entry"
    else:
        return "exit"


def point_on_line_segment(
    line_p1: tuple, 
    line_p2: tuple, 
    point: tuple, 
    tolerance: float = 1.0
) -> bool:
    """
    Check if a point lies on a line segment (within tolerance).
    
    Args:
        line_p1: First endpoint of segment
        line_p2: Second endpoint of segment
        point: Point to check
        tolerance: Maximum distance from line to be considered "on" it
    
    Returns:
        True if point is on the segment
    """
    # Check if point is within bounding box of segment
    min_x = min(line_p1[0], line_p2[0]) - tolerance
    max_x = max(line_p1[0], line_p2[0]) + tolerance
    min_y = min(line_p1[1], line_p2[1]) - tolerance
    max_y = max(line_p1[1], line_p2[1]) + tolerance
    
    if not (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y):
        return False
    
    # Check distance to line
    distance = get_distance_to_line(line_p1, line_p2, point)
    return distance <= tolerance