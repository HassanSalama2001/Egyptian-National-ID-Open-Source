import numpy as np
import cv2

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 coordinates in the order: top-left, top-right, bottom-right, bottom-left.
    Args:
        pts: Array of shape (4, 2)
    Returns:
        Ordered array of shape (4, 2)
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left point has the smallest sum, bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right has the smallest difference, bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Apply a perspective transform to extract a top-down view of the card.
    Args:
        image: Original image
        pts: 4 corner points (as detected)
        target_width: Desired width of result
        target_height: Desired height of result
    Returns:
        The rectified card image
    """
    rect = order_points(pts)
    
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    
    return warped
