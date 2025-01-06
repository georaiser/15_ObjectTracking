# Description: LineCounter class to track object crossings over multiple lines.

import cv2
import numpy as np

class LineCounter:
    def __init__(self, lines_config):
        """
        Initialize the LineCounter with multiple lines.
        """
        self.lines = []

        for i, line_config in enumerate(lines_config):
            line = {
                'start_point': line_config['start_point'],
                'end_point': line_config['end_point'],
                'name': line_config.get('name', f'Line {i + 1}'),
                'color': line_config.get('color'),
                'counts': {
                    'up': {},
                    'down': {}
                },
                'tracked_objects': {}
            }
            self.lines.append(line)

    def _compute_line_side(self, point, line):
        """
        Determine which side of the line a point is on.
        """
        x, y = point
        x1, y1 = line['start_point']
        x2, y2 = line['end_point']
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    def _is_bbox_center_on_line(self, bbox, line):
        """
        Check if the bounding box center touches the line.
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        center_point = (center_x, center_y)

        line_start = line['start_point']
        line_end = line['end_point']

        # Calculate the distance from the point to the line
        line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
        point_vec = np.array([center_x - line_start[0], center_y - line_start[1]])
        line_length = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_length
        projection_length = np.dot(point_vec, line_unit_vec)
        projection_vec = line_unit_vec * projection_length
        closest_point = np.array(line_start) + projection_vec

        # Threshold distance to consider "touching"
        threshold_distance = line_length * 0.55

        # Check if closest point is on the line segment
        if (
            min(line_start[0], line_end[0]) <= closest_point[0] <= max(line_start[0], line_end[0]) and
            min(line_start[1], line_end[1]) <= closest_point[1] <= max(line_start[1], line_end[1])
        ):
            # Calculate distance from bbox center to line
            distance = np.linalg.norm(np.array([center_x, center_y]) - closest_point)
            return distance < threshold_distance

        return False

    def update(self, tracked_objects):
        """
        Update line crossings for both single and multi-class tracking.
        """
       
        if isinstance(tracked_objects, dict):
            objects = [
                {**obj, 'class': class_name}
                for class_name, class_objects in tracked_objects.items()
                for obj in class_objects
            ]
        else:
            objects = tracked_objects
    
        for line in self.lines:
            for obj in objects:
                track_id = obj['track_id']
                class_name = obj.get('class', 'default')
                bbox = obj['bbox']

                # Debug: Log bbox and line details
                #print(f"Processing Track ID: {track_id}, BBox: {bbox}, Line: {line['name']}")

                # Skip objects whose center does not touch the line
                if not self._is_bbox_center_on_line(bbox, line):
                    continue

                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                object_center = (center_x, center_y)

                # Initialize object tracking state
                if track_id not in line['tracked_objects']:
                    line['tracked_objects'][track_id] = {
                        'prev_line_pos': self._compute_line_side(object_center, line),
                        'crossed': False,
                        'class': class_name
                    }
                
                current_state = line['tracked_objects'][track_id]
                current_line_pos = self._compute_line_side(object_center, line)

                # Check for line crossing
                if not current_state['crossed']:
                    prev_line_pos = current_state['prev_line_pos']

                    if prev_line_pos * current_line_pos <= 0:
                        # direction
                        direction = 'down' if current_line_pos > 0 else 'up'

                        if class_name not in line['counts'][direction]:
                            line['counts'][direction][class_name] = 0

                        # Increment class-specific count
                        line['counts'][direction][class_name] += 1
                        # Mark as crossed to prevent multiple counting
                        current_state['crossed'] = True

                        # Debug: Log crossing event
                        #print(f"Object {track_id} ({class_name}) crossed {line['name']} going {direction}.")

                    # Update previous line position
                    current_state['prev_line_pos'] = current_line_pos


    def draw_stats(self, frame):
        """
        Visualize lines and crossing statistics on the frame in a matrix-like format.
        """
        # Font configurations
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        line_spacing = 20  # line spacing
    
        # Base vertical offset
        base_y_offset = 30
    
        # Collect all unique classes across all lines
        all_classes = set()
        for line in self.lines:
            for direction in ['up', 'down']:
                all_classes.update(line['counts'][direction].keys())
        
        # Sort classes for consistent ordering
        sorted_classes = sorted(list(all_classes))
    
        for line_index, line in enumerate(self.lines):
            # Draw the line
            cv2.line(
                frame,
                line['start_point'],
                line['end_point'],
                line['color'],
                3)
        
            # Vertical offset for this specific line
            y_offset = base_y_offset + (line_index * 100)  # Reduced spacing between line stats
            
            # Prepare overall line summary
            total_up = sum(line['counts']['up'].values())
            total_down = sum(line['counts']['down'].values())
            
            # Overall line summary
            line_summary = f"{line['name']}: Up: {total_up}, Down: {total_down}"
            
            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                line_summary, font, font_scale, font_thickness
            )
            
            # Draw background rectangle for line summary
            cv2.rectangle(
                frame, 
                (10, y_offset - text_height - baseline), 
                (10 + text_width, y_offset + baseline), 
                (200, 200, 200),    # Medium light grey
                cv2.FILLED
            )         
            # Draw line summary text
            cv2.putText(
                frame,
                line_summary,
                (10, y_offset),
                font,
                font_scale,
                line['color'],
                font_thickness
            )
            
            # Increment y_offset
            y_offset += line_spacing + 10
    
            # Prepare matrix-like display of class counts
            # Create a grid-like layout
            x_start = 10
            cell_width = 70
            cell_height = 20
    
            # Draw column headers (classes)
            for class_idx, class_name in enumerate(sorted_classes):
                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    class_name, font, 0.5, 1
                )
                # Background for class name
                cv2.rectangle(
                    frame,
                    (x_start + (class_idx + 1) * cell_width, y_offset - text_height - baseline),
                    (x_start + (class_idx + 2) * cell_width, y_offset + baseline),
                    (220, 220, 220),
                    cv2.FILLED
                )                
                cv2.putText(
                    frame, 
                    class_name, 
                    (x_start + (class_idx + 1) * cell_width, y_offset - 2), 
                    font, 
                    0.5, 
                    (0, 0, 0),
                    1
                )
    
            y_offset += cell_height
    
            # Directions to track
            directions = [
                ('Up', (255, 255, 0)),   # Cyan in BGR
                ('Down', (255, 255, 0))   
            ]            
    
            for direction, direction_color in directions:
                # Draw direction label
                cv2.putText(
                    frame, 
                    direction, 
                    (x_start, y_offset - 2), 
                    font, 
                    0.5, 
                    direction_color,
                    1
                )
    
                # Draw class counts
                for class_idx, class_name in enumerate(sorted_classes):
                    # Get count, default to 0 if not exists
                    count = line['counts'][direction.lower()].get(class_name, 0)               
                    # Prepare count text
                    count_text = str(count)                 
                    # Calculate text size
                    (text_width, text_height), _ = cv2.getTextSize(
                        count_text, font, 0.5, 1
                    )                    
                    # Draw background rectangle
                    cv2.rectangle(
                        frame, 
                        (x_start + (class_idx + 1) * cell_width, y_offset - cell_height), 
                        (x_start + (class_idx + 2) * cell_width, y_offset), 
                        (220, 220, 220),  # Light gray background
                        cv2.FILLED
                    )                    
                    # Draw count
                    cv2.putText(
                        frame, 
                        count_text, 
                        (x_start + (class_idx + 1) * cell_width + (cell_width - text_width) // 2, 
                         y_offset - 2), 
                        font, 
                        0.5, 
                        (0, 0, 0),
                        1
                    )
    
                # Move to next row
                y_offset += cell_height
    
        return frame
        
    def get_stats(self):
        """
        Get detailed crossing statistics for all lines.
        """
        return [
            {
                'name': line['name'],
                'up_counts': line['counts']['up'],
                'down_counts': line['counts']['down']
            }
            for line in self.lines]
