# SingleClassTracker to support line counting

import cv2
import numpy as np
from PIL import Image

from deep_sort_realtime.deepsort_tracker import DeepSort

class SingleClassTracker:
    def __init__(self, detector, tracker_name, max_age, target_class, line_counter):
        """
        Initialize single-class tracker with optional line counter
        """
        self.detector = detector
        self.target_class = target_class
        self.tracker_name = tracker_name
        self.max_age = max_age
        self.line_counter = line_counter
        
         # Initialize tracker
        if self.tracker_name == 'deepsort':
            self.tracker = DeepSort(max_age=self.max_age)  
                 
    def track(self, frame):
        """
        Perform object detection and tracking for a specific class
        """
        # Convert frame to numpy array if it's a PIL Image
        if isinstance(frame, Image.Image):
            frame_array = np.array(frame)
        else:
            frame_array = frame

        # Perform detection
        _, boxes, labels, scores = self.detector.detect(frame)

        # Filter detections for the target class
        target_class_index = list(self.detector.classes.keys()).index(self.target_class)
        class_mask = labels == target_class_index
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        # Prepare detections for DeepSORT
        detections = []
        for bbox, score in zip(class_boxes, class_scores):
            x1, y1, x2, y2 = bbox
            detection = (
                [x1, y1, x2 - x1, y2 - y1],  # [left, top, width, height]
                score,  # confidence
                target_class_index  # detection class
            )
            detections.append(detection)

        # Update tracking
        tracked_objects = self.tracker.update_tracks(
            detections, 
            frame=frame_array
        )

        # Process tracked objects
        confirmed_tracks = []
        for track in tracked_objects:
            if not track.is_confirmed() or track.time_since_update > self.max_age:
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Get bounding box [left, top, right, bottom]
            
            confirmed_tracks.append({
                'bbox': ltrb,
                'track_id': track_id
            })

        # Visualize tracked objects
        self._draw_tracked_objects(frame_array, confirmed_tracks)

        # Update line counter if it exists
        if self.line_counter:
            self.line_counter.update(confirmed_tracks)        
            # Draw lines and stats
            frame_array = self.line_counter.draw_stats(frame_array)            
            # Optionally get stats
            stats = self.line_counter.get_stats()
            
        return frame_array, confirmed_tracks, stats
 
    def _draw_tracked_objects(self, frame, tracked_objects):
        """
        Visualize tracked objects with track IDs
        """
        for obj in tracked_objects:
            bbox = obj['bbox']
            track_id = obj['track_id']
            
            # Draw bounding box
            cv2.rectangle(
                frame, 
                (int(bbox[0]), int(bbox[1])), 
                (int(bbox[2]), int(bbox[3])), 
                (0, 255, 0), 
                1
            )
            
            # Draw track ID
            cv2.putText(
                frame, 
                f'{self.target_class} {track_id}', 
                (int(bbox[0]), int(bbox[1] - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                1
            )