# MultiClassTracker to support line counting

import cv2
import numpy as np
from PIL import Image

from deep_sort_realtime.deepsort_tracker import DeepSort

class MultiClassTracker:
    def __init__(self, detector, tracker_name, max_age, line_counter=None, target_classes=None):
        """
        Initialize multi-class tracker
        
        :param detector: Object detector
        :param tracker: tracker algorithm
        :param max_age: Maximum number of frames to track a lost object
        :param line_counter: Optional LineCounter for tracking line crossings
        :param target_classes: List of classes to track (default: ['car', 'bus', 'motorcycle'])
        """
        self.detector = detector
        self.tracker_name = tracker_name

        self.max_age = max_age
        self.line_counter = line_counter
        
        # Initialize tracker
        if self.tracker_name == 'deepsort':
            self.tracker = DeepSort(max_age=self.max_age)
        
        # Default target classes if not specified
        self.target_classes = target_classes or ['car', 'bus', 'motorcycle']
           
        # Dictionary to store class-specific tracking information
        self.class_trackers = {}

    def track(self, frame):
        """
        Perform object detection and tracking on input frame
        
        :param frame: Input video frame
        :return: Processed frame, tracked objects, and line crossing stats
        """
        # Convert frame to numpy array if it's a PIL Image
        if isinstance(frame, Image.Image):
            frame_array = np.array(frame)
        else:
            frame_array = frame
            
        # Perform detection
        _, boxes, labels, scores = self.detector.detect(frame)  
        
        # Filter detections for target classes
        target_indices = [
            i for i, label in enumerate(labels) 
            if self.detector.get_class_name(label) in self.target_classes
        ]
        
        # Apply filtering
        filtered_boxes = boxes[target_indices]
        filtered_labels = labels[target_indices]
        filtered_scores = scores[target_indices]
        
        # Track objects
        tracked_objects = {}
        
        # Prepare detections for DeepSORT
        detections = []
        for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
            # Convert bbox format
            x1, y1, x2, y2 = box
            detection = (
                [x1, y1, x2 - x1, y2 - y1],  # [left, top, width, height]
                score,  # confidence
                label  # detection class
            )
            detections.append(detection)
        
        # Update tracking
        tracked_class_objects = self.tracker.update_tracks(
            detections, 
            frame=frame_array
        )
        
        # Process tracked objects
        for track in tracked_class_objects:
            if not track.is_confirmed() or track.time_since_update > self.max_age:
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Get bounding box [left, top, right, bottom]
            
            # Get class name
            label = track.det_class
            class_name = self.detector.get_class_name(label)
            
            # Store tracked object
            if class_name not in tracked_objects:
                tracked_objects[class_name] = []
            
            tracked_objects[class_name].append({
                'bbox': ltrb,
                'track_id': track_id
            })
        
        # Visualize tracked objects
        self._draw_tracked_objects(frame_array, tracked_objects)

        # Update line counter if it exists
        stats = None
        if self.line_counter:
            self.line_counter.update(tracked_objects)        
            # Draw lines and stats
            frame_array = self.line_counter.draw_stats(frame_array)           
            # Get stats
            stats = self.line_counter.get_stats()
        
        return frame_array, tracked_objects, stats

    def _draw_tracked_objects(self, frame, tracked_objects):
        """
        Visualize tracked objects with class labels and track IDs
        
        :param frame: Frame to draw on
        :param tracked_objects: Dictionary of tracked objects by class
        """
        for class_name, class_objects in tracked_objects.items():
            for obj in class_objects:
                bbox = obj['bbox']
                track_id = obj['track_id']
                
                # Draw bounding box
                cv2.rectangle(
                    frame, 
                    (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), 
                    (0, 255, 0), 
                    1)
                
                # Draw track ID and class name
                label_track = f'{class_name[:3]}{track_id}'
                cv2.putText(
                    frame, 
                    label_track, 
                    (int(bbox[0]), int(bbox[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1)