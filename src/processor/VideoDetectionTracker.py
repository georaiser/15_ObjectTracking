import cv2
from PIL import Image
import numpy as np

# Video Processor 
class VideoDetectionTracker:
    def __init__(self, detector=None, tracker_option=None, confidence_threshold=None, resize=None):
        """
        Initializes the VideoProcessor class.
        """      
        self.detector = detector 
        self.tracker = tracker_option
        self.confidence_threshold = confidence_threshold
        self.resize = resize

    def _resize_frame(self, frame_width, frame_height):
        """
        Calculates new dimensions for resizing while maintaining aspect ratio.
        """
        if self.resize:
            min_original_size = float(min(frame_width, frame_height))
            max_original_size = float(max(frame_width, frame_height))

            scale = self.resize[0] / min_original_size
            if scale * max_original_size > self.resize[1]:
                scale = self.resize[1] / max_original_size

            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            return new_width, new_height

        else:
            return frame_width, frame_height

    def process_video(self, video_path, output_path, frame_skip, fps):
        """
        Processes a video frame by frame for object detection or tracking.
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return
    
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_width, new_height = self._resize_frame(frame_width, frame_height)
    
        if fps is None:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
    
        if frame_skip is not None:
            fps = int(fps/frame_skip)
        else:
            frame_skip = 1
                 
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
        frame_count = 0
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
    
            if frame_count % frame_skip == 0:
                if self.resize:
                    frame = cv2.resize(frame, (new_width, new_height))
    
                # Convert to PIL Image
                image = Image.fromarray(frame)
    
                # Check if tracker is None (Object Detection mode)
                if self.tracker is None:
                    # Get predictions using the global detector
                    _, boxes, labels, scores = self.detector.detect(image)
                    
                    # Display the processed frame
                    cv2.imshow('Object Detection', frame) 
                    # Write frame to output video
                    out.write(frame)
                
                else:
                    # Perform object detection and tracking
                    tracked_frame, tracked_objects, stats = self.tracker.track(image)
                    # Display the frame
                    cv2.imshow('Multi-Class Object Tracking', tracked_frame) 
                    # Write frame to output video
                    out.write(tracked_frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
            frame_count += 1
    
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    