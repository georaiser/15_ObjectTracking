import cv2

class LineDrawer:
    def __init__(self, video_path, resize=None):
        """
        Initialize the LineDrawer class.

        :param video_path: Path to the video file.
        :param resize: Tuple (width, height) for resizing the video while maintaining aspect ratio.
        """
        self.video_path = video_path
        self.resize = resize
        self.lines_config = []
        self.frame_with_lines = None
        self.temp_start_point = None
        self.line_counter = 1
        self.color_palette = [
            (0, 0, 255),     # Red in BGR
            (255, 0, 0),     # Blue in BGR
            (255, 0, 255),   # Magenta in BGR 
            (0, 165, 255),   # Orange in BGR
            (0, 255, 255),   # Yellow in BGR
            (128, 0, 128),   # Purple in BGR 
            (0, 128, 128),   # Olive in BGR
            (0, 255, 0)      # Green in BGR 
        ]
        self._load_video_frame()
    
    def _load_video_frame(self, frame_number=60):
        """
        Load a specific frame from the video and resize it if needed.
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # Skip frames
        for _ in range(frame_number):
            ret, frame = cap.read()
            if not ret:
                # If we can't read the specified frame, fall back to first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                break
        
        cap.release()
    
        if not ret:
            raise RuntimeError(f"Failed to read the video: {self.video_path}")
    
        # Resize the frame if the resize tuple is provided
        self.frame_with_lines = self._resize_frame(frame)
    
    def _resize_frame(self, frame):
        """
        Resize the frame while maintaining the aspect ratio.
        """
        if self.resize:
            frame_height, frame_width = frame.shape[:2]
            min_original_size = float(min(frame_width, frame_height))
            max_original_size = float(max(frame_width, frame_height))

            scale = self.resize[0] / min_original_size
            
            if scale * max_original_size > self.resize[1]:
                scale = self.resize[1] / max_original_size

            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        else:
            return frame

    def _draw_line(self, event, x, y, flags, param):
        """
        Callback function to handle mouse events and draw lines.

        :param event: Mouse event.
        :param x: X-coordinate of the event.
        :param y: Y-coordinate of the event.
        :param flags: Additional flags for the event.
        :param param: Additional parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # On left mouse button click
            if self.temp_start_point is None:
                # Store the first point of the line
                self.temp_start_point = (x, y)
            else:
                # Store the second point, draw the line, and reset
                temp_end_point = (x, y)

                # Use a color from the palette
                color = self.color_palette[(self.line_counter - 1) % len(self.color_palette)]

                # Define the line configuration
                line_config = {
                    'start_point': self.temp_start_point,
                    'end_point': temp_end_point,
                    'name': f'Line{self.line_counter}',
                    'color': color
                }

                self.lines_config.append(line_config)
                self.line_counter += 1

                # Draw the line on the frame
                cv2.line(self.frame_with_lines, line_config['start_point'], line_config['end_point'], color, 2)
                cv2.imshow("Draw Lines", self.frame_with_lines)

                # Reset the start point
                self.temp_start_point = None

    def run(self):
        """
        Start the line drawing interaction.
        """
        cv2.namedWindow("Draw Lines")
        cv2.setMouseCallback("Draw Lines", self._draw_line)

        print("Click to define points for lines (two clicks per line). Press 'q' to quit.")
        while True:
            cv2.imshow("Draw Lines", self.frame_with_lines)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to quit
                break
            # Saving the frame
            cv2.imwrite('frame.jpg', self.frame_with_lines)

        cv2.destroyAllWindows()
        return self.lines_config
        