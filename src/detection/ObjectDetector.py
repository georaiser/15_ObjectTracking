# Object Detector

import torch
import torchvision
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

class ObjectDetector:
    def __init__(self, model, classes, device, 
                 confidence_threshold, nms_threshold, resize):
        """
        Initializes the Detector class.
        """
        self.model = model.eval().to(device)
        self.classes = classes
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.resize = resize

    def _resize_image(self, image):
        """
        Args:
            image: The input image (PIL Image).
        Returns:
            Resized image (PIL Image).
        """
        if self.resize:
            original_width, original_height = image.size
            min_original_size = float(min(original_width, original_height))
            max_original_size = float(max(original_width, original_height))

            scale = self.resize[0] / min_original_size
            if scale * max_original_size > self.resize[1]:
                scale = self.resize[1] / max_original_size

            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            image = T.Resize((new_height, new_width))(image)
        return image

    def detect(self, image_path):
        """
        Performs object detection on the input image.
        Args:
            image_path: Path to the image or a PIL Image object.
        """
        # Load and preprocess the image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path         
        
        image = self._resize_image(image)
        image_tensor = T.ToTensor()(image).to(self.device)

        # Run inference
        with torch.no_grad():
            predictions = self.model([image_tensor])

        boxes = predictions[0]['boxes'].detach().cpu()
        labels = predictions[0]['labels'].detach().cpu()
        scores = predictions[0]['scores'].detach().cpu()

        # Filter predictions by class target
        indices = [i for i, label in enumerate(labels) if label.item() in self.classes.values()]
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Apply Non-Maximum Suppression (NMS)
        nms_indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.nms_threshold)

        # PIL to array
        image_array = np.array(image)
        boxes = boxes[nms_indices].numpy()
        labels = labels[nms_indices].numpy()
        scores = scores[nms_indices].numpy()

        self._draw_predictions(image_array, boxes, labels, scores)

        return (image_array, boxes, labels, scores)

    def get_class_name(self, label): 
        classes_inverted = {v: k for k, v in self.classes.items()}
        class_name = classes_inverted.get(label, "Unknown")
        
        return class_name

    def _draw_predictions(self, frame, boxes, labels, scores):
        """
        Draws predictions on the video frame.
        """
        for box, label, score in zip(boxes, labels, scores):
            if score > self.confidence_threshold:               
                box = box.astype(np.int32)
                class_name = self.get_class_name(label)
                # Draw box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                # Add label
                label_text = f'{class_name}: {score:.2f}'
                cv2.putText(frame, label_text, (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        