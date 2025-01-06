import  matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np


class ObjectDetectionVisualizer:
   def __init__(self, threshold=0.8):
       self.threshold = threshold
       self.BOX_COLOR = (0, 0, 255)  # BGR red
       self.TEXT_COLOR = (255, 255, 255)  # BGR white 
       self.TEXT_BG_COLOR = (0, 0, 255)  # BGR red

   def plot_with_plt(self, image, boxes, labels, scores, classes):
       fig, ax = plt.subplots(1, figsize=(20, 6))
       ax.imshow(image)
       
       for box, label, score in zip(boxes, labels, scores):
           if score > self.threshold:
               x_min, y_min, x_max, y_max = box
               rect = patches.Rectangle(
                   (x_min, y_min), 
                   x_max - x_min, 
                   y_max - y_min,
                   linewidth=2,
                   edgecolor='r',
                   facecolor='none'
               )
               ax.add_patch(rect)
               
               class_name = classes[label]
               ax.text(
                   x_min, 
                   y_min - 10,
                   f"{class_name}: {score:.2f}",
                   color='white',
                   fontsize=10,
                   bbox=dict(facecolor='red', alpha=0.5)
               )
       
       #plt.savefig('frame_detections.jpg')
       plt.show()

   def plot_with_cv2(self, image, boxes, labels, scores, classes):
       image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       img_draw = image_bgr.copy()

       for box, label, score in zip(boxes, labels, scores):
           if score > self.threshold:
               x_min, y_min, x_max, y_max = map(int, box)
               
               # Draw box
               cv2.rectangle(img_draw, (x_min, y_min), (x_max, y_max), 
                           self.BOX_COLOR, 2)
               
               # Add label
               class_name = classes[label]
               label_text = f"{class_name}: {score:.2f}"
               
               (text_width, text_height), baseline = cv2.getTextSize(
                   label_text,
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   1
               )
               
               # Draw text background
               cv2.rectangle(
                   img_draw,
                   (x_min, y_min - text_height - baseline - 5),
                   (x_min + text_width, y_min),
                   self.TEXT_BG_COLOR,
                   cv2.FILLED
               )
               
               # Draw text
               cv2.putText(
                   img_draw,
                   label_text,
                   (x_min, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   self.TEXT_COLOR,
                   1
               )

       cv2.imwrite('frame_detections.jpg', img_draw)
       plt.figure(figsize=(20, 4))
       plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
       plt.axis('off')
       plt.show()

