🚀 Advanced Object Tracking with Faster R-CNN and DeepSORT 🛠️

Object tracking project that combine Faster R-CNN for robust object detection and DeepSORT for real-time tracking. 
Here's what makes this project stand out:

1️⃣ Customizable Detection:

Utilized the COCO dataset to focus on specific classes like cars, buses, and motorcycles.
Integrated adjustable thresholds for confidence and non-maximum suppression (NMS), ensuring precise detections.
2️⃣ Flexible Tracking:

Implemented Multi-Class Tracking, enabling simultaneous tracking of multiple object types.
Added Single-Class Tracking for targeted scenarios, ideal for use cases like traffic analysis.
3️⃣ Line-Counter Integration:

Designed a line-crossing tracker to measure directional movement across virtual boundaries.
Provided comprehensive stats, visual overlays, and real-time updates.
4️⃣ Scalable Video Processing:

Optimized for live and recorded video with configurable frame resizing, skipping, and customizable FPS handling.
5️⃣ Interactive Line Drawing Tool:

Developed an intuitive interface for users to define tracking zones visually.
🎥 Results: The pipeline effectively processes complex traffic videos, delivering insights into movement patterns, object counts, and more. This powerful combination of detection and tracking opens doors for applications in smart cities, security, and autonomous systems.

💡 Technology Stack: Python, PyTorch, OpenCV, torchvision, DeepSORT
