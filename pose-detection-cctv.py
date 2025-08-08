import cv2
import numpy as np
from ultralytics import YOLO
import math
from collections import defaultdict
import time

class PersonTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, keypoints, bbox):
        """Register a new person with unique ID"""
        self.objects[self.next_id] = {
            'centroid': centroid,
            'keypoints': keypoints,
            'bbox': bbox,
            'last_seen': time.time()
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        
    def deregister(self, object_id):
        """Remove a person from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def get_centroid(self, bbox):
        """Calculate centroid from bounding box"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for better matching"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections):
        """Update tracker with new detections using improved matching"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = []
        input_keypoints = []
        input_bboxes = []
        
        for detection in detections:
            bbox, keypoints = detection
            centroid = self.get_centroid(bbox)
            input_centroids.append(centroid)
            input_keypoints.append(keypoints)
            input_bboxes.append(bbox)
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_keypoints[i], input_bboxes[i])
        else:
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_bboxes = [obj['bbox'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            
            # Combine distance and IoU for better matching
            cost_matrix = np.zeros((len(object_ids), len(input_centroids)))
            
            for i, (obj_centroid, obj_bbox) in enumerate(zip(object_centroids, object_bboxes)):
                for j, (inp_centroid, inp_bbox) in enumerate(zip(input_centroids, input_bboxes)):
                    distance = self.calculate_distance(obj_centroid, inp_centroid)
                    iou = self.calculate_iou(obj_bbox, inp_bbox)
                    
                    # Combined cost: lower is better
                    # Normalize distance and add negative IoU (higher IoU = lower cost)
                    cost = distance / self.max_distance - iou * 2
                    cost_matrix[i, j] = cost
            
            # Find optimal assignment using Hungarian-like greedy approach
            rows = cost_matrix.min(axis=1).argsort()
            cols = cost_matrix.argmin(axis=1)[rows]
            
            used_row_idxs = set()
            used_col_idxs = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue
                
                # Accept match if cost is reasonable
                if cost_matrix[row, col] < 2.0:  # Adjusted threshold
                    object_id = object_ids[row]
                    self.objects[object_id]['centroid'] = input_centroids[col]
                    self.objects[object_id]['keypoints'] = input_keypoints[col]
                    self.objects[object_id]['bbox'] = input_bboxes[col]
                    self.objects[object_id]['last_seen'] = time.time()
                    self.disappeared[object_id] = 0
                    
                    used_row_idxs.add(row)
                    used_col_idxs.add(col)
            
            unused_row_idxs = set(range(0, cost_matrix.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, cost_matrix.shape[1])).difference(used_col_idxs)
            
            if cost_matrix.shape[0] >= cost_matrix.shape[1]:
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_idxs:
                    self.register(input_centroids[col], input_keypoints[col], input_bboxes[col])
        
        return self.objects

class ClassroomPoseDetector:
    def __init__(self, model_path='yolov8s-pose.pt', max_persons=30):
        """Initialize the classroom pose detector"""
        print("Loading YOLOv8n-pose model...")
        self.model = YOLO(model_path)
        # Adjusted for CCTV: smaller distance threshold, longer patience for seated students
        self.tracker = PersonTracker(max_disappeared=90, max_distance=80)
        self.max_persons = max_persons
        
        # Define pose keypoint connections for visualization
        self.pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Colors for different person IDs
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (255, 165, 0), (255, 20, 147)
        ]
    
    def apply_custom_nms(self, detections_with_conf, iou_threshold=0.4):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections_with_conf) == 0:
            return []
        
        # Sort by confidence (highest first)
        detections_with_conf.sort(key=lambda x: x[0], reverse=True)
        
        keep = []
        while detections_with_conf:
            # Take the detection with highest confidence
            current = detections_with_conf.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU overlap
            remaining = []
            for detection in detections_with_conf:
                iou = self.calculate_iou(current[1], detection[1])
                if iou < iou_threshold:  # Keep if IoU is below threshold
                    remaining.append(detection)
            detections_with_conf = remaining
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def process_frame(self, frame):
        """Process a single frame for person detection and pose estimation"""
        # Enhanced settings for CCTV classroom footage
        # Stronger NMS to prevent double boxes
        results = self.model(frame, conf=0.2, classes=[0], imgsz=1280, 
                           iou=0.3, agnostic_nms=True, max_det=50)
        
        detections_with_conf = []
        
        # Process detections
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Handle keypoints if available, otherwise use None
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                else:
                    keypoints = [None] * len(boxes)
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    # Lower confidence threshold for small students
                    if conf > 0.15:  
                        # Filter out very small detections (likely false positives)
                        box_width = box[2] - box[0]
                        box_height = box[3] - box[1]
                        box_area = box_width * box_height
                        
                        # Minimum size filter for realistic person detections
                        if box_area > 400 and box_width > 15 and box_height > 20:
                            kpts = keypoints[i] if keypoints[i] is not None else np.zeros((17, 2))
                            
                            # Combined score: confidence + size
                            area_score = min(box_area / 5000, 1.0)  # Normalize area
                            combined_score = conf * 0.8 + area_score * 0.2
                            
                            detections_with_conf.append((combined_score, box.astype(int), kpts))
        
        # Apply additional custom NMS to catch any remaining duplicates
        filtered_detections = self.apply_custom_nms(detections_with_conf, iou_threshold=0.4)
        
        # Limit to maximum number of persons
        if len(filtered_detections) > self.max_persons:
            filtered_detections = filtered_detections[:self.max_persons]
        
        # Convert back to detection format
        detections = [(box, kpts) for _, box, kpts in filtered_detections]
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        return tracked_objects
    
    def draw_pose(self, frame, keypoints, color, thickness=2):
        """Draw pose skeleton on frame"""
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # Only draw visible keypoints
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Draw connections
        for connection in self.pose_connections:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and
                keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0):
                
                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                cv2.line(frame, pt1, pt2, color, thickness)
    
    def visualize_results(self, frame, tracked_objects):
        """Draw tracking results on frame"""
        for person_id, obj in tracked_objects.items():
            color = self.colors[person_id % len(self.colors)]
            
            # Draw bounding box
            bbox = obj['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw person ID
            cv2.putText(frame, f'Person {person_id}', 
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Draw pose
            self.draw_pose(frame, obj['keypoints'], color)
            
            # Draw centroid
            centroid = obj['centroid']
            cv2.circle(frame, centroid, 5, color, -1)
        
        # Display statistics
        cv2.putText(frame, f'Persons tracked: {len(tracked_objects)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

def main():
    """Main function to run the classroom pose detection"""
    # Initialize detector
    detector = ClassroomPoseDetector()
    
    # Open video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture('Test.mp4')  # Change to video file path if needed
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Starting classroom pose detection...")
    print("Press 'q' to quit")
    
    frame_count = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        # Process frame
        tracked_objects = detector.process_frame(frame)
        
        # Visualize results
        output_frame = detector.visualize_results(frame, tracked_objects)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps_end_time = time.time()
            fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
            print(f"FPS: {fps:.2f}, Persons: {len(tracked_objects)}")
        
        # Display frame
        cv2.imshow('Classroom Pose Detection', output_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection completed!")

if __name__ == "__main__":
    main()
