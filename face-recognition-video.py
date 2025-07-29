import cv2
import os
import numpy as np
from deepface import DeepFace
import time
from collections import defaultdict

# Specify the path to your video file here
VIDEO_PATH = "Test1.mp4"  # Change this to your .mp4 file

class FaceRecognitionSystem:
    def __init__(self, database_path="database", confidence_threshold=0.6):
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.known_faces = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognition_cache = {}
        self.cache_timeout = 2  # Cache results for 2 seconds
        self.load_database()
    def load_database(self):
        print("Loading face database...")
        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' not found!")
            return
        for person_name in os.listdir(self.database_path):
            person_folder = os.path.join(self.database_path, person_name)
            if os.path.isdir(person_folder):
                self.known_faces[person_name] = []
                for image_file in os.listdir(person_folder):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(person_folder, image_file)
                        self.known_faces[person_name].append(image_path)
                print(f"Loaded {len(self.known_faces[person_name])} images for {person_name}")
        print(f"Database loaded with {len(self.known_faces)} people")
    def recognize_face(self, face_image):
        try:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            best_match = None
            best_distance = float('inf')
            for person_name, image_paths in self.known_faces.items():
                for image_path in image_paths:
                    try:
                        result = DeepFace.verify(
                            img1_path=face_rgb,
                            img2_path=image_path,
                            model_name='Dlib',
                            distance_metric='cosine'
                        )
                        distance = result['distance']
                        if distance < best_distance and result['verified']:
                            best_distance = distance
                            best_match = person_name
                    except Exception as e:
                        continue
            if best_match and best_distance < self.confidence_threshold:
                confidence = (1 - best_distance) * 100
                return best_match, confidence
            else:
                return "Unknown", 0
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0
    def detect_and_recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        results = []
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            cache_key = f"{x}_{y}_{w}_{h}_{int(time.time() / self.cache_timeout)}"
            if cache_key in self.recognition_cache:
                name, confidence = self.recognition_cache[cache_key]
            else:
                name, confidence = self.recognize_face(face_region)
                self.recognition_cache[cache_key] = (name, confidence)
                current_time = int(time.time() / self.cache_timeout)
                keys_to_remove = [k for k in self.recognition_cache.keys() 
                                if int(k.split('_')[-1]) < current_time - 1]
                for k in keys_to_remove:
                    del self.recognition_cache[k]
            results.append({
                'bbox': (x, y, w, h),
                'name': name,
                'confidence': confidence
            })
        return results
    def draw_results(self, frame, results):
        for result in results:
            x, y, w, h = result['bbox']
            name = result['name']
            confidence = result['confidence']
            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name}"
            if confidence > 0:
                label += f" ({confidence:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        return frame
    def run(self):
        print("Starting face recognition system on video file...")
        print("Press 'q' to quit, 'r' to reload database")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video file {VIDEO_PATH}")
            return
        fps_counter = 0
        fps_start_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame.")
                    break
                results = self.detect_and_recognize_faces(frame)
                frame = self.draw_results(frame, results)
                fps_counter += 1
                if fps_counter >= 30:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                    print(f"FPS: {fps:.1f}")
                cv2.imshow('Face Recognition System (Video)', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Reloading database...")
                    self.known_faces = {}
                    self.recognition_cache = {}
                    self.load_database()
        except KeyboardInterrupt:
            print("\nStopping face recognition system...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Face recognition system stopped")
def main():
    system = FaceRecognitionSystem(
        database_path="database",
        confidence_threshold=0.6
    )
    system.run()
if __name__ == "__main__":
    main() 