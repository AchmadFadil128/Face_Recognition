import cv2
import os
import numpy as np
from deepface import DeepFace
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import threading
from queue import Queue

class OptimizedFaceRecognitionSystem:
    def __init__(self, database_path="database", confidence_threshold=0.4, 
                 embeddings_file="face_embeddings.pkl"):
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.embeddings_file = embeddings_file
        self.known_embeddings = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Performance optimization settings
        self.process_scale = 0.5  # Process frames at 50% size
        self.skip_frames = 2  # Process every 3rd frame
        self.frame_counter = 0
        self.last_results = []
        
        # Threading for background processing
        self.recognition_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing_thread = None
        self.stop_processing = False
        
        # Load or create embeddings
        self.load_or_create_embeddings()
        
    def load_or_create_embeddings(self):
        """Load precomputed embeddings or create them if they don't exist"""
        if os.path.exists(self.embeddings_file):
            print("Loading precomputed embeddings...")
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                print(f"Loaded embeddings for {len(self.known_embeddings)} people")
                return
            except Exception as e:
                print(f"Error loading embeddings: {e}")
        
        print("Creating embeddings from database...")
        self.create_embeddings()
    
    def create_embeddings(self):
        """Create embeddings for all images in the database"""
        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' not found!")
            return
        
        self.known_embeddings = {}
        total_images = 0
        
        for person_name in os.listdir(self.database_path):
            person_folder = os.path.join(self.database_path, person_name)
            
            if os.path.isdir(person_folder):
                print(f"Processing {person_name}...")
                person_embeddings = []
                
                for image_file in os.listdir(person_folder):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(person_folder, image_file)
                        
                        try:
                            # Generate embedding using DeepFace
                            embedding = DeepFace.represent(
                                img_path=image_path,
                                model_name='Facenet512',  # Faster model
                                enforce_detection=False
                            )
                            
                            if isinstance(embedding, list) and len(embedding) > 0:
                                person_embeddings.append(embedding[0]['embedding'])
                                total_images += 1
                                print(f"  Processed {image_file}")
                            
                        except Exception as e:
                            print(f"  Error processing {image_file}: {e}")
                            continue
                
                if person_embeddings:
                    self.known_embeddings[person_name] = person_embeddings
                    print(f"  Created {len(person_embeddings)} embeddings for {person_name}")
        
        # Save embeddings to file
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
            print(f"Saved embeddings to {self.embeddings_file}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
        
        print(f"Total embeddings created: {total_images} for {len(self.known_embeddings)} people")
    
    def get_face_embedding(self, face_image):
        """Get embedding for a face image"""
        try:
            # Resize face for faster processing
            face_resized = cv2.resize(face_image, (112, 112))
            
            embedding = DeepFace.represent(
                img_path=face_resized,
                model_name='Facenet512',
                enforce_detection=False
            )
            
            if isinstance(embedding, list) and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                return None
                
        except Exception as e:
            return None
    
    def recognize_face_by_embedding(self, face_embedding):
        """Recognize face using precomputed embeddings"""
        if face_embedding is None:
            return "Unknown", 0
        
        best_match = None
        best_similarity = 0
        
        for person_name, embeddings_list in self.known_embeddings.items():
            for stored_embedding in embeddings_list:
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [face_embedding], 
                    [stored_embedding]
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name
        
        # Convert similarity to confidence percentage
        confidence_threshold = 1 - self.confidence_threshold
        if best_similarity > confidence_threshold:
            confidence = best_similarity * 100
            return best_match, confidence
        else:
            return "Unknown", 0
    
    def process_frame_background(self):
        """Background thread for face recognition processing"""
        while not self.stop_processing:
            try:
                if not self.recognition_queue.empty():
                    faces_data = self.recognition_queue.get(timeout=0.1)
                    results = []
                    
                    for face_data in faces_data:
                        face_image = face_data['image']
                        bbox = face_data['bbox']
                        
                        # Get embedding and recognize
                        embedding = self.get_face_embedding(face_image)
                        name, confidence = self.recognize_face_by_embedding(embedding)
                        
                        results.append({
                            'bbox': bbox,
                            'name': name,
                            'confidence': confidence
                        })
                    
                    # Put results in queue (non-blocking)
                    if not self.result_queue.full():
                        self.result_queue.put(results)
                        
            except:
                continue
    
    def detect_faces_optimized(self, frame):
        """Optimized face detection with lower resolution"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, None, fx=self.process_scale, fy=self.process_scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in smaller frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(30, 30),
            maxSize=(150, 150)
        )
        
        # Scale coordinates back to original frame size
        scale_factor = 1 / self.process_scale
        faces_data = []
        
        for (x, y, w, h) in faces:
            # Scale back coordinates
            x = int(x * scale_factor)
            y = int(y * scale_factor)
            w = int(w * scale_factor)
            h = int(h * scale_factor)
            
            # Extract face region from original frame
            face_region = frame[y:y+h, x:x+w]
            
            faces_data.append({
                'image': face_region,
                'bbox': (x, y, w, h)
            })
        
        return faces_data
    
    def draw_results(self, frame, results):
        """Draw bounding boxes and labels on the frame"""
        for result in results:
            x, y, w, h = result['bbox']
            name = result['name']
            confidence = result['confidence']
            
            # Choose color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{name}"
            if confidence > 0:
                label += f" ({confidence:.1f}%)"
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def run(self):
        """Main function to run the optimized face recognition system"""
        print("Starting optimized face recognition system...")
        print("Press 'q' to quit, 'r' to reload embeddings")
        
        # Initialize camera with lower resolution
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        # Start background processing thread
        self.processing_thread = threading.Thread(target=self.process_frame_background)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process every nth frame
                self.frame_counter += 1
                if self.frame_counter % (self.skip_frames + 1) == 0:
                    # Detect faces
                    faces_data = self.detect_faces_optimized(frame)
                    
                    if faces_data and self.recognition_queue.empty():
                        self.recognition_queue.put(faces_data)
                
                # Get latest results from background processing
                if not self.result_queue.empty():
                    self.last_results = self.result_queue.get()
                
                # Draw results on frame
                if self.last_results:
                    frame = self.draw_results(frame, self.last_results)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter >= 30:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                    print(f"FPS: {fps:.1f}")
                
                # Display frame
                cv2.imshow('Optimized Face Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Reloading embeddings...")
                    self.stop_processing = True
                    if self.processing_thread.is_alive():
                        self.processing_thread.join()
                    self.create_embeddings()
                    self.stop_processing = False
                    self.processing_thread = threading.Thread(target=self.process_frame_background)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
                
        except KeyboardInterrupt:
            print("\nStopping face recognition system...")
        
        finally:
            # Clean up
            self.stop_processing = True
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)
            cap.release()
            cv2.destroyAllWindows()
            print("Face recognition system stopped")

def main():
    # Create optimized face recognition system
    system = OptimizedFaceRecognitionSystem(
        database_path="database",
        confidence_threshold=0.4,  # Lowered for better matches
        embeddings_file="face_embeddings.pkl"
    )
    
    # Run the system
    system.run()

if __name__ == "__main__":
    main()