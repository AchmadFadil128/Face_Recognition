import cv2
import os
from deepface import DeepFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import threading
from queue import Queue
import requests


class OptimizedFaceRecognitionSystem:
    def __init__(self, database_path="database", confidence_threshold=0.4, 
                 embeddings_file="face_embeddings.pkl"):
        
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.embeddings_file = embeddings_file
        self.known_embeddings = {}
        
        # Initialize MediaPipe face detection with improved settings for small faces
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            # Use model 1 (full range) instead of 0 (short range) for better small face detection
            # Lower min_detection_confidence for detecting smaller/lower quality faces
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Changed from 0 to 1 for full range detection
                min_detection_confidence=0.05  # Even lower for very small faces
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except ImportError:
            print("❌ Error: MediaPipe not installed. Install with: pip install mediapipe")
            raise Exception("MediaPipe not found")
        
        # Performance optimization settings - adjusted for better small face detection
        self.process_scale = 0.95  # Even higher resolution for small faces
        self.skip_frames = 0  # Process every frame for maximum detection
        self.frame_counter = 0
        self.last_results = []
        
        # Threading for background processing
        self.recognition_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing_thread = None
        self.stop_processing = False
        
        # Manual mapping from recognized names to student numbers
        self.name_to_student_number = {
            "Achmad": "SW001",
            "Adel": "SW002",
            "Aziz": "SW003",
            "Ibrahim": "SW004",
        }
        
        # Track sent attendances to avoid duplicates per run
        self.sent_student_numbers = set()
        
        # Load or create embeddings
        self.load_or_create_embeddings()

    def send_attendance(self, detections):
        """Send attendance for newly recognized students via POST, avoiding duplicates per run."""
        if not detections:
            return
        endpoint = "http://localhost:3000/api/attendance/mark"
        for det in detections:
            name = det.get('name')
            if not name or name == "Unknown":
                continue
            student_number = self.name_to_student_number.get(name)
            if not student_number:
                continue
            if student_number in self.sent_student_numbers:
                continue
            payload = {
                "studentNumber": student_number,
                "status": "PRESENT",
            }
            requests.post(endpoint, json=payload)
            self.sent_student_numbers.add(student_number)

    def load_or_create_embeddings(self):
        """Load precomputed embeddings or create them if they don't exist"""
        if os.path.exists(self.embeddings_file):
            try:
                print(f"📂 Loading existing embeddings from {self.embeddings_file}")
                with open(self.embeddings_file, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                print(f"✅ Loaded {len(self.known_embeddings)} persons with embeddings")
                return
            except Exception as e:
                print(f"❌ Failed to load embeddings: {str(e)}")
                print("🔄 Will create new embeddings...")
        else:
            print(f"📂 No existing embeddings file found: {self.embeddings_file}")
            print("🔄 Creating new embeddings...")
        
        self.create_embeddings()
    
    def create_embeddings(self):
        """Create embeddings for all images in the database"""
        if not os.path.exists(self.database_path):
            print(f"❌ Database path '{self.database_path}' does not exist")
            return
        
        print(f"📁 Scanning database folder: {self.database_path}")
        self.known_embeddings = {}
        person_count = 0
        image_count = 0
        success_count = 0
        
        for person_name in os.listdir(self.database_path):
            person_folder = os.path.join(self.database_path, person_name)
            if not os.path.isdir(person_folder):
                print(f"⚠️  Skipping non-directory: {person_name}")
                continue
            
            print(f"👤 Processing person: {person_name}")
            person_count += 1
            person_embeddings = []
            
            for image_file in os.listdir(person_folder):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    print(f"⚠️  Skipping non-image file: {image_file}")
                    continue
                
                image_path = os.path.join(person_folder, image_file)
                image_count += 1
                print(f"🖼️  Processing image: {image_file}")
                
                try:
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name='ArcFace',
                        enforce_detection=False
                    )
                    if isinstance(embedding, list) and len(embedding) > 0:
                        person_embeddings.append(embedding[0]['embedding'])
                        success_count += 1
                        print(f"✅ Successfully created embedding for {image_file}")
                    else:
                        print(f"❌ No embedding generated for {image_file}")
                except Exception as e:
                    print(f"❌ Error processing {image_file}: {str(e)}")
                    continue
            
            if person_embeddings:
                self.known_embeddings[person_name] = person_embeddings
                print(f"✅ Added {len(person_embeddings)} embeddings for {person_name}")
            else:
                print(f"❌ No valid embeddings for {person_name}")
        
        print(f"\n📊 Summary:")
        print(f"   Persons found: {person_count}")
        print(f"   Images processed: {image_count}")
        print(f"   Successful embeddings: {success_count}")
        print(f"   Persons with embeddings: {len(self.known_embeddings)}")
        
        if self.known_embeddings:
            try:
                with open(self.embeddings_file, 'wb') as f:
                    pickle.dump(self.known_embeddings, f)
                print(f"💾 Saved embeddings to {self.embeddings_file}")
            except Exception as e:
                print(f"❌ Failed to save embeddings: {str(e)}")
        else:
            print("❌ No embeddings were created successfully")
    
    def get_face_embedding(self, face_image):
        """Get embedding for a face image with improved preprocessing"""
        try:
            # Check if face is too small
            h, w = face_image.shape[:2]
            if h < 32 or w < 32:  # If face is smaller than 32x32
                # Upscale using INTER_LANCZOS4 for better quality
                scale_factor = max(3, 80 // min(h, w))  # More aggressive upscaling
                face_image = cv2.resize(face_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
            
            # Enhance image quality for small faces
            if h < 80 or w < 80:
                # Apply bilateral filter for noise reduction
                face_image = cv2.bilateralFilter(face_image, 9, 75, 75)
                # Apply unsharp mask for sharpening
                gaussian = cv2.GaussianBlur(face_image, (0, 0), 2.0)
                face_image = cv2.addWeighted(face_image, 1.5, gaussian, -0.5, 0)
            
            # Resize to optimal size for ArcFace (112x112)
            face_resized = cv2.resize(face_image, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            
            embedding = DeepFace.represent(
                img_path=face_resized,
                model_name='ArcFace',
                enforce_detection=False
            )
            
            if isinstance(embedding, list) and len(embedding) > 0:
                return embedding[0]['embedding']
            else:
                return None
                
        except Exception as e:
            return None
    
    def recognize_face_by_embedding(self, face_embedding):
        """Recognize face using precomputed embeddings"""
        if face_embedding is None or not self.known_embeddings:
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
        
        # Adjust confidence threshold for small faces (more lenient)
        confidence_threshold = 1 - (self.confidence_threshold * 0.8)  # More lenient for small faces
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
        """Improved face detection using MediaPipe with multi-scale approach for small faces"""
        faces_data = []
        try:
            # Multi-scale detection for better small face coverage
            scales = [1.0, 0.8, 0.6]  # Try different scales
            h, w = frame.shape[:2]
            
            for scale in scales:
                if scale != 1.0:
                    process_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    scale_factor = 1 / scale
                else:
                    process_frame = frame
                    scale_factor = 1.0
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                results_mp = self.face_detection.process(rgb_frame)
                
                if results_mp.detections:
                    for detection in results_mp.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = process_frame.shape
                        
                        # Convert relative coordinates to absolute
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)
                        
                        # Scale back to original frame coordinates
                        x = int(x * scale_factor)
                        y = int(y * scale_factor)
                        w = int(w * scale_factor)
                        h = int(h * scale_factor)
                        
                        # Ensure coordinates are within frame bounds
                        orig_h, orig_w = frame.shape[:2]
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, orig_w - x)
                        h = min(h, orig_h - y)
                        
                        # Accept smaller faces (reduced minimum size)
                        if w > 8 and h > 8:  # Even smaller minimum size
                            # Add padding for small faces to get more context
                            padding = max(8, min(w, h) // 3)  # More aggressive padding
                            x_pad = max(0, x - padding)
                            y_pad = max(0, y - padding)
                            w_pad = min(orig_w - x_pad, w + 2 * padding)
                            h_pad = min(orig_h - y_pad, h + 2 * padding)
                            
                            # Extract face region with padding
                            face_region = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                            
                            if face_region.size > 0:
                                # Check if this face overlaps significantly with already detected faces
                                is_duplicate = False
                                for existing_face in faces_data:
                                    ex, ey, ew, eh = existing_face['bbox']
                                    # Calculate overlap
                                    overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                                    overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                                    overlap_area = overlap_x * overlap_y
                                    face_area = w * h
                                    if overlap_area > 0.5 * face_area:  # 50% overlap threshold
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    faces_data.append({
                                        'image': face_region,
                                        'bbox': (x, y, w, h)  # Original bbox without padding
                                    })
        except Exception as e:
            print(f"Face detection error: {e}")
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
        """Run face recognition loop with improved small face detection."""
        if not self.known_embeddings:
            print("❌ No known embeddings found. Please add face images to the database folder.")
            return
        
        # cap = cv2.VideoCapture("crow.mp4")
        cap = cv2.VideoCapture(0)
        
        print("✅ Starting face recognition with improved small face detection...")
        print("Press 'q' to quit, 'r' to reload embeddings")
        
        self.processing_thread = threading.Thread(target=self.process_frame_background)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                
                self.frame_counter += 1
                if self.frame_counter % (self.skip_frames + 1) == 0:
                    faces_data = self.detect_faces_optimized(frame)
                    if faces_data and self.recognition_queue.empty():
                        self.recognition_queue.put(faces_data)
                
                if not self.result_queue.empty():
                    self.last_results = self.result_queue.get()
                    self.send_attendance(self.last_results)
                
                if self.last_results:
                    frame = self.draw_results(frame, self.last_results)
                
                # Add debug info
                cv2.putText(frame, f"Faces detected: {len(self.last_results)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Optimized Face Recognition - Small Face Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("🔄 Reloading embeddings...")
                    self.stop_processing = True
                    if self.processing_thread.is_alive():
                        self.processing_thread.join()
                    self.create_embeddings()
                    self.stop_processing = False
                    self.processing_thread = threading.Thread(target=self.process_frame_background)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
                    print("✅ Embeddings reloaded")
        except KeyboardInterrupt:
            print("\n🛑 Stopping face recognition...")
        finally:
            self.stop_processing = True
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)
            cap.release()
            cv2.destroyAllWindows()

def main():
    if not os.path.exists("database"):
        os.makedirs("database", exist_ok=True)
        print("📁 Created database folder. Please add person folders with face images.")
    
    system = OptimizedFaceRecognitionSystem(
        database_path="database",
        confidence_threshold=0.4,
        embeddings_file="face_embeddings.pkl"
    )
    system.run()

if __name__ == "__main__":
    main()