import cv2
import os
from deepface import DeepFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import threading
from queue import Queue
import datetime as dt

try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except Exception:
    MYSQL_AVAILABLE = False
    mysql_connector = None
    MySQLError = Exception

class MySQLLogger:
    def __init__(self, host, port, user, password, database, table):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.conn = None
        self.cur = None
        self._connect_and_prepare()

    def _connect_and_prepare(self):
        try:
            self.conn = mysql_connector.connect(
                host=self.host, port=self.port, user=self.user, password=self.password
            )
            self.cur = self.conn.cursor()
            self.cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{self.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            self.cur.execute(f"USE `{self.database}`")
            self.cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{self.table}` (
                    `id` INT NOT NULL AUTO_INCREMENT,
                    `detected_at` DATETIME NOT NULL,
                    `name` VARCHAR(255) NOT NULL,
                    `confidence` FLOAT NOT NULL,
                    `x` INT NOT NULL,
                    `y` INT NOT NULL,
                    `w` INT NOT NULL,
                    `h` INT NOT NULL,
                    PRIMARY KEY (`id`),
                    INDEX `idx_detected_at` (`detected_at`),
                    INDEX `idx_name` (`name`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            self.conn.commit()
        except MySQLError as e:
            self.close()
            raise e

    def log(self, detections, detected_at):
        if not detections or self.conn is None:
            return
        rows = []
        for det in detections:
            x, y, w, h = det.get('bbox', (0, 0, 0, 0))
            name = det.get('name', 'Unknown') or 'Unknown'
            confidence = float(det.get('confidence', 0) or 0)
            rows.append((detected_at, name, confidence, int(x), int(y), int(w), int(h)))
        try:
            insert_sql = (
                f"INSERT INTO `{self.table}` (detected_at, name, confidence, x, y, w, h) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s)"
            )
            self.cur.executemany(insert_sql, rows)
            self.conn.commit()
        except MySQLError:
            try:
                self.conn.rollback()
            except MySQLError:
                pass

    def close(self):
        try:
            if self.cur is not None:
                self.cur.close()
            if self.conn is not None and self.conn.is_connected():
                self.conn.close()
        except Exception:
            pass

class OptimizedFaceRecognitionSystem:
    def __init__(self, database_path="database", confidence_threshold=0.4, 
                 embeddings_file="face_embeddings.pkl",
                 mysql_enabled=True,
                 mysql_host=None,
                 mysql_port=None,
                 mysql_user=None,
                 mysql_password=None,
                 mysql_db=None,
                 mysql_table=None):
        
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.embeddings_file = embeddings_file
        self.known_embeddings = {}
        
        # Initialize face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("❌ Error: Could not load face cascade classifier")
            raise Exception("Face cascade not found")
        # Performance optimization settings
        self.process_scale = 0.5
        self.skip_frames = 2
        self.frame_counter = 0
        self.last_results = []
        
        # Threading for background processing
        self.recognition_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing_thread = None
        self.stop_processing = False
        
        # MySQL configuration
        self.mysql_enabled = bool(mysql_enabled) and MYSQL_AVAILABLE
        self.mysql_host = mysql_host or os.getenv('MYSQL_HOST', 'localhost')
        self.mysql_port = int(mysql_port or os.getenv('MYSQL_PORT', '3306'))
        self.mysql_user = mysql_user or os.getenv('MYSQL_USER', 'facerecog')
        self.mysql_password = mysql_password or os.getenv('MYSQL_PASSWORD', 'facerecog')
        self.mysql_db = mysql_db or os.getenv('MYSQL_DB', 'face_recognition')
        self.mysql_table = mysql_table or os.getenv('MYSQL_TABLE', 'face_detections')

        # Setup MySQL logger
        self.mysql_logger = None
        if mysql_enabled and not MYSQL_AVAILABLE:
            self.mysql_enabled = False
        elif self.mysql_enabled:
            try:
                self.mysql_logger = MySQLLogger(self.mysql_host, self.mysql_port, self.mysql_user, self.mysql_password, self.mysql_db, self.mysql_table)
            except MySQLError:
                self.mysql_enabled = False
        
        # Load or create embeddings
        self.load_or_create_embeddings()
    



    def log_detections(self, detections):
        if not self.mysql_enabled or not detections or self.mysql_logger is None:
            return
        try:
            self.mysql_logger.log(detections, dt.datetime.now())
        except MySQLError as e:
            print(f"❌ Failed to log detections: {e}")
        
    def load_or_create_embeddings(self):
        """Load precomputed embeddings or create them if they don't exist"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                return
            except Exception:
                pass
        self.create_embeddings()
    
    def create_embeddings(self):
        """Create embeddings for all images in the database"""
        if not os.path.exists(self.database_path):
            return
        
        self.known_embeddings = {}
        for person_name in os.listdir(self.database_path):
            person_folder = os.path.join(self.database_path, person_name)
            if not os.path.isdir(person_folder):
                continue
            person_embeddings = []
            for image_file in os.listdir(person_folder):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                image_path = os.path.join(person_folder, image_file)
                try:
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name='Facenet512',
                        enforce_detection=False
                    )
                    if isinstance(embedding, list) and len(embedding) > 0:
                        person_embeddings.append(embedding[0]['embedding'])
                except Exception:
                    continue
            if person_embeddings:
                self.known_embeddings[person_name] = person_embeddings
        
        if self.known_embeddings:
            try:
                with open(self.embeddings_file, 'wb') as f:
                    pickle.dump(self.known_embeddings, f)
            except Exception:
                pass
    
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
        """Run face recognition loop (concise)."""
        if not self.known_embeddings:
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        
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
                    self.log_detections(self.last_results)
                
                if self.last_results:
                    frame = self.draw_results(frame, self.last_results)
                
                cv2.imshow('Optimized Face Recognition', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.stop_processing = True
                    if self.processing_thread.is_alive():
                        self.processing_thread.join()
                    self.create_embeddings()
                    self.stop_processing = False
                    self.processing_thread = threading.Thread(target=self.process_frame_background)
                    self.processing_thread.daemon = True
                    self.processing_thread.start()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_processing = True
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)
            cap.release()
            cv2.destroyAllWindows()
            try:
                if self.mysql_logger is not None:
                    self.mysql_logger.close()
            except Exception:
                pass

def main():
    if not os.path.exists("database"):
        os.makedirs("database", exist_ok=True)
    system = OptimizedFaceRecognitionSystem(
        database_path="database",
        confidence_threshold=0.4,
        embeddings_file="face_embeddings.pkl"
    )
    system.run()

if __name__ == "__main__":
    main()