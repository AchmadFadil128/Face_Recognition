import cv2
import os
import numpy as np
from deepface import DeepFace
import time
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

def test_mysql_connection(host, port, user, password, database=None):
    """Test MySQL connection with given parameters"""
    try:
        if database:
            connection = mysql_connector.connect(
                host=host, port=port, user=user, password=password, database=database
            )
        else:
            connection = mysql_connector.connect(
                host=host, port=port, user=user, password=password
            )
        
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            cursor.close()
            connection.close()
            return True, f"MySQL {version[0]}"
        return False, "Could not connect"
    except MySQLError as e:
        return False, str(e)

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
        
        print("="*60)
        print("OPTIMIZED FACE RECOGNITION SYSTEM")
        print("="*60)
        
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
        else:
            print("✅ Face cascade classifier loaded")
        
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
        self._db_conn = None
        self._db_cursor = None

        # Check MySQL setup
        if mysql_enabled and not MYSQL_AVAILABLE:
            print("❌ MySQL logging disabled: mysql-connector-python not installed")
            print("   Install with: pip install mysql-connector-python")
            self.mysql_enabled = False
        elif self.mysql_enabled:
            self._setup_mysql()
        else:
            print("ℹ️  MySQL logging disabled by configuration")
        
        # Load or create embeddings
        self.load_or_create_embeddings()
        print("="*60)
    
    def _setup_mysql(self):
        """Setup MySQL connection with comprehensive testing"""
        print("\n🔍 Setting up MySQL connection...")
        print(f"   Host: {self.mysql_host}:{self.mysql_port}")
        print(f"   User: {self.mysql_user}")
        print(f"   Database: {self.mysql_db}")
        print(f"   Table: {self.mysql_table}")
        
        # Test basic connection
        success, message = test_mysql_connection(
            self.mysql_host, self.mysql_port, self.mysql_user, self.mysql_password
        )
        
        if not success:
            print(f"❌ MySQL connection failed: {message}")
            print("   Please check your MySQL server and credentials")
            print("   MySQL logging will be disabled")
            self.mysql_enabled = False
            return
        
        print(f"✅ MySQL server connected: {message}")
        
        # Initialize database and table
        try:
            self._init_db()
            if self.mysql_enabled:
                print(f"✅ MySQL logging enabled → {self.mysql_host}:{self.mysql_port}/{self.mysql_db}.{self.mysql_table}")
        except Exception as e:
            print(f"❌ MySQL setup failed: {e}")
            self.mysql_enabled = False

    def _init_db(self):
        """Initialize MySQL connection and ensure target DB/table exist"""
        try:
            # Try connect directly to target DB
            try:
                self._db_conn = mysql_connector.connect(
                    host=self.mysql_host,
                    port=self.mysql_port,
                    user=self.mysql_user,
                    password=self.mysql_password,
                    database=self.mysql_db,
                    autocommit=False
                )
            except MySQLError as conn_err:
                msg = str(conn_err).lower()
                if 'unknown database' in msg or 'does not exist' in msg:
                    print(f"   📝 Creating database '{self.mysql_db}'...")
                    # Connect without database and create
                    tmp_conn = mysql_connector.connect(
                        host=self.mysql_host,
                        port=self.mysql_port,
                        user=self.mysql_user,
                        password=self.mysql_password,
                        autocommit=True
                    )
                    tmp_cursor = tmp_conn.cursor()
                    tmp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.mysql_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                    tmp_cursor.close()
                    tmp_conn.close()
                    print(f"   ✅ Database '{self.mysql_db}' created")
                    
                    # Reconnect to created DB
                    self._db_conn = mysql_connector.connect(
                        host=self.mysql_host,
                        port=self.mysql_port,
                        user=self.mysql_user,
                        password=self.mysql_password,
                        database=self.mysql_db,
                        autocommit=False
                    )
                else:
                    raise
                    
            self._db_cursor = self._db_conn.cursor()
            self._create_table_if_not_exists()
            
        except MySQLError as e:
            print(f"❌ MySQL initialization failed: {e}")
            self.mysql_enabled = False
            self._db_conn = None
            self._db_cursor = None

    def _ensure_connection(self):
        """Ensure DB connection is alive; reconnect if needed"""
        if not self.mysql_enabled:
            return False
        try:
            if self._db_conn is None or not self._db_conn.is_connected():
                self._init_db()
            return self._db_conn is not None and self._db_conn.is_connected()
        except MySQLError:
            return False

    def _create_table_if_not_exists(self):
        """Create detections table if missing"""
        print(f"   📝 Creating table '{self.mysql_table}' if not exists...")
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS `{self.mysql_table}` (
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
        self._db_cursor.execute(create_sql)
        self._db_conn.commit()
        print(f"   ✅ Table '{self.mysql_table}' ready")

    def log_detections(self, detections):
        """Insert a batch of detection rows into MySQL"""
        if not self.mysql_enabled or not detections:
            return
        if not self._ensure_connection():
            return
            
        now = dt.datetime.now()
        rows = []
        for det in detections:
            x, y, w, h = det.get('bbox', (0, 0, 0, 0))
            name = det.get('name', 'Unknown') or 'Unknown'
            confidence = float(det.get('confidence', 0) or 0)
            rows.append((now, name, confidence, int(x), int(y), int(w), int(h)))
            
        try:
            insert_sql = (
                f"INSERT INTO `{self.mysql_table}` (detected_at, name, confidence, x, y, w, h) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s)"
            )
            self._db_cursor.executemany(insert_sql, rows)
            self._db_conn.commit()
            # print(f"📊 Logged {len(rows)} detections to database")
        except MySQLError as e:
            try:
                self._db_conn.rollback()
            except MySQLError:
                pass
            print(f"❌ Failed to log detections: {e}")
        
    def load_or_create_embeddings(self):
        """Load precomputed embeddings or create them if they don't exist"""
        print(f"\n🧠 Loading face embeddings...")
        
        if os.path.exists(self.embeddings_file):
            print(f"   📂 Loading from {self.embeddings_file}...")
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                print(f"   ✅ Loaded embeddings for {len(self.known_embeddings)} people")
                
                # Display loaded people
                for person, embeddings in self.known_embeddings.items():
                    print(f"      - {person}: {len(embeddings)} embeddings")
                return
            except Exception as e:
                print(f"   ❌ Error loading embeddings: {e}")
        
        print("   📝 Creating new embeddings from database...")
        self.create_embeddings()
    
    def create_embeddings(self):
        """Create embeddings for all images in the database"""
        if not os.path.exists(self.database_path):
            print(f"❌ Database path '{self.database_path}' not found!")
            print(f"   Please create the directory structure:")
            print(f"   {self.database_path}/")
            print(f"   ├── PersonName1/")
            print(f"   │   ├── photo1.jpg")
            print(f"   │   └── photo2.jpg")
            print(f"   └── PersonName2/")
            print(f"       └── photo1.jpg")
            return
        
        self.known_embeddings = {}
        total_images = 0
        total_people = 0
        
        for person_name in os.listdir(self.database_path):
            person_folder = os.path.join(self.database_path, person_name)
            
            if os.path.isdir(person_folder):
                print(f"   👤 Processing {person_name}...")
                person_embeddings = []
                
                image_files = [f for f in os.listdir(person_folder) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if not image_files:
                    print(f"      ⚠️  No image files found in {person_folder}")
                    continue
                
                for image_file in image_files:
                    image_path = os.path.join(person_folder, image_file)
                    
                    try:
                        # Generate embedding using DeepFace
                        embedding = DeepFace.represent(
                            img_path=image_path,
                            model_name='Facenet512',
                            enforce_detection=False
                        )
                        
                        if isinstance(embedding, list) and len(embedding) > 0:
                            person_embeddings.append(embedding[0]['embedding'])
                            total_images += 1
                            print(f"      ✅ {image_file}")
                        else:
                            print(f"      ❌ No face detected in {image_file}")
                        
                    except Exception as e:
                        print(f"      ❌ Error processing {image_file}: {e}")
                        continue
                
                if person_embeddings:
                    self.known_embeddings[person_name] = person_embeddings
                    total_people += 1
                    print(f"      → Created {len(person_embeddings)} embeddings for {person_name}")
                else:
                    print(f"      ⚠️  No valid embeddings created for {person_name}")
        
        # Save embeddings to file
        if self.known_embeddings:
            try:
                with open(self.embeddings_file, 'wb') as f:
                    pickle.dump(self.known_embeddings, f)
                print(f"   💾 Saved embeddings to {self.embeddings_file}")
            except Exception as e:
                print(f"   ❌ Error saving embeddings: {e}")
        
        print(f"   📊 Summary: {total_images} embeddings created for {total_people} people")
        
        if total_people == 0:
            print(f"   ⚠️  No people found! Please add photos to {self.database_path}")
    
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
        """Main function to run the optimized face recognition system"""
        if not self.known_embeddings:
            print("❌ No face embeddings available. Please add photos to the database folder.")
            return
            
        print(f"\n🚀 Starting face recognition system...")
        print(f"   📊 {len(self.known_embeddings)} people in database")
        print(f"   🎯 Confidence threshold: {self.confidence_threshold}")
        print(f"   📝 MySQL logging: {'✅ Enabled' if self.mysql_enabled else '❌ Disabled'}")
        print("\nControls:")
        print("   'q' - Quit")
        print("   'r' - Reload embeddings")
        print("-" * 60)
        
        # Initialize camera
        cap = cv2.VideoCapture("test.mp4")
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
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
                    print("❌ Error: Could not read frame")
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
                    # Log detections to MySQL (if enabled)
                    self.log_detections(self.last_results)
                
                # Draw results on frame
                if self.last_results:
                    frame = self.draw_results(frame, self.last_results)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter >= 30:
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                    print(f"🎯 FPS: {fps:.1f}")
                
                # Display frame
                cv2.imshow('Optimized Face Recognition', frame)
                
                # Handle key presses
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
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping face recognition system...")
        
        finally:
            # Clean up
            self.stop_processing = True
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)
            cap.release()
            cv2.destroyAllWindows()
            # Close DB connection
            try:
                if self._db_cursor is not None:
                    self._db_cursor.close()
                if self._db_conn is not None and self._db_conn.is_connected():
                    self._db_conn.close()
                    print("🔌 MySQL connection closed")
            except Exception:
                pass
            print("✅ Face recognition system stopped")

def main():
    print("Starting Optimized Face Recognition System with MySQL Integration")
    
    # Check if database folder exists
    if not os.path.exists("database"):
        print("📁 Creating database folder...")
        os.makedirs("database", exist_ok=True)
        print("   Please add person folders with photos to the 'database' directory")
        print("   Example structure:")
        print("   database/")
        print("   ├── Alice/")
        print("   │   ├── alice1.jpg")
        print("   │   └── alice2.jpg")
        print("   └── Bob/")
        print("       └── bob1.jpg")
        
        # Ask user if they want to continue
        response = input("\nDo you want to continue without face data? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting. Please add photos then rerun.")
            return
    
    # Initialize system (uses environment variables for MySQL if set)
    system = OptimizedFaceRecognitionSystem(
        database_path="database",
        confidence_threshold=0.4,
        embeddings_file="face_embeddings.pkl"
    )
    
    # Run the system (no auto-execution outside of script run)
    system.run()

if __name__ == "__main__":
    main()