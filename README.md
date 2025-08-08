# MySQL Setup Guide for Face Recognition System

## 1. Install MySQL Server

### For Windows:
1. Download MySQL Community Server from [https://dev.mysql.com/downloads/mysql/](https://dev.mysql.com/downloads/mysql/)
2. Run the installer and follow the setup wizard
3. Remember the root password you set during installation

### For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

### For macOS:
```bash
brew install mysql
brew services start mysql
```

## 2. Install Python MySQL Connector

```bash
pip install mysql-connector-python
```

## 3. Create Database and User (Optional)

Connect to MySQL as root:
```bash
mysql -u root -p
```

Then run these SQL commands:

```sql
-- Create database (optional - your code will create it automatically)
CREATE DATABASE IF NOT EXISTS face_recognition CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create a dedicated user for face recognition (recommended)
CREATE USER 'facerecog'@'localhost' IDENTIFIED BY 'your_password_here';

-- Grant privileges
GRANT ALL PRIVILEGES ON face_recognition.* TO 'facerecog'@'localhost';
FLUSH PRIVILEGES;

-- Exit MySQL
EXIT;
```

## 4. Configuration Options

### Option A: Use Root User (Simple)
Your code will work with default settings if MySQL root has no password:
```python
system = OptimizedFaceRecognitionSystem(
    database_path="database",
    confidence_threshold=0.4,
    embeddings_file="face_embeddings.pkl"
    # MySQL will use defaults: localhost:3306, user=root, password=""
)
```

### Option B: Use Environment Variables (Recommended)
Create a `.env` file in your project directory:
```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=facerecog
MYSQL_PASSWORD=your_password_here
MYSQL_DB=face_recognition
MYSQL_TABLE=face_detections
```

### Option C: Pass Parameters Directly
```python
system = OptimizedFaceRecognitionSystem(
    database_path="database",
    confidence_threshold=0.4,
    embeddings_file="face_embeddings.pkl",
    mysql_enabled=True,
    mysql_host="localhost",
    mysql_port=3306,
    mysql_user="facerecog",
    mysql_password="your_password_here",
    mysql_db="face_recognition",
    mysql_table="face_detections"
)
```

## 5. Test MySQL Connection

Create a simple test script:

```python
import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(
        host='localhost',
        port=3306,
        user='root',  # or your username
        password='',  # your password
    )
    
    if connection.is_connected():
        print("Successfully connected to MySQL")
        cursor = connection.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"MySQL version: {version[0]}")
        cursor.close()
    
except Error as e:
    print(f"Error connecting to MySQL: {e}")
    
finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("MySQL connection closed")
```

## 6. Database Schema

Your code will automatically create this table structure:

```sql
CREATE TABLE IF NOT EXISTS `face_detections` (
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
```

## 7. View Logged Data

To see the face detection logs:

```sql
USE face_recognition;
SELECT * FROM face_detections ORDER BY detected_at DESC LIMIT 10;
```

## 8. Troubleshooting

### Common Issues:

1. **"Access denied for user 'root'@'localhost'"**
   - Reset MySQL root password or create a new user

2. **"Unknown database 'face_recognition'"**
   - Don't worry, your code will create it automatically

3. **"Can't connect to MySQL server"**
   - Make sure MySQL service is running
   - Check if port 3306 is available

4. **"mysql-connector-python not installed"**
   - Run: `pip install mysql-connector-python`

### Check MySQL Service Status:

**Windows:**
```cmd
net start mysql
```

**Linux/macOS:**
```bash
sudo systemctl status mysql
# or
brew services list | grep mysql
```

## 9. Performance Tips

1. **Index Optimization**: The table includes indexes on `detected_at` and `name` for faster queries
2. **Batch Inserts**: Your code uses batch inserts for better performance
3. **Connection Pooling**: For production, consider connection pooling

## 10. Security Recommendations

1. Use a dedicated database user (not root)
2. Set strong passwords
3. Limit database privileges
4. Consider SSL connections for production
5. Store credentials in environment variables or config files (not in code)