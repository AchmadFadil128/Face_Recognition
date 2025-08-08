#!/usr/bin/env python3
"""
Quick MySQL Connection Test
Focused test to identify the exact connection issue
"""

print("🔍 QUICK MYSQL CONNECTION TEST")
print("=" * 40)

# Step 1: Test import
print("\n1️⃣ Testing mysql-connector-python import...")
try:
    import mysql.connector
    from mysql.connector import Error
    print("   ✅ SUCCESS - mysql.connector imported")
except ImportError as e:
    print(f"   ❌ FAILED - {e}")
    print("   Fix: pip install mysql-connector-python")
    exit(1)

# Step 2: Test basic connection (without database)
print("\n2️⃣ Testing basic MySQL connection...")
try:
    conn = mysql.connector.connect(
        host='localhost',
        port=3306,
        user='facerecog',
        password='facerecog'
    )
    if conn.is_connected():
        print("   ✅ SUCCESS - Connected to MySQL server")
        conn.close()
    else:
        print("   ❌ FAILED - Could not connect")
        exit(1)
except Error as e:
    print(f"   ❌ FAILED - {e}")
    print("\n   🔧 Try these fixes:")
    print("   1. Check if user exists:")
    print("      sudo mysql")
    print("      SELECT User, Host FROM mysql.user WHERE User='facerecog';")
    print("   2. If no results, recreate user:")
    print("      DROP USER IF EXISTS 'facerecog'@'localhost';")
    print("      CREATE USER 'facerecog'@'localhost' IDENTIFIED BY 'facerecog';")
    print("      GRANT ALL PRIVILEGES ON *.* TO 'facerecog'@'localhost';")
    print("      FLUSH PRIVILEGES;")
    exit(1)

# Step 3: Test database connection
print("\n3️⃣ Testing connection to face_recognition database...")
try:
    conn = mysql.connector.connect(
        host='localhost',
        port=3306,
        user='facerecog',
        password='facerecog',
        database='face_recognition'
    )
    if conn.is_connected():
        print("   ✅ SUCCESS - Connected to face_recognition database")
        
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        print(f"   📊 Current database: {db_name}")
        
        cursor.close()
        conn.close()
    else:
        print("   ❌ FAILED - Could not connect to database")
except Error as e:
    print(f"   ❌ FAILED - {e}")
    if "Unknown database" in str(e):
        print("   🔧 Database doesn't exist, creating it...")
        try:
            conn = mysql.connector.connect(
                host='localhost',
                port=3306,
                user='facerecog',
                password='facerecog'
            )
            cursor = conn.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS face_recognition CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            conn.commit()
            cursor.close()
            conn.close()
            print("   ✅ Database created successfully")
        except Error as create_err:
            print(f"   ❌ Could not create database: {create_err}")
            exit(1)
    else:
        exit(1)

# Step 4: Test table operations
print("\n4️⃣ Testing table creation and operations...")
try:
    conn = mysql.connector.connect(
        host='localhost',
        port=3306,
        user='facerecog',
        password='facerecog',
        database='face_recognition'
    )
    
    cursor = conn.cursor()
    
    # Create table
    create_sql = """
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
    """
    
    cursor.execute(create_sql)
    conn.commit()
    print("   ✅ SUCCESS - Table created/verified")
    
    # Test insert
    import datetime
    now = datetime.datetime.now()
    insert_sql = "INSERT INTO face_detections (detected_at, name, confidence, x, y, w, h) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    cursor.execute(insert_sql, (now, "TestUser", 99.9, 10, 20, 100, 150))
    conn.commit()
    print("   ✅ SUCCESS - Insert test data")
    
    # Test select
    cursor.execute("SELECT COUNT(*) FROM face_detections")
    count = cursor.fetchone()[0]
    print(f"   ✅ SUCCESS - Query data (found {count} records)")
    
    # Clean up
    cursor.execute("DELETE FROM face_detections WHERE name = 'TestUser'")
    conn.commit()
    print("   ✅ SUCCESS - Cleanup test data")
    
    cursor.close()
    conn.close()
    
except Error as e:
    print(f"   ❌ FAILED - {e}")
    exit(1)

# Step 5: Create configuration
print("\n5️⃣ Creating configuration file...")
config = """# Face Recognition MySQL Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=facerecog
MYSQL_PASSWORD=facerecog
MYSQL_DB=face_recognition
MYSQL_TABLE=face_detections
"""

try:
    with open('.env', 'w') as f:
        f.write(config)
    print("   ✅ SUCCESS - Created .env file")
except Exception as e:
    print(f"   ❌ FAILED - Could not create .env file: {e}")

# Final result
print("\n" + "=" * 40)
print("🎉 ALL TESTS PASSED!")
print("=" * 40)
print("Your MySQL setup is working correctly!")
print("\nNext steps:")
print("1. Create face database folder:")
print("   mkdir -p database/YourName")
print("   # Add photos to database/YourName/")
print("2. Run the face recognition system")

print("\nCredentials confirmed:")
print("   Host: localhost:3306")
print("   User: facerecog")
print("   Pass: facerecog") 
print("   DB:   face_recognition")
print("   Table: face_detections")