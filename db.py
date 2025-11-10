import mysql.connector
from mysql.connector import errorcode
from datetime import date, datetime

# Database configuration (as requested)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'maruti2004#',
    'database': 'traffic_db',
}


def init_db():
    """Create the database and results table if they don't exist."""
    try:
        # Connect without specifying database to ensure we can create it
        cnx = mysql.connector.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'])
        cursor = cnx.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS {}".format(DB_CONFIG['database']))
        cnx.database = DB_CONFIG['database']

        create_table = (
            "CREATE TABLE IF NOT EXISTS results ("
            "  id INT AUTO_INCREMENT PRIMARY KEY," 
            "  Date_of_Uploading DATE,")
        # Build the remainder separately for readability
        create_table = (
            "CREATE TABLE IF NOT EXISTS results ("
            "id INT AUTO_INCREMENT PRIMARY KEY, "
            "Date_of_Uploading DATE, "
            "Time_of_Uploading TIME, "
            "Region_Name VARCHAR(255), "
            "Intersection_ID VARCHAR(100), "
            "Max_Vehicle_Count INT, "
            "Max_Green_Time INT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ") ENGINE=InnoDB"
        )

        cursor.execute(create_table)
        cursor.close()
        cnx.close()
        print("[db] Initialized database and ensured results table exists.")
    except mysql.connector.Error as err:
        print(f"[db] Error initializing DB: {err}")


def get_connection():
    """Return a new connection to the traffic_db database."""
    return mysql.connector.connect(**DB_CONFIG)


def insert_result(region_name: str, intersection_id: str, date_obj: date, time_obj, max_vehicle_count: int, max_green_time: int):
    """Insert a single result row into the results table.

    date_obj should be a datetime.date, time_obj a datetime.time or string.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = (
            "INSERT INTO results (Date_of_Uploading, Time_of_Uploading, Region_Name, Intersection_ID, Max_Vehicle_Count, Max_Green_Time) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        # Convert date/time to strings to avoid driver conversion issues and ensure predictable values
        d_val = date_obj.isoformat() if hasattr(date_obj, 'isoformat') else date_obj
        t_val = time_obj.isoformat() if hasattr(time_obj, 'isoformat') else time_obj
        # Ensure no None values are sent for strings
        region_val = region_name if region_name is not None else ''
        intersection_val = intersection_id if intersection_id is not None else ''

        vals = (d_val, t_val, region_val, intersection_val, int(max_vehicle_count) if max_vehicle_count is not None else 0, int(max_green_time) if max_green_time is not None else 0)

        # Debug/log the values being inserted for diagnosis
        print(f"[db] Executing SQL: {sql}")
        print(f"[db] With values: {vals}")

        cursor.execute(sql, vals)
        conn.commit()
        last_id = cursor.lastrowid
        cursor.close()
        conn.close()
        print(f"[db] Inserted result id={last_id}")
        return last_id
    except mysql.connector.Error as err:
        print(f"[db] Error inserting result: {err}")
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        return None


def fetch_latest_result(region_name: str, intersection_id: str):
    """Return the most recent result row for the given region and intersection.

    Returns a dict or None.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        sql = (
            "SELECT * FROM results WHERE Region_Name = %s AND Intersection_ID = %s "
            "ORDER BY id DESC LIMIT 1"
        )
        cursor.execute(sql, (region_name, intersection_id))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row
    except mysql.connector.Error as err:
        print(f"[db] Error fetching latest result: {err}")
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        return None
