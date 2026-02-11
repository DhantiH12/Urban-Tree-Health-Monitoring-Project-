import sqlite3
from datetime import datetime
import os

DB_PATH = "database/tree_health.db"

def init_db():
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tree_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        area_name TEXT,
        predicted_health TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_record(image_name, area_name, predicted_health, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO tree_predictions (
        image_name, area_name, predicted_health, confidence, timestamp
    ) VALUES (?, ?, ?, ?, ?)
    """, (
        image_name,
        area_name,
        predicted_health,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def fetch_all_records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tree_predictions ORDER BY timestamp DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows
