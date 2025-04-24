#!/usr/bin/env python3
import os
import csv
import json
import psycopg2
from psycopg2.extras import execute_values
import logging
from datetime import datetime
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database connection parameters
# These should be customized based on your PostgreSQL configuration
DB_PARAMS = {
    'dbname': 'restaurant_db',
    'user': 'postgres',
    'password': '2112', 
    'host': 'localhost',
    'port': '5433'
}

# Paths
CSV_FILE = 'data/restaurants_menu_data_cleaned.csv'
SCHEMA_FILE = 'src\db\schema.sql'

def create_db_schema():
    """Create database schema if it doesn't exist"""
    try:
        # Connect to PostgreSQL server - initially connect to postgres db
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_PARAMS['user'],
            password=DB_PARAMS['password'],
            host=DB_PARAMS['host'],
            port=DB_PARAMS['port']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_PARAMS['dbname']}'")
        exists = cursor.fetchone()
        
        if not exists:
            logging.info(f"Creating database {DB_PARAMS['dbname']}...")
            cursor.execute(f"CREATE DATABASE {DB_PARAMS['dbname']}")
            logging.info(f"Database {DB_PARAMS['dbname']} created successfully.")
        else:
            logging.info(f"Database {DB_PARAMS['dbname']} already exists.")
        
        # Close connection to postgres db
        cursor.close()
        conn.close()
        
        # Connect to the target database
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        # Read schema file and execute it
        with open(SCHEMA_FILE, 'r') as f:
            schema_sql = f.read()
            cursor.execute(schema_sql)
        
        conn.commit()
        logging.info("Schema created successfully.")
        
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        logging.error(f"Error creating schema: {e}")
        return False

def parse_csv_row(row):
    """Parse CSV row and convert data types appropriately"""
    try:
        # Parse and transform complex types
        restaurant_phone_numbers = json.loads(row['restaurant_phone_numbers']) if row['restaurant_phone_numbers'] else []
        restaurant_cuisines = json.loads(row['restaurant_cuisines']) if row['restaurant_cuisines'] else []
        restaurant_opening_hours = json.loads(row['restaurant_opening_hours']) if row['restaurant_opening_hours'] else []
        description_clean = row['description_clean']
        # Convert boolean
        item_is_veg = row['item_is_veg'].lower() == 'true'
        
        # Convert numeric
        item_price = float(row['item_price']) if row['item_price'] else None
        restaurant_dining_rating = float(row['restaurant_dining_rating']) if row['restaurant_dining_rating'] else None
        restaurant_delivery_rating = float(row['restaurant_delivery_rating']) if row['restaurant_delivery_rating'] else None
        restaurant_dining_ratings_count = int(row['restaurant_dining_ratings_count']) if row['restaurant_dining_ratings_count'] else 0
        restaurant_delivery_ratings_count = int(row['restaurant_delivery_ratings_count']) if row['restaurant_delivery_ratings_count'] else 0
        city= row['city'] or 'New Delhi'
        # Format dates
        created_at = row['created_at'] if row['created_at'] else datetime.now().isoformat()
        updated_at = row['updated_at'] if row['updated_at'] else datetime.now().isoformat()
        
    
        
        return {
            'item_id': int(row['item_id']),
            'item_name': row['item_name'],
            'item_description': row['item_description'],
            'item_price': item_price,
            'item_category': row['item_category'],
            'item_is_veg': item_is_veg,
            'restaurant_name': row['restaurant_name'],
            'restaurant_address': row['restaurant_address'],
            'restaurant_phone_numbers': restaurant_phone_numbers,
            'restaurant_cuisines': restaurant_cuisines,
            'restaurant_opening_hours': json.dumps(restaurant_opening_hours),  # Convert to JSONB
            'restaurant_dining_rating': restaurant_dining_rating,
            'restaurant_delivery_rating': restaurant_delivery_rating,
            'restaurant_dining_ratings_count': restaurant_dining_ratings_count,
            'restaurant_delivery_ratings_count': restaurant_delivery_ratings_count,
            'restaurant_source_url': row['restaurant_source_url'],
            'city': city,
            'created_at': created_at,
            'updated_at': updated_at,
            'description_clean': description_clean
        }
    except Exception as e:
        logging.error(f"Error parsing row: {e}")
        logging.error(f"Problematic row: {row}")
        return None

def import_csv_to_db():
    """Import CSV data into PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        # Truncate table if needed - uncomment if you want to clear the table before importing
        # cursor.execute("TRUNCATE TABLE restaurant_menu_flat RESTART IDENTITY CASCADE")
        
        # Read CSV file
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            
            batch_size = 100
            records = []
            total_rows = 0
            
            # Prepare column list
            columns = [
                'item_id', 'item_name', 'item_description', 'item_price', 'item_category', 
                'item_is_veg', 'restaurant_name', 'restaurant_address', 'restaurant_phone_numbers', 
                'restaurant_cuisines', 'restaurant_opening_hours', 'restaurant_dining_rating', 
                'restaurant_delivery_rating', 'restaurant_dining_ratings_count', 
                'restaurant_delivery_ratings_count', 'restaurant_source_url', 'city', 'created_at', 'updated_at', 
                'description_clean'
            ]
            
            for row in csv_reader:
                parsed_row = parse_csv_row(row)
                if parsed_row:
                    # Extract values in the same order as columns
                    values = [parsed_row[column] for column in columns]
                    records.append(values)
                    
                    # Process in batches
                    if len(records) >= batch_size:
                        execute_values(
                            cursor,
                            f"INSERT INTO restaurant_menu_flat ({', '.join(columns)}) VALUES %s ON CONFLICT (item_id) DO NOTHING",
                            records
                        )
                        total_rows += len(records)
                        logging.info(f"Imported {total_rows} rows so far...")
                        records = []
            
            # Insert any remaining records
            if records:
                execute_values(
                    cursor,
                    f"INSERT INTO restaurant_menu_flat ({', '.join(columns)}) VALUES %s ON CONFLICT (item_id) DO NOTHING",
                    records
                )
                total_rows += len(records)
            
            conn.commit()
            logging.info(f"Successfully imported {total_rows} rows into restaurant_menu_flat table.")
        
        cursor.close()
        conn.close()
        
        return True
    
    except Exception as e:
        logging.error(f"Error importing data: {e}")
        return False

def analyze_table():
    """Run ANALYZE to update statistics for the query planner"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        logging.info("Running ANALYZE on restaurant_menu_flat table...")
        cursor.execute("ANALYZE restaurant_menu_flat")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info("ANALYZE completed successfully.")
        return True
    
    except Exception as e:
        logging.error(f"Error analyzing table: {e}")
        return False

def main():
    """Main function to run the import process"""
    logging.info("Starting database import process...")
    
    # Step 1: Create schema if needed
    if not create_db_schema():
        logging.error("Failed to create schema. Exiting.")
        return
    
    # Step 2: Import CSV data
    if not import_csv_to_db():
        logging.error("Failed to import CSV data. Exiting.")
        return
    
    # Step 3: Update statistics for query optimizer
    if not analyze_table():
        logging.warning("Failed to analyze table. Query performance may be affected.")
    
    logging.info("Database import process completed successfully.")

if __name__ == "__main__":
    main() 