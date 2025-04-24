import json
import csv
import datetime

# Load JSON data
with open('data/restaurants_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define CSV header based on the PostgreSQL schema
csv_header = [
    'item_id',
    'item_name',
    'item_description',
    'item_price',
    'item_category',
    'item_is_veg',
    'restaurant_name',
    'restaurant_address',
    'restaurant_phone_numbers',
    'restaurant_cuisines',
    'restaurant_opening_hours',
    'restaurant_dining_rating',
    'restaurant_delivery_rating',
    'restaurant_dining_ratings_count',
    'restaurant_delivery_ratings_count',
    'restaurant_source_url',
    'created_at',
    'updated_at'
]

# Open CSV file for writing
with open('data/restaurant_menu_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)
    
    # Generate rows
    item_id = 1
    now = datetime.datetime.now().isoformat()
    
    for restaurant in data:
        # Get restaurant details
        restaurant_name = restaurant.get('restraunt_name', '')  # Note the typo in JSON
        restaurant_address = restaurant.get('address', '')
        restaurant_phone_numbers = json.dumps(restaurant.get('phone', []))
        restaurant_cuisines = json.dumps(restaurant.get('what_they_serve', []))
        restaurant_opening_hours = json.dumps(restaurant.get('opening_hours', []))
        restaurant_dining_rating = restaurant.get('dining_ratings', None)
        restaurant_delivery_rating = restaurant.get('delivery_ratings', None)
        restaurant_dining_ratings_count = restaurant.get('dining_ratings_count', 0)
        restaurant_delivery_ratings_count = restaurant.get('delivery_ratings_count', 0)
        restaurant_source_url = restaurant.get('source_url', '')
        
        # Process each menu item
        for item in restaurant.get('menu', []):
            item_name = item.get('item_name', '')
            item_description = item.get('description', '')
            item_price = item.get('price', 0)
            item_category = item.get('category', '')
            item_is_veg = 'true' if item.get('veg', False) else 'false'
            
            # Write row to CSV
            writer.writerow([
                item_id,
                item_name,
                item_description,
                item_price,
                item_category,
                item_is_veg,
                restaurant_name,
                restaurant_address,
                restaurant_phone_numbers,
                restaurant_cuisines,
                restaurant_opening_hours,
                restaurant_dining_rating,
                restaurant_delivery_rating,
                restaurant_dining_ratings_count,
                restaurant_delivery_ratings_count,
                restaurant_source_url,
                now,
                now
            ])
            
            item_id += 1

print(f"Conversion complete. CSV file created: data/restaurant_menu_data.csv") 