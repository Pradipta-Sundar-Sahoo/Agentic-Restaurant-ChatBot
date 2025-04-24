import sys
import json
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def parse_restaurant_html(html_content):
    """
    Parses the Zomato restaurant HTML content to extract relevant information.

    Args:
        html_content: A string containing the HTML source code.

    Returns:
        A dictionary containing the extracted restaurant information,
        or None if essential data cannot be found.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {
        "restraunt_name": None,
        "address": None,
        "phone": [],
        "what_they_serve": None,
        "opening_hours": None,
        "dining_ratings": None,
        "delivery_ratings": None,
        "dining_ratings_count": None,
        "delivery_ratings_count": None,
        "menu": []
    }

    # --- Extract data using JSON-LD first (often more reliable) ---
    json_ld_scripts = soup.find_all('script', type='application/ld+json')
    restaurant_json_ld = None
    menu_json_ld = None # Keep reference, although specific item details are often missing here

    for script in json_ld_scripts:
        # Ensure script has content before trying to parse
        if script.string:
            try:
                ld_data = json.loads(script.string)
                # Find the script block specifically for the Restaurant schema
                if isinstance(ld_data, dict) and ld_data.get('@type') == 'Restaurant':
                    restaurant_json_ld = ld_data
                # Find the script block specifically for the Menu schema
                elif isinstance(ld_data, dict) and ld_data.get('@type') == 'Menu':
                    menu_json_ld = ld_data # Keep it for potential cross-checking if needed
            except json.JSONDecodeError:
                # print(f"Warning: Could not decode JSON-LD: {script.string[:100]}...") # Optional: for debugging
                continue # Ignore invalid JSON

    # Extract from Restaurant JSON-LD if found
    if restaurant_json_ld:
        data['restraunt_name'] = restaurant_json_ld.get('name')

        address_data = restaurant_json_ld.get('address', {})
        if isinstance(address_data, dict): # Ensure address_data is a dict
            street = address_data.get('streetAddress', '')
            locality = address_data.get('addressLocality', '')
            region = address_data.get('addressRegion', '')
            # Combine only non-empty parts
            address_parts = [part for part in [street, locality, region] if part]
            data['address'] = ', '.join(address_parts) if address_parts else None
        else:
             data['address'] = None # Handle case where address is not a dict

        cuisine = restaurant_json_ld.get('servesCuisine')
        data['what_they_serve'] = cuisine if cuisine else None

        opening_hours = restaurant_json_ld.get('openingHours')
        data['opening_hours'] = opening_hours if opening_hours else None

        telephone = restaurant_json_ld.get('telephone')
        if telephone and isinstance(telephone, str):
            # Split potentially comma-separated numbers
            data['phone'] = [p.strip() for p in telephone.split(',') if p.strip()]
        elif telephone and isinstance(telephone, list): # Handle if it's already a list
             data['phone'] = [str(p).strip() for p in telephone if str(p).strip()]


        rating_data = restaurant_json_ld.get('aggregateRating', {})
        if isinstance(rating_data, dict): # Ensure rating_data is a dict
            rating_value = rating_data.get('ratingValue')
            rating_count = rating_data.get('ratingCount')
            try:
                data['dining_ratings'] = float(rating_value) if rating_value else None
            except (ValueError, TypeError):
                data['dining_ratings'] = None
            try:
                data['dining_ratings_count'] = int(rating_count) if rating_count else None
            except (ValueError, TypeError):
                 data['dining_ratings_count'] = None
        else:
             data['dining_ratings'] = None
             data['dining_ratings_count'] = None

    # --- Fallback or Supplement with HTML Parsing ---

    # Restaurant Name (Fallback if JSON-LD failed)
    if not data['restraunt_name']:
        # Using a more specific selector based on the provided HTML structure
        name_tag = soup.select_one('div.sc-jeCdPy h1.sc-7kepeu-0')
        if name_tag:
            data['restraunt_name'] = name_tag.get_text(strip=True)

    # Address (Fallback if JSON-LD failed)
    if not data['address']:
        # Try the primary address class first
        address_tag_div = soup.find('div', class_=re.compile(r'sc-clNaTc'))
        if address_tag_div:
            data['address'] = address_tag_div.get_text(strip=True)
        else:
            # Fallback to the <p> tag selector if the div wasn't found
            address_tag_p = soup.select_one('section.sc-fQejPQ > div.sc-clNaTc') # More specific parent
            if address_tag_p:
                data['address'] = address_tag_p.get_text(strip=True)


    # Phone (Supplement if JSON-LD had only one or was missing)
    # Use setdefault to initialize if phone key doesn't exist yet
    data.setdefault('phone', [])
    if not data['phone']: # Check if list is still empty
        phone_section = soup.find('div', class_=re.compile(r'sc-ePZHVD')) # Class around phone icons/numbers
        if phone_section:
             phone_links = phone_section.find_all('a', href=lambda href: href and href.startswith('tel:'))
             # Add only unique numbers found via HTML to the list
             html_phones = set(a.get_text(strip=True) for a in phone_links)
             # Combine with JSON-LD phones if any, ensuring uniqueness
             data['phone'] = sorted(list(set(data['phone']).union(html_phones)))


    # Cuisines (Fallback if JSON-LD failed)
    if not data['what_they_serve']:
        cuisine_section = soup.find('div', class_=re.compile(r'sc-gVyKpa')) # Class around cuisine links
        if cuisine_section:
            cuisine_links = cuisine_section.find_all('a', class_=re.compile(r'sc-eXNvrr'))
            data['what_they_serve'] = ', '.join([a.get_text(strip=True) for a in cuisine_links])

    # Opening Hours (Fallback if JSON-LD failed)
    if not data['opening_hours']:
        # Try finding the simple "Opens at" text first
        opens_at_tag = soup.find('span', class_=re.compile(r'sc-kasBVs')) # Class for 'Opens at...'
        if opens_at_tag:
            # Combine with the preceding status text if available
            status_tag = opens_at_tag.find_previous_sibling('span', class_=re.compile(r'sc-iGPElx'))
            status_text = status_tag.get_text(strip=True) if status_tag else ""
            opens_text = opens_at_tag.get_text(strip=True)
            data['opening_hours'] = f"{status_text} - {opens_text}".strip(' - ') if status_text else opens_text
        else:
            # If the above wasn't found, look for the tooltip text (less ideal)
            tooltip_div = soup.find('div', class_=re.compile(r'sc-iFUGim')) # Tooltip content div
            if tooltip_div:
                 hours_spans = tooltip_div.find_all('span', class_=re.compile(r'sc-eqPNPO|sc-ileJJU'))
                 if len(hours_spans) >= 2:
                     data['opening_hours'] = f"{hours_spans[0].get_text(strip=True)} {hours_spans[1].get_text(strip=True)}"


    # Ratings (HTML Parsing needed for Delivery ratings, fallback for Dining)
    def extract_rating_details(rating_label_div):
        if not rating_label_div: return None, None
        # Navigate up more reliably to find the common parent containing both rating and label
        parent_container = rating_label_div.find_parent('div', class_=re.compile(r'sc-(?:keVrkP|bFADNz)')) # Matches either potential parent
        if not parent_container:
            # Try another potential parent structure seen in some cases
            parent_container = rating_label_div.find_parent('div', class_=re.compile(r'sc-1q7bklc-5'))
            if parent_container: # If found this way, look for siblings
                 rating_div = parent_container.find('div', class_=re.compile(r'sc-1q7bklc-10')) # The colored box
                 label_div = parent_container.find('div', class_=re.compile(r'sc-1q7bklc-7')) # The text part
                 if rating_div and label_div:
                      rating_val_tag = rating_div.find('div', class_=re.compile(r'sc-1q7bklc-1'))
                      rating_count_tag = label_div.find('div', class_=re.compile(r'sc-1q7bklc-8'))
                 else: return None, None # Couldn't find parts this way either
            else: return None, None # Parent not found
        else: # Original parent structure found
            rating_val_tag = parent_container.find('div', class_=re.compile(r'sc-1q7bklc-1'))
            rating_count_tag = parent_container.find('div', class_=re.compile(r'sc-1q7bklc-8'))


        rating_val = None
        rating_count = None

        if rating_val_tag:
            try:
                rating_val = float(rating_val_tag.get_text(strip=True))
            except (ValueError, TypeError):
                pass # Keep as None

        if rating_count_tag:
            count_text = rating_count_tag.get_text(strip=True).upper()
            # More robust cleaning: remove ALL non-digit characters except 'K' and '.'
            count_text_cleaned = re.sub(r'[^\d.K]', '', count_text)
            try:
                if 'K' in count_text_cleaned:
                    # Ensure 'K' is treated correctly, handle potential like '11.7K'
                    rating_count = int(float(count_text_cleaned.replace('K', '')) * 1000)
                elif count_text_cleaned: # Ensure not empty after cleaning
                    rating_count = int(float(count_text_cleaned)) # Allow float conversion first for safety, then int
                else:
                    rating_count = None
            except (ValueError, TypeError):
                 pass # Keep as None

        return rating_val, rating_count

    # Find the specific label divs
    dining_rating_label_div = soup.find('div', class_=re.compile(r'sc-1q7bklc-9'), string='Dining Ratings')
    delivery_rating_label_div = soup.find('div', class_=re.compile(r'sc-1q7bklc-9'), string='Delivery Ratings')

    # Only parse dining from HTML if not found in JSON-LD
    if data['dining_ratings'] is None or data['dining_ratings_count'] is None:
         data['dining_ratings'], data['dining_ratings_count'] = extract_rating_details(dining_rating_label_div)

    # Always try to parse delivery from HTML as it's not usually in JSON-LD
    data['delivery_ratings'], data['delivery_ratings_count'] = extract_rating_details(delivery_rating_label_div)


    # --- Menu Parsing (HTML is necessary here based on analysis) ---
    menu_sections = soup.find_all('section', class_=re.compile(r'sc-bZVNgQ')) # Sections like "Party Specials" etc.

    for section in menu_sections:
        category_tag = section.find('h4', class_=re.compile(r'sc-1hp8d8a-0'))
        if not category_tag:
            continue
        # Clean category name (remove trailing counts like '(4)')
        category_name = re.sub(r'\s*\(\d+\)$', '', category_tag.get_text(strip=True)).strip()
        if not category_name: # Skip if category name parsing failed
            continue

        item_containers = section.find_all('div', class_=re.compile(r'sc-jhLVlY')) # Individual item containers

        for item_cont in item_containers:
            menu_item = {
                "item_name": None,
                "description": None,
                "price": None,
                "category": category_name,
                "veg": None # True for veg, False for non-veg, None if undetermined
            }

            # Item Name
            name_tag = item_cont.find('h4', class_=re.compile(r'sc-cGCqpu'))
            if name_tag:
                menu_item['item_name'] = name_tag.get_text(strip=True)
            else:
                continue # Skip this item if it has no name

            # Description
            desc_p = item_cont.find('p', class_=re.compile(r'sc-gsxalj'))
            if desc_p:
                 desc_text = desc_p.get_text(strip=True)
                 # Remove common "read more" text variations and potential leading/trailing noise
                 cleaned_desc = re.sub(r'\s*\.\.\.\s*read more$', '', desc_text, flags=re.IGNORECASE).strip()
                 # Avoid adding description if it's effectively empty after cleaning
                 menu_item['description'] = cleaned_desc if cleaned_desc else None


            # Price
            price_span = item_cont.find('span', class_=re.compile(r'sc-17hyc2s-1'))
            if price_span:
                price_text = price_span.get_text(strip=True)
                # Clean price: remove currency symbols, commas etc.
                price_cleaned = re.sub(r'[^\d.]', '', price_text)
                try:
                    # Handle cases where price is zero (like items with options)
                    parsed_price = float(price_cleaned) if price_cleaned else 0.0
                    menu_item['price'] = parsed_price
                except (ValueError, TypeError):
                    menu_item['price'] = 0.0 # Default to 0.0 if parsing fails but tag exists

            # If price tag wasn't found at all, set to None explicitly
            else:
                 menu_item['price'] = None

             # --- Determine Veg/Non-Veg status (with priority) ---
            # Priority 1: Check for the explicit 'type' attribute on a nested div
            veg_type_div = item_cont.find('div', attrs={'type': 'veg'})
            non_veg_type_div = item_cont.find('div', attrs={'type': 'non-veg'})

            if veg_type_div:
                menu_item['veg'] = True
            elif non_veg_type_div:
                menu_item['veg'] = False

            # Priority 2: Check SVG icon <use> tag 'href' if type attribute wasn't found
            if menu_item['veg'] is None:
                svg_indicator = item_cont.find('svg', class_=re.compile(r'sc-eOnLuU'))
                if svg_indicator:
                    use_tag = svg_indicator.find('use')
                    if use_tag:
                        href = use_tag.get('href', '').strip()
                        if '#veg-icon' in href:
                            menu_item['veg'] = True
                        elif '#non-veg-icon' in href:
                            menu_item['veg'] = False

                        # Priority 3: Check <use> tag 'fill' color if href didn't match
                        if menu_item['veg'] is None:
                             fill_color = use_tag.get('fill', '').strip().upper()
                             if fill_color == '#3AB757':
                                 menu_item['veg'] = True
                             elif fill_color == '#BF4C43':
                                 menu_item['veg'] = False

                    # Priority 4: Check <svg> tag 'fill' color if <use> tag check failed
                    if menu_item['veg'] is None:
                         fill_color = svg_indicator.get('fill', '').strip().upper()
                         if fill_color == '#3AB757':
                             menu_item['veg'] = True
                         elif fill_color == '#BF4C43':
                             menu_item['veg'] = False
            # --- End Veg/Non-Veg check ---

            # Add item only if it has a name and a valid price (including 0.0)
            if menu_item['item_name'] and menu_item['price'] is not None:
                 data['menu'].append(menu_item)

    # Basic check if we got anything useful
    if not data['restraunt_name']:
        print("Warning: Could not extract restaurant name.")

    return data

def scrape_zomato_restaurants(url_list):
    """
    Scrapes Zomato restaurant data from a list of URLs.
    
    Args:
        url_list: A list of Zomato restaurant URLs to scrape.
        
    Returns:
        A list of dictionaries containing restaurant data.
    """
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')

    results = []
    output_json_path = 'data/restaurants_data.json'

    with webdriver.Chrome(options=chrome_options) as driver:
        for idx, url in enumerate(url_list):
            print(f"Processing {idx+1}/{len(url_list)}: {url}")
            try:
                driver.get(url)
                time.sleep(5)  # Wait for the page to load
                html = driver.page_source
                data = parse_restaurant_html(html)
                # Add the source URL to the data for reference
                data['source_url'] = url 
                results.append(data)
                # Save progress after each URL
                with open(output_json_path, 'w', encoding='utf-8') as outfile:
                    json.dump(results, outfile, indent=4)
                print(f"Done: {url}")
            except Exception as e:
                print(f"Error processing {url}: {e}")
                # Add error information along with the URL
                results.append({"url": url, "error": str(e)}) 
                with open(output_json_path, 'w', encoding='utf-8') as outfile:
                    json.dump(results, outfile, indent=4)

    print(f"\nAll done. Data saved to {output_json_path}")
    return results

def main():
    # List of Zomato restaurant URLs to scrape
    url_list = [
        'https://www.zomato.com/ncr/burger-king-karol-bagh-new-delhi/order',
        'https://www.zomato.com/ncr/house-of-biryan-biryani-kepsa-and-more-connaught-place-new-delhi/order',
        'https://www.zomato.com/ncr/salad-days-gole-market-new-delhi/order',
        'https://www.zomato.com/ncr/nomad-pizza-traveller-series-connaught-place-new-delhi/order',
        'https://www.zomato.com/ncr/kfc-2-paharganj-new-delhi/order',
        'https://www.zomato.com/ncr/the-burger-club-1-connaught-place-new-delhi/order',
        'https://www.zomato.com/ncr/dominos-pizza-4-connaught-place-new-delhi/order',
        'https://www.zomato.com/ncr/burger-singh-big-punjabi-burgers-moti-bagh-new-delhi/order',
        'https://www.zomato.com/bangalore/imperio-restaurant-koramangala-5th-block-bangalore/order',
        'https://www.zomato.com/bangalore/gochick-koramangala-5th-block-bangalore/order'
    ]
    
    # Start scraping
    scrape_zomato_restaurants(url_list)

if __name__ == "__main__":
    main()