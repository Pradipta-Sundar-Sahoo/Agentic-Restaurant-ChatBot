-- Optional: Enable extensions if needed for advanced indexing (like trigram)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE restaurant_menu_flat (
    -- Item Details
    item_id SERIAL PRIMARY KEY, -- Still useful to uniquely identify each item row
    item_name TEXT NOT NULL,
    item_description TEXT,
    item_price NUMERIC(10, 2) NOT NULL CHECK (item_price >= 0),
    item_category TEXT,
    item_is_veg BOOLEAN NOT NULL,

    -- Restaurant Details (Duplicated per item)
    restaurant_name TEXT NOT NULL,
    restaurant_address TEXT,
    restaurant_phone_numbers TEXT[], -- Store as an array of text
    restaurant_cuisines TEXT[],      -- Store as an array of text
    restaurant_opening_hours JSONB,  -- Store the original opening hours array structure as JSONB
    restaurant_dining_rating NUMERIC(2, 1) CHECK (restaurant_dining_rating >= 0 AND restaurant_dining_rating <= 5),
    restaurant_delivery_rating NUMERIC(2, 1) CHECK (restaurant_delivery_rating >= 0 AND restaurant_delivery_rating <= 5),
    restaurant_dining_ratings_count INTEGER DEFAULT 0 CHECK (restaurant_dining_ratings_count >= 0),
    restaurant_delivery_ratings_count INTEGER DEFAULT 0 CHECK (restaurant_delivery_ratings_count >= 0),
    restaurant_source_url TEXT,
    description_clean TEXT,
    city TEXT,
    -- Timestamps (Optional but good practice)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP -- Use trigger below
);

-- Indexing Strategy (Crucial for a large denormalized table)

-- Indexes for common filtering on items
CREATE INDEX idx_flat_item_name ON restaurant_menu_flat (item_name);
CREATE INDEX idx_flat_item_name_trgm ON restaurant_menu_flat USING gin (item_name gin_trgm_ops); -- For fuzzy search on item name
CREATE INDEX idx_flat_item_price ON restaurant_menu_flat (item_price);
CREATE INDEX idx_flat_item_category ON restaurant_menu_flat (item_category); -- If filtering by category often
CREATE INDEX idx_flat_item_is_veg ON restaurant_menu_flat (item_is_veg);
CREATE INDEX idx_flat_city ON restaurant_menu_flat (city);

-- Indexes for common filtering on restaurants
CREATE INDEX idx_flat_restaurant_name ON restaurant_menu_flat (restaurant_name);
CREATE INDEX idx_flat_restaurant_name_trgm ON restaurant_menu_flat USING gin (restaurant_name gin_trgm_ops); -- For fuzzy search on restaurant name
CREATE INDEX idx_flat_restaurant_delivery_rating ON restaurant_menu_flat (restaurant_delivery_rating);
CREATE INDEX idx_flat_restaurant_dining_rating ON restaurant_menu_flat (restaurant_dining_rating);

-- Indexes for Array/JSONB columns (use GIN indexes for element/key lookups)
-- Index for searching within phone numbers (e.g., WHERE '+9111...' = ANY(restaurant_phone_numbers))
CREATE INDEX idx_flat_restaurant_phones_gin ON restaurant_menu_flat USING GIN (restaurant_phone_numbers);

-- Index for searching within cuisines (e.g., WHERE 'Pizza' = ANY(restaurant_cuisines))
CREATE INDEX idx_flat_restaurant_cuisines_gin ON restaurant_menu_flat USING GIN (restaurant_cuisines);

-- Index for searching within opening hours JSON (e.g., finding restaurants open on 'Monday')
-- Example: WHERE restaurant_opening_hours @> '[{"day": "Monday"}]'::jsonb
CREATE INDEX idx_flat_restaurant_opening_hours_gin ON restaurant_menu_flat USING GIN (restaurant_opening_hours);


-- Optional: Composite index if you often filter by restaurant AND item properties
CREATE INDEX idx_flat_resto_item_veg ON restaurant_menu_flat (restaurant_name, item_is_veg);
CREATE INDEX idx_flat_resto_item_price ON restaurant_menu_flat (restaurant_name, item_price);


-- Trigger function to automatically update `updated_at` columns
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply the trigger to the table
CREATE TRIGGER set_timestamp_flat
BEFORE UPDATE ON restaurant_menu_flat
FOR EACH ROW
EXECUTE FUNCTION trigger_set_timestamp();



CREATE INDEX idx_flat_description_clean_trgm ON restaurant_menu_flat USING gin (description_clean gin_trgm_ops);
-- If full-text search is needed across descriptions
CREATE INDEX idx_flat_description_clean_fts ON restaurant_menu_flat
  USING gin (to_tsvector('english', description_clean));