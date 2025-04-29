import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_housing_data(num_samples=1000):
    # Define locations in Pune with their relative price factors
    locations = {
        'Baner': 1.2,
        'Kothrud': 1.3,
        'Wakad': 1.1,
        'Hadapsar': 0.9,
        'Hinjewadi': 1.15,
        'Viman Nagar': 1.25,
        'Kalyani Nagar': 1.4,
        'Kondhwa': 0.85,
        'Aundh': 1.35,
        'Magarpatta': 1.2
    }

    # Generate random data
    np.random.seed(42)  # for reproducibility
    
    # Base features
    area = np.random.normal(1200, 400, num_samples)  # sq ft
    bedrooms = np.random.choice([1, 2, 3, 4, 5], num_samples, p=[0.05, 0.3, 0.4, 0.2, 0.05])
    bathrooms = np.clip(np.random.normal(bedrooms * 0.8, 0.5, num_samples).round(), 1, 5)
    location_names = np.random.choice(list(locations.keys()), num_samples)
    location_factors = np.array([locations[loc] for loc in location_names])
    
    # Building age (0-20 years)
    age = np.random.uniform(0, 20, num_samples).round(1)
    
    # Amenities (0-10 scale)
    amenities = np.random.randint(1, 11, num_samples)
    
    # Floor number (1-20)
    floor = np.random.randint(1, 21, num_samples)
    
    # Calculate base price (₹ per sq ft)
    base_price_per_sqft = 5000  # Base price per sq ft in Pune
    
    # Calculate final price with various factors
    price = (
        area * 
        base_price_per_sqft * 
        location_factors * 
        (1.1 ** (bedrooms - 2)) *  # bedroom factor
        (0.98 ** age) *  # age depreciation
        (1 + amenities * 0.02) *  # amenities premium
        (1 + floor * 0.01)  # floor premium
    )
    
    # Add some random noise to prices (±10%)
    price = price * np.random.uniform(0.9, 1.1, num_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'area_sqft': area.round(),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'location': location_names,
        'building_age': age,
        'amenities_score': amenities,
        'floor': floor,
        'price': price.round(-3)  # round to nearest thousand
    })
    
    return df

def main():
    # Generate dataset
    df = generate_housing_data(5000)
    
    # Save to CSV
    output_file = 'pune_housing_data.csv'
    df.to_csv(output_file, index=False)
    
    # Print sample statistics
    print("\nPune Housing Dataset Generated Successfully!")
    print(f"\nDataset Shape: {df.shape}")
    print("\nSample Statistics:")
    print(f"Average Price: ₹{df['price'].mean():,.2f}")
    print(f"Price Range: ₹{df['price'].min():,.2f} - ₹{df['price'].max():,.2f}")
    print(f"\nLocation Distribution:\n{df['location'].value_counts()}")
    
    print(f"\nDataset saved as: {output_file}")

if __name__ == "__main__":
    main()