import ee
import numpy as np
import matplotlib.pyplot as plt
import geemap
import sys

## 1. USER CONFIGURATION ##

GEE_ASSET_PATH = 'projects/bramhacks-2025-477622/assets/brampton_boundary' # <-- DEFINED HERE
FIRE_INCIDENTS = 'projects/bramhacks-2025-477622/assets/brampton_fires'
LANDUSE_ASSET = 'projects/bramhacks-2025-477622/assets/land_use'

#MUST add up to 1.0
WEIGHTS = {
    'heat': 0.15,   # Land Surface Temperature
    'density': 0.20, # Nighttime Lights (proxy for human activity/buildings)
    'fuel': 0.30,    # NDVI (proxy for dry vegetation)
    'water': 0.175,   # Distance from water
    'slope': 0.175,   # Elevation slope
}

# time range for analysis
START_DATE = '2012-01-01'
END_DATE = '2016-12-31'

def main():
    try:
        ee.Initialize(project='bramhacks-2025-477622')
        print("GEE authentication successful.")
    except Exception as e:
        print(f"Error initializing GEE: {e}")
        print("Please run 'earthengine-api authenticate' in your terminal.")
        sys.exit(1)
        
    if abs(sum(WEIGHTS.values()) - 1.0) > 0.001:
        print(f"Error: Weights add up to {sum(WEIGHTS.values())}, not 1.0. Please check WEIGHTS.")
        sys.exit(1)

    # --- PHASE 1: LOAD GEE ASSETS ---
    print("Phase 1: Loading GEE assets...")
    try:
        aoi = ee.FeatureCollection(GEE_ASSET_PATH).union()
        region = aoi.geometry()
        fire_points = ee.FeatureCollection(FIRE_INCIDENTS)
        print(f"Loaded Brampton boundary and {fire_points.size().getInfo()} fire incidents.")
    except Exception as e:
        print(f"Error loading GEE assets: {e}")
        print("Check your GEE_ASSET_PATH and GEE_FIRE_ASSET paths in the config.")
        sys.exit(1)
        
    # --- Load Land Use Data ---
    landuse = ee.FeatureCollection(LANDUSE_ASSET)

    residential_mask = landuse.filter(ee.Filter.eq('OP_LANDUSE', 'RESIDENTIAL')).geometry()

    openspace_mask = landuse.filter(ee.Filter.eq('OP_LANDUSE', 'OPENSPACE')).geometry()
    
    # --- NEW: Categorize Fire Points ---
    # This "joins" the fire points to the landuse polygons
    print("Categorizing all fire incidents by land use zone...")
    
    spatial_filter = ee.Filter.intersects(
        leftField='.geo',
        rightField='.geo',
        maxError=1
    )
    
    # This is the join
    join = ee.Join.saveFirst(
        matchKey='landuse_data' # This property will hold the matching polygon
    )
    
    # Apply the join
    # This creates a new collection of fire points, each with a
    # 'landuse_data' property
    categorized_fire_points = join.apply(fire_points, landuse, spatial_filter)

    # --- PHASE 2: PROCESS SPACE-BASED LAYERS ---
    print("Phase 2: Processing space-based data (this takes a moment)...")
    
    # Layer 1: Land Surface Temperature (Heat)
    lst = ee.ImageCollection('MODIS/061/MOD11A2') \
            .filterDate(START_DATE, END_DATE) \
            .select('LST_Day_1km') \
            .mean() \
            .clip(region) \
            .rename('heat')

    # Layer 2: Nighttime Lights (Human Density)
    ntl = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG') \
            .filterDate(START_DATE, END_DATE) \
            .select('avg_rad') \
            .mean() \
            .clip(region) \
            .rename('density')

    # Layer 3: NDVI (Vegetation/Fuel)
    def maskS2clouds(image):
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
               qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask)

    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
           .filterDate(START_DATE, END_DATE) \
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
           .filterBounds(region) \
           .map(maskS2clouds) \
           .mean()

    ndvi = s2.normalizedDifference(['B8', 'B4']).multiply(-1).rename('fuel')

    # Layer 4: Elevation & Slope
    srtm = ee.Image('USGS/SRTMGL1_003')
    elevation = srtm.select('elevation').clip(region)
    slope = ee.Terrain.slope(elevation).rename('slope')

    # Layer 5: Distance from Water
    # We load it as a collection, filter for the 2021 image, and take the first() result.
    world_cover = ee.ImageCollection('ESA/WorldCover/v200') \
                    .filter(ee.Filter.eq('system:index', '2021')) \
                    .first() \
                    .select('Map')
    
    water_mask = world_cover.eq(80) # 80 is the class for "Permanent water bodies"
    distance_to_water = water_mask.distance(
        ee.Kernel.euclidean(50000, 'meters')
    ).unmask(0).rename('water')

    # --- PHASE 3: NORMALIZE & BUILD MODEL ---
    print("Phase 3: Normalizing layers and building risk model...")

    def normalize(image):
        # Get the band name (e.g., 'heat', 'density')
        band_name = image.bandNames().get(0)
        
        # Get the min/max stats for that band
        stats = image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=region,
            scale=1000, # Use a coarse scale for stats
            bestEffort=True
        )
        
        # This is the correct way to get a value or a default
        min_val = ee.Number(stats.get(ee.String(band_name).cat('_min'), 0))
        max_val = ee.Number(stats.get(ee.String(band_name).cat('_max'), 1))
        
        # Prevent 0/0 error if min == max
        # If min=max, set min=0 and max=1. This makes the whole image 0.
        safe_min = ee.Algorithms.If(min_val.eq(max_val), 0, min_val)
        safe_max = ee.Algorithms.If(min_val.eq(max_val), 1, max_val)

        # Pass the safe min and max to unitScale
        return image.unitScale(safe_min, safe_max)

    # Normalize all layers and combine into one image
    all_layers = normalize(lst).addBands(
        normalize(ntl)
    ).addBands(
        normalize(ndvi)
    ).addBands(
        normalize(distance_to_water)  
    ).addBands(
        normalize(slope)             
    )

    # Apply the "Risk Recipe" (Weighted Overlay)
    risk_image = all_layers.expression(
        '(b("heat") * W_HEAT) + '
        '(b("density") * W_DENSITY) + '
        '(b("fuel") * W_FUEL) + '
        '(b("water") * W_WATER) + '
        '(b("slope") * W_SLOPE)',
        {
            'W_HEAT': WEIGHTS['heat'],
            'W_DENSITY': WEIGHTS['density'],
            'W_FUEL': WEIGHTS['fuel'],
            'W_WATER': WEIGHTS['water'],
            'W_SLOPE': WEIGHTS['slope'],
        }
    ).rename('risk_score') \
    .unmask(0) # <-- ADD THIS LINE

    # --- PHASE 4: DOWNLOAD & PLOT PREDICTION MAP ---
    print("Phase 4: Downloading risk heatmap (can be slow)...")
    
    # Download the final risk image as a NumPy array
    # Scale=200 (200m) is a good balance of detail and speed.
    try:
        risk_numpy = geemap.ee_to_numpy(risk_image, region=region, scale=200)
        risk_numpy = np.squeeze(risk_numpy)
        risk_numpy[risk_numpy == 0] = np.nan
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading NumPy array: {e}")
        print("Try increasing the 'scale' (e.g., to 500) if it times out.")
        sys.exit(1)

    print("Plotting 'brampton_risk_heatmap.png'...")
    plt.figure(figsize=(10, 10))
    plt.imshow(risk_numpy, cmap='hot')
    plt.title('Predicted Fire Risk Heatmap - Brampton')
    plt.colorbar(label='Fire Risk Score (0.0 to 1.0)')
    plt.axis('off')
    plt.savefig('brampton_risk_heatmap.png', dpi=300)
    print("Saved 'brampton_risk_heatmap.png'")

# --- PHASE 5: VALIDATE & PLOT PROOF ---
    print("Phase 5: Validating model and plotting proof...")
    
    # --- 1. Get Risk at Fire Points ---
    # Use our NEW categorized collection from Phase 1
    fire_risk_scores = risk_image.sampleRegions(
        collection=categorized_fire_points, # <-- USING THE JOINED DATA
        scale=200, # Using your 200m scale
        geometries=True,
        tileScale=16
    ).getInfo()

    # --- 2. Calculate Stats & Sort ---
    # Create dictionaries to hold the lists of scores for each category
    scores_by_category = {
        'RESIDENTIAL': [],
        'OPENSPACE': [],
        'OTHER': [] # For fires in other zones
    }
    
    all_fire_coords_lon = []
    all_fire_coords_lat = []
    
    for f in fire_risk_scores['features']:
        score = f['properties'].get('risk_score')
        if score is None:
            continue # Skip if no risk score

        # Get the landuse data that we joined in Phase 1
        landuse_data = f['properties'].get('landuse_data')
        
        category = 'OTHER' # Default category
        
        if landuse_data is not None:
            # Get the landuse name from the joined data
            category = landuse_data['properties']['OP_LANDUSE']

        # Add the score to the correct list
        # We group all residential types together
        if category in ['RESIDENTIAL', 'ESTATE RESIDENTIAL', 'RESIDENTIAL SEE SECTION 4.2.1.16']:
            scores_by_category['RESIDENTIAL'].append(score)
        elif category in scores_by_category:
            scores_by_category[category].append(score)
        else:
            scores_by_category['OTHER'].append(score)
            
        # Get coords for the map plot
        coords = f['geometry']['coordinates']
        all_fire_coords_lon.append(coords[0])
        all_fire_coords_lat.append(coords[1])
        
    
    # --- 3. Get Avg. Risk for Each Land Use Type ---
    def get_avg_risk(mask):
        """Helper function to get avg risk for a masked area."""
        try:
            if mask.area(maxError=1000).getInfo() == 0:
                print("Warning: A land use mask was empty. Skipping.")
                return 0.0
            
            val = risk_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=mask,
                scale=1000,
                bestEffort=True,
                tileScale=16
            ).get('risk_score').getInfo()
            return val if val is not None else 0.0
        except Exception as e:
            print(f"Warning: Could not calculate risk for a mask. Error: {e}")
            return 0.0

    print("Calculating risk for land use categories...")
    avg_residential_risk = get_avg_risk(residential_mask)
    avg_openspace_risk = get_avg_risk(openspace_mask)

    # --- 4. Create the Final Ranked List ---
    print("\n--- MODEL VALIDATION (GRANULAR RISK PROFILE) ---")

    def print_comparison(category_name, fire_scores, land_risk):
        """Helper to print the final comparison."""
        print(f"\n  CATEGORY: {category_name}")
        if not fire_scores: # Check if the list is empty
            print(f"    - Avg. Land Risk: {land_risk:.4f} (No fire incidents in this category)")
            return
        
        avg_fire_risk = np.mean(fire_scores)
        lift = avg_fire_risk - land_risk
        print(f"    - Avg. Fire Point Risk: {avg_fire_risk:.4f}  ({len(fire_scores)} fires)")
        print(f"    - Avg. Land Risk:       {land_risk:.4f}")
        print(f"    - MODEL LIFT:           {lift:+.4f}")

    # Print the comparisons
    print_comparison('Residential', scores_by_category['RESIDENTIAL'], avg_residential_risk)
    print_comparison('Open Space', scores_by_category['OPENSPACE'], avg_openspace_risk)
    
    if scores_by_category['OTHER']:
        print(f"\n  CATEGORY: Other Fires (e.g., Industrial, Utility, etc.)")
        print(f"    - Avg. Fire Point Risk: {np.mean(scores_by_category['OTHER']):.4f}  ({len(scores_by_category['OTHER'])} fires)")

    print("\n--- (A positive 'LIFT' means our model correctly found hotspots) ---\n")


    # --- 5. Plot the "Proof" map ---
    print("Plotting 'brampton_validation_map.png'...")
    
    bbox = region.bounds().getInfo()['coordinates'][0]
    lon_min, lat_min = bbox[0]
    lon_max, lat_max = bbox[2]

    # Use the 'all_fire_coords' lists
    fire_x = (np.array(all_fire_coords_lon) - lon_min) / (lon_max - lon_min)
    fire_y = (np.array(all_fire_coords_lat) - lat_min) / (lat_max - lat_min) # Corrected Y-coord

    height, width = risk_numpy.shape
    
    plt.figure(figsize=(10, 10))
    plt.imshow(risk_numpy, cmap='hot', extent=[0, width, 0, height])
    
    plt.scatter(
        fire_x * width, 
        fire_y * height, 
        c='#00FFFF',        
        s=12.5,             
        alpha=1.0,         
        edgecolors='black',
        linewidths=0.5
    )
    
    plt.title('Validation Map: Fire Risk Heatmap vs. Actual Fire Incidents')
    plt.axis('off')
    plt.savefig('brampton_validation_map.png', dpi=300)
    print("Saved 'brampton_validation_map.png'")
    print("\n--- HACKATHON SCRIPT COMPLETE ---")

if __name__ == '__main__':
    main()