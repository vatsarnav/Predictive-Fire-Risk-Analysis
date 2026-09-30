# Predictive Fire Risk Mapping

Predictive Fire Risk Mapping is a geospatial analytics project that identifies areas with elevated fire risk by combining environmental and geographic data into a weighted risk model.

The project uses **Python and Google Earth Engine** to process spatial data, calculate fire-risk scores, and visualize risk patterns across **Brampton, Ontario**. The model was evaluated against historical observations and achieved approximately **89% predictive accuracy**.

## Overview

Fire risk is influenced by multiple environmental conditions rather than a single factor. This project brings these factors together into one analytical model to provide a geographic representation of areas with varying levels of fire risk.

The workflow uses geospatial datasets from Google Earth Engine, processes relevant variables, applies a weighted scoring methodology, and generates risk classifications that can be mapped and analyzed.

## Key Features

- Processes environmental and geographic data using Google Earth Engine
- Combines multiple fire-risk indicators into a weighted risk score
- Identifies geographic areas with relatively higher fire risk
- Produces map-based visualizations for easier interpretation
- Uses historical observations to evaluate predictive performance
- Achieved approximately **89% predictive accuracy** during model evaluation

## Tech Stack

**Languages and Tools**
- Python
- Google Earth Engine

**Concepts**
- Geospatial Analysis
- Data Preprocessing
- Feature Normalization
- Weighted Risk Modelling
- Predictive Analytics
- Data Visualization

## How It Works

### 1. Data Collection

Relevant environmental and geographic datasets are collected and processed using Google Earth Engine.

### 2. Data Preprocessing

The selected variables are cleaned and transformed into formats suitable for comparison and analysis.

### 3. Risk Factor Normalization

Variables with different scales and units are normalized so they can contribute consistently to the final model.

### 4. Weighted Risk Calculation

Each selected factor contributes to an overall fire-risk score based on its assigned weight.

```text id="m4mtcf"
Fire Risk Score = Σ (Normalized Risk Factor × Assigned Weight)
```

### 5. Risk Classification

The resulting scores are grouped into risk categories to distinguish lower-risk areas from areas with elevated predicted risk.

### 6. Geospatial Visualization

The calculated risk levels are mapped geographically, making it easier to identify spatial patterns and potential high-risk areas.

### 7. Model Evaluation

Predicted risk levels are compared with historical observations to evaluate how effectively the model identifies fire-risk patterns.

## Results

The final model achieved approximately **89% predictive accuracy** during evaluation.

The resulting maps demonstrate how environmental and geographic information can be combined to identify spatial fire-risk patterns and communicate them through geographic visualization.

## Getting Started

### Prerequisites

Before running the project, make sure you have:

- Python installed
- A Google Earth Engine account
- Google Earth Engine authentication configured
- Required Python dependencies installed

### Clone the Repository

```bash id="10xd2e"
git clone <repository-url>
cd <repository-name>
```

### Install Dependencies

If the repository contains a `requirements.txt` file:

```bash id="w67pvo"
pip install -r requirements.txt
```

### Google Earth Engine Authentication

Authenticate your Earth Engine account if required:

```bash id="9e8h8w"
earthengine authenticate
```

After authentication, run the scripts or notebooks included in the repository.

## Future Improvements

Potential improvements include:

- Incorporating additional environmental variables
- Testing alternative weighting techniques
- Expanding the analysis to additional geographic regions
- Exploring machine-learning approaches for dynamic fire-risk prediction
- Improving model validation using larger historical datasets

## Contributors

**Omika Kansra**  
Computer Science (Honours, Co-op)  
Algoma University

**Arnav Vats** 
Computer Science

Algoma University

## Disclaimer

This project was developed for educational and research purposes. The generated risk classifications are analytical predictions and should not be interpreted as official fire warnings or emergency-management guidance.
