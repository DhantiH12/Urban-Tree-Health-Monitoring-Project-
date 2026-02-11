# Urban Tree Health Monitoring System

## Overview

The **Urban Tree Health Monitoring System** is a fully offline, AI-powered computer vision application designed to assess the health of urban trees using images. The system allows users to upload tree images through an interactive dashboard, automatically predicts tree health status, and provides actionable care and maintenance recommendations to support urban green infrastructure management.

## Key Features

- Upload tree images via a professional offline dashboard
- Automatic tree health classification into:
  - Healthy
  - Moderate / Stressed
  - Unhealthy / Diseased
- Prediction confidence score and class probability distribution
- Identification of likely health problems for stressed and unhealthy trees
- Practical care and maintenance recommendations
- Local SQLite database for storing predictions with metadata
- Area-wise and city-level analytics with visual summaries
- Export stored records as CSV files
- Supports multiple image formats (JPG, PNG, BMP, TIFF, WEBP)
- Fully offline operation (no cloud APIs or internet required)

### Dataset Access

The dataset used for training and evaluation is hosted on Google Drive:

**Dataset Link:**  
https://drive.google.com/drive/folders/1KNba49rdf8T9BJmJ00MntgEaLKlim_12?usp=drive_link

## Installation and Setup

### 1. Clone the Repository

git clone https://github.com/your-username/urban-tree-health-monitoring.git


### 2. Create a Virtual Environment

python -m venv venv


Activate the environment:
- **Windows:** `venv\Scripts\activate`
- **Linux / macOS:** `source venv/bin/activate`

### 3. Install Dependencies

pip install -r requirements.txt

