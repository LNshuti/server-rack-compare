## Design Document for FastAPI Application with Conda Environment, requirements.txt, and Docker

### Overview
This document outlines the steps to create and deploy a FastAPI application using a Conda environment, a `requirements.txt` file, and Docker. The application will provide information about urban centres in the Americas based on the GHS Urban Centre Database.

### Objectives
- Set up a Conda environment for the FastAPI application.
- Create a `requirements.txt` file for the application dependencies.
- Develop a Dockerfile to containerize the application.
- Implement a FastAPI application to serve data about urban centres.

### Prerequisites
- Docker installed on the development machine.
- Conda installed on the development machine.
- Basic understanding of FastAPI, Python, and Docker.

### Directory Structure
```
fastapi-urban-centres/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── __init__.py
│       └── urban_centres.py
├── data/
│   └── urban_centres.csv
├── environment.yml
├── requirements.txt
└── Dockerfile
```

### Steps

#### 1. Set Up Conda Environment
Create an `environment.yml` file to define the Conda environment:

```yaml
name: fastapi-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - fastapi
  - uvicorn
  - pandas
  - requests
  - pip
  - pip:
      - fastapi
      - uvicorn
      - pandas
      - requests
```

Create the Conda environment using the following command:

```sh
conda env create -f environment.yml
```

#### 2. Create a `requirements.txt` File
Generate the `requirements.txt` file from the Conda environment:

```sh
pip freeze > requirements.txt
```

#### 3. Develop the FastAPI Application
Create the FastAPI application files.

##### `app/main.py`
```python
from fastapi import FastAPI
from app.routes import urban_centres

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Urban Centres API"}

app.include_router(urban_centres.router)
```

##### `app/routes/urban_centres.py`
```python
from fastapi import APIRouter
import pandas as pd

router = APIRouter()

@router.get("/urban_centres")
def get_urban_centres():
    df = pd.read_csv("data/urban_centres.csv")
    return df.to_dict(orient="records")
```

##### Sample Data
Store the sample data in `data/urban_centres.csv`.

#### 4. Create a Dockerfile
Develop the Dockerfile to containerize the FastAPI application:

```Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME FastAPI

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

#### 5. Build and Run the Docker Container
Build the Docker image:

```sh
docker build -t fastapi-urban-centres .
```

Run the Docker container:

```sh
docker run -p 80:80 fastapi-urban-centres
```

### Additional Information

#### Urban Centres Data
The GHS Urban Centre Database from the European Commission’s Joint Research Centre provides data on urban centres. Urban centres are defined by specific thresholds on resident population and built-up surface share in a 1x1 km uniform global grid. This methodology provides consistent city definitions across geographical locations and over time, although boundaries may differ from administrative city boundaries.

### Conclusion
This design document provides a comprehensive guide to setting up, developing, and deploying a FastAPI application using a Conda environment, a `requirements.txt` file, and Docker. The application will serve data on urban centres in the Americas, utilizing data from the GHS Urban Centre Database.