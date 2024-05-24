from fastapi import FastAPI
from app.routes import urban_centres

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Urban Centres API"}

app.include_router(urban_centres.router)