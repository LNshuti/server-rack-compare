from fastapi import APIRouter
import pandas as pd

router = APIRouter()

@router.get("/urban_centres")
def get_urban_centres():
    df = pd.read_csv("data/urban_centres.csv")
    return df.to_dict(orient="records")