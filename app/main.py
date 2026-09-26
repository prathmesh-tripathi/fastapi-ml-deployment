from fastapi import FastAPI
from pydantic import BaseModel

from app.model import model


app = FastAPI(
    title="Iris Prediction API",
    description="A simple ML prediction API using FastAPI",
    version="1.0.0",
)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {
        "message": "Iris Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction = model.predict(features)

    # 0 → setosa
    # 1 → versicolor
    # 2 → virginica    

    mapping_dict = {0 : "setosa",
                    1 : "versicolor",
                    2 : "virginica"}  

    return {
        "prediction": mapping_dict[int(prediction[0])]
    }