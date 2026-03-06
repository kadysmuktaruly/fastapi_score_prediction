from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression

# Create app
app = FastAPI()

# ---- Train a simple ML model ----
# Fake dataset: hours studied vs exam score
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([50, 55, 65, 70, 80, 90])

model = LinearRegression()
model.fit(X, y)

# ---- Define request format ----
class StudyInput(BaseModel):
    hours_studied: float

# ---- Create prediction endpoint ----
@app.post("/predict")
def predict(data: StudyInput):
    prediction = model.predict(np.array([[data.hours_studied]]))
    if float(prediction[0]) <100.0:
        val = float(prediction[0])
    else:        val = 100.0
    return {"predicted_score": val}