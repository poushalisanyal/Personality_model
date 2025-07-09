from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Literal
from pydantic import BaseModel
from typing import Annotated, Literal
import pickle
import pandas as pd
import numpy as np
with open('model/Personality_model.pkl','rb') as f:
    model=pickle.load(f)

with open('model/stage_fear_encoder.pkl', 'rb') as f:
    stage_fear_encoder = pickle.load(f)

with open('model/drained_encoder.pkl', 'rb') as f:
    drained_encoder = pickle.load(f)

with open('model/personality_encoder.pkl', 'rb') as f:
    personality_encoder = pickle.load(f)

app=FastAPI()

class PersonalityFeatures(BaseModel):
    Time_spent_Alone: float
    Stage_fear: Literal['Yes','No']
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: Literal['Yes','No']
    Friends_circle_size: float
    Post_frequency: float

@app.post("/predict")
def predict(data: PersonalityFeatures):
    try:
        # Encode categorical features
        encoded_stage_fear = stage_fear_encoder.transform([data.Stage_fear])[0]
        encoded_drained = drained_encoder.transform([data.Drained_after_socializing])[0]

        # Prepare input for the model
        input_features = np.array([[
            data.Time_spent_Alone,
            encoded_stage_fear,
            data.Social_event_attendance,
            data.Going_outside,
            encoded_drained,
            data.Friends_circle_size,
            data.Post_frequency
        ]])

        # Make prediction
        prediction = model.predict(input_features)

        # Decode predicted label
        predicted_personality = personality_encoder.inverse_transform([prediction[0]])[0]

        return {"predicted_personality": predicted_personality}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )
@app.get("/")
def read_root():
    return {"message": "Personality Prediction API is running!"}
