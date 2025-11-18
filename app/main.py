from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.model import predict
from PIL import Image
import io

app = FastAPI(title="Animal Image Classifier")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image from uploaded file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Run prediction
        label, confidence, probs = predict(img)

        return JSONResponse(content={
            "predicted_label": label,
            "confidence": round(confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in probs.items()}
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)