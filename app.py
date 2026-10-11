import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session

app = FastAPI(title="PNG Maker API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session variable for lazy loading
session = None

def get_session():
    global session
    if session is None:
        session = new_session("u2netp")
    return session

# Supports both GET and HEAD for Render's health checker
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "ok", "message": "PNG Maker API is running on Render!"}

@app.post("/api/make-png")
async def make_png(
    file: UploadFile = File(...),
    mode: str = Form("general")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if mode == "signature":
            rgba = image.convert("RGBA")
            data = np.array(rgba)

            r, g, b = data[..., 0], data[..., 1], data[..., 2]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b

            paper_threshold = 215.0
            ink_threshold = 80.0

            alpha = (paper_threshold - luminance) / (paper_threshold - ink_threshold) * 255.0
            alpha = np.clip(alpha, 0, 255).astype(np.uint8)
            data[..., 3] = alpha

            output_image = Image.fromarray(data, mode="RGBA")
        else:
            model_session = get_session()
            output_image = remove(image, session=model_session)

        output_bytes = io.BytesIO()
        output_image.save(output_bytes, format="PNG")
        output_bytes.seek(0)

        return Response(content=output_bytes.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))