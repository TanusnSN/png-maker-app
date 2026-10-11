import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session

app = FastAPI(title="PNG Maker API")

# Enable CORS for everyone (including GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use the lightweight model so Render's 512MB RAM doesn't run out of memory
# u2netp is the lightweight, fast version of u2net
session = new_session("u2netp")

@app.get("/")
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
            output_image = remove(image, session=session)

        output_bytes = io.BytesIO()
        output_image.save(output_bytes, format="PNG")
        output_bytes.seek(0)

        return Response(content=output_bytes.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))