import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from rembg import remove, new_session

app = FastAPI(title="PNG Maker App")
app.mount("/static", StaticFiles(directory="static"), name="static")

session = new_session("u2net")

def extract_signature(img: Image.Image) -> Image.Image:
    """
    Extracts ink strokes from paper with zero haloing.
    Preserves ink color while mapping paper brightness to transparency.
    """
    img = img.convert("RGBA")
    data = np.array(img, dtype=np.float32)

    r, g, b, _ = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Calculate luminance/brightness (grayscale)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    # Map paper brightness: white (255) -> 0 alpha (transparent)
    # Ink/dark strokes (0) -> 255 alpha (opaque)
    # Smooth cutoff to remove off-white paper texture/shadows
    alpha = 255.0 - luminance

    # Boost contrast so light paper becomes 100% transparent
    # Any paper with brightness > 220 becomes completely transparent
    paper_threshold = 215.0
    ink_threshold = 80.0

    alpha = np.clip((paper_threshold - luminance) / (paper_threshold - ink_threshold) * 255.0, 0, 255)

    data[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(data.astype(np.uint8), mode="RGBA")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/make-png")
async def make_png(
    file: UploadFile = File(...),
    mode: str = Form("auto")  # "auto", "general", or "signature"
):
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGBA")

    # If the user chooses signature mode, run the signature extraction algorithm
    if mode == "signature":
        output_image = extract_signature(input_image)
    else:
        # Default AI model for photos/people/pets
        output_image = remove(input_image, session=session)

    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return Response(
        content=output_buffer.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=cutout.png"}
    )