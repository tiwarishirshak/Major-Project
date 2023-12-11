from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import cv2
import os

app = FastAPI()
model = YOLO('yolov8l.yaml')




output_folder = "static/output_image"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    
    
    results_list = model(image)


    
    output_image_paths = []

    
    for idx, results in enumerate(results_list):
       
        output_image_path = os.path.join(output_folder, f"output_{idx}.jpg")
        cv2.imwrite(output_image_path, results.orig_img)

        
        output_image_paths.append(output_image_path)

    return {"output_image_paths": output_image_paths}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def main():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='localhost', port=8000)
