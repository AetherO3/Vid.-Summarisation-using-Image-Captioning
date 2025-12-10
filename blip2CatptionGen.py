import os
import glob
import cv2
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

dirPath = 'Frames'
framesPath = os.path.join(dirPath, "*.jpg")
imageFiles = glob.glob(framesPath)


# --- Check if the images have been successfully read----
if not imageFiles:
    print(f"Could not read any files at address {framesPath}")
    exit()

images = [cv2.imread(file) for file in imageFiles]
images = [img for img in images if img is not None]

if not images:
    print("Could not get the valid images")
    exit()

for i, image in enumerate(images):
    cv2.imshow('View', image) 
    key = cv2.waitKey(0)

cv2.destroyAllWindows()



processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for i, img in enumerate(images):
    print(f"Image {i+1} of {len(images)} ({os.path.basename(imageFiles[i])})")
    
    inputs = processor(img, return_tensors="pt").to(device, torch.float16) 
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print(f"Caption : {generated_text}\n")
