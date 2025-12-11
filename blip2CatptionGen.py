import os
import cv2
import glob
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Blip2ForConditionalGeneration

dirPath = 'Your Frames Directory'
framesPath = os.path.join(dirPath, "*.jpg")
imageFiles = sorted(glob.glob(framesPath))

# --- Getting the vid fps to then decide the timestamp acc to the respective frame no ---
cap = cv2.VideoCapture("Your Video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
   fps = 30 # --- incase it fails ---
frame_numbers = [int(os.path.basename(f).split('.')[0].split('_')[-1]) for f in imageFiles]
timestamps = [fn / fps for fn in frame_numbers]

# --- Check if the images have been successfully read ----
if not imageFiles:
    print(f"Could not read any files at address {framesPath}")
    exit()

images = [cv2.imread(file) for file in imageFiles]
images = [img for img in images if img is not None]

if not images:
    print("Could not get the valid images")
    exit()

# --- initialize the model and processor ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl") # --- for smaller GPU Salesforce/blip2-opt-2.7b ---
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16)

model.to(device)

captions = []

# --- run the caption generator the frames and store them ---
for i, img in enumerate(images):
    print(f"Image {i+1} of {len(images)} ({os.path.basename(imageFiles[i])})")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(img, return_tensors="pt").to(device, torch.float16) 
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print(f"Caption : {generated_text}\n")
    captions.append(generated_text)

# --- convert the captions into embeddings ---
sumModel = SentenceTransformer('all-mpnet-base-v2')
embeds = sumModel.encode(captions, normalize_embeddings=True)

# --- calculate the similarity score ---
simScore = [np.dot(embeds[i], embeds[i + 1]) for i in range(len(embeds) - 1)]

threshold = 0.60
clusters = []
current = [0]

# --- Clubbing together the clusters that have high similarity ---
for i, score in enumerate(simScore):
  if(score >= threshold):
    current.append(i + 1)
  else:
    clusters.append(current)
    current = [i + 1]

clusters.append(current)

# --- Printing the clusters with their respective timestamps ---
for idx, cluster in enumerate(clusters):
  start = cluster[0]
  end = cluster[-1]
  print(f"Between {timestamps[start]:.2f}s and {timestamps[end]:.2f}s")
  print(f"\t{captions[start]}")
  print(f"\n")