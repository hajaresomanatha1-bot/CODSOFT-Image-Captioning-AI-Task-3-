from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Upload image
from google.colab import files
uploaded = files.upload()

# Open image
image = Image.open(list(uploaded.keys())[0])

# Process image
inputs = processor(image, return_tensors="pt")

# Generate caption
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated Caption:", caption)
