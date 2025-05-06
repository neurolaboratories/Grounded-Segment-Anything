import torch
from PIL import Image
import time
import requests
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

# Model setup
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
model.eval()

# Image and prompt
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
text_labels = [["a cat", "a remote control"]]
num_requests = 64
batch_size = 4
num_batches = num_requests // batch_size

# Pre-processed input for non-batched
inputs_single = processor(images=image, text=text_labels, return_tensors="pt").to(
    device
)

# Pre-processed input for batched
images_batched = [image] * batch_size
texts_batched = text_labels * batch_size
inputs_batched = processor(
    images=images_batched, text=texts_batched, return_tensors="pt"
).to(device)

# --------------------------
# Warm-up (non-batched)
# --------------------------
for _ in range(5):
    with torch.no_grad():
        _ = model(**inputs_single)

# --------------------------
# Benchmark NON-BATCHED
# --------------------------
torch.cuda.synchronize()
start = time.time()

for _ in tqdm(range(num_requests), desc="Non-Batched"):
    with torch.no_grad():
        _ = model(**inputs_single)

torch.cuda.synchronize()
end = time.time()
elapsed_single = end - start
print(
    f"\n[Non-Batched] Total: {elapsed_single:.2f}s | Avg: {elapsed_single / num_requests:.4f}s | Throughput: {num_requests / elapsed_single:.2f} img/s"
)

# --------------------------
# Warm-up (batched)
# --------------------------
for _ in range(3):
    with torch.no_grad():
        _ = model(**inputs_batched)

# --------------------------
# Benchmark BATCHED
# --------------------------
torch.cuda.synchronize()
start = time.time()

for _ in tqdm(range(num_batches), desc="Batched"):
    with torch.no_grad():
        _ = model(**inputs_batched)

torch.cuda.synchronize()
end = time.time()
elapsed_batch = end - start
print(
    f"\n[Batched x{batch_size}] Total: {elapsed_batch:.2f}s | Avg: {elapsed_batch / num_requests:.4f}s | Throughput: {num_requests / elapsed_batch:.2f} img/s"
)
