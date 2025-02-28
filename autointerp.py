#%%
import litellm
import os
import base64
from pathlib import Path
from itertools import islice
from litellm.caching import Cache
from litellm import completion, acompletion
from PIL import Image

import json
from asyncio import gather
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
import matplotlib


from diffusers import AutoencoderKL
import torch

matplotlib.use("Agg")

ae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, subfolder="vae")

import numpy as np
nf4 = np.asarray(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)

def latent_to_img(latent: np.ndarray) -> np.ndarray:
    latent = latent[None]
    latent = np.stack((latent & 0x0F, (latent & 0xF0) >> 4), -1).reshape(*latent.shape[:-1], -1)
    latent = nf4[latent]
    latent = latent * 5.0

    image = ae.decode(z=torch.from_numpy(latent / ae.config.scaling_factor + ae.config.shift_factor).to(torch.bfloat16)).sample
    image_np = ((image + 1) * 127).clip(0, 255).to(torch.uint8).numpy().squeeze().transpose(1, 2, 0)

    return image_np

from hashlib import md5
latent_cache_path = Path("latent_cache")
latent_cache_path.mkdir(exist_ok=True)
def path_to_img(path: str) -> np.ndarray:
    cached_npz_path = (latent_cache_path / md5(str(path).encode()).hexdigest()).with_suffix(".npz")
    if cached_npz_path.exists():
        return np.load(cached_npz_path)["arr_0"]
    arr = latent_to_img(np.load(path)["arr_0"])
    np.savez_compressed(cached_npz_path, arr)
    return arr


DISK_CACHE_DIR = Path(".dspy_cache")
DISK_CACHE_LIMIT = 1e10 # 10 GB

litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")

if litellm.cache.cache.disk_cache.size_limit != DISK_CACHE_LIMIT:
    litellm.cache.cache.disk_cache.reset("size_limit", DISK_CACHE_LIMIT)

litellm.telemetry = False

# Turn off by default to avoid LiteLLM logging during every LM call.
litellm.suppress_debug_info = True
# maxacts_name = "maxacts_double_l18_img"
maxacts_name = "maxacts_double_l18_img"
#%%
from PIL import Image
import io

def encode_image(image_pil: Image):
    png_bytes = io.BytesIO()
    image_pil.save(png_bytes, format="PNG")
    return base64.b64encode(png_bytes.getvalue()).decode("utf-8")


def load_images():
    from huggingface_hub import HfApi
    api = HfApi()

    images_zip = api.hf_hub_download("nev/flux1-saes", filename=f"{maxacts_name}/images.zip")
    image_activations_zip = api.hf_hub_download("nev/flux1-saes", filename=f"{maxacts_name}/image_activations.zip")

    
    import zipfile

    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
        zip_ref.extractall("images_saes")

    
    with zipfile.ZipFile(image_activations_zip, 'r') as zip_ref:
        zip_ref.extractall("image_activations_saes")


maxacts_folder = Path("flux1-saes") / maxacts_name


images_folder = Path("images_saes")
image_activations_folder = Path("image_activations_saes")


img_path = images_folder / "51.npz"

img = np.load(img_path, allow_pickle=True)["arr_0"]

# print(list(maxacts_folder.glob("*")))
# print(os.getcwd())

acts_bd = np.load(maxacts_folder / "feature_acts.db.npy")


from scored_storage import ScoredStorage

top_k_activations = 1024
scored_storage = ScoredStorage(
    maxacts_folder / "feature_acts.db",
    3, top_k_activations,
    mode="r", use_backup=True
)


from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
image_activations_dir = image_activations_folder
HEIGHT, WIDTH = 16, 16

def maxacts(feature_id: int, random_order=False):
    rows = scored_storage.get_rows(feature_id)

    # Group rows by idx
    grouped_rows = {}
    for (idx, h, w), score in rows:
        key = idx
        if key not in grouped_rows:
            grouped_rows[key] = np.zeros((HEIGHT, WIDTH), dtype=float)

        # Add score to the corresponding location in the grid
        grouped_rows[key][h, w] = score
 
    if random_order:
        grouped_rows = sorted(grouped_rows.items(), key=lambda x: x[1].max(), reverse=True)
    for idx, grid in grouped_rows:
        try:
            full_activations = np.load(image_activations_dir / f"{idx}.npz")
            img_path = images_folder / f"{idx}.npz"
            if not img_path.exists():
                continue
        except FileNotFoundError:
            continue
        gravel = grid.ravel()
        k = full_activations["arr_0"].shape[1]
        for i, (f, w) in enumerate(zip(full_activations["arr_0"].ravel(), full_activations["arr_1"].ravel())):
            if f == feature_id:
                gravel[i // k] = w
        if (gravel > 5).sum() < 6:
            continue

        img = path_to_img(img_path)
        # Normalize the grid for color intensity
        normalized_grid = (grid - grid.min()) / (grid.max() - grid.min()) if grid.max() > grid.min() else grid

        fig, ax = plt.subplots()
        ax.imshow(np.asarray(img) / 255, extent=[0, 1, 0, 1])
        grid = ax.imshow(grid, alpha=np.sqrt(normalized_grid) * 0.7, extent=[0, 1, 0, 1], cmap="Blues")
        ax.axis("off")
        fig.colorbar(grid, ax=ax)
        # plt.imshow(normalized_grid, cmap=cmap)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        yield img
        plt.close("all")

activations_template = """You will be given a list of images.
Each image will have activations for a specific neuron highlighted in blue.
You should describe a common pattern or feature that the neuron is capturing.
First, write for each image, which parts are higlighted by the neuron.
Then, write a common pattern or feature that the neuron is capturing.
"""

class ActivationDescription(BaseModel):
    pattern_descriptions: list[str]
    common_pattern: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_activation_descriptions(images: list) -> str:
    # pil_images = [Image.fromarray(latent_to_img(img)) for img in images]
    pil_images = images
    encoded_images = [encode_image(pil_img) for pil_img in pil_images]

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": activations_template
        }]
    }]
    for encoded_image in encoded_images:
        messages.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }]
        })

    response = completion(
        model="openai/google/gemini-2.0-flash-001",
        messages=messages,
        base_url="https://openrouter.ai/api/v1",
        response_format=ActivationDescription
    )

    return response.choices[0].message.content


judge_template = """You will be given an image. And a neuron's activations description.
The image will have activations for the neuron highlighted in blue.
You should judge whether the description of the neuron's pattern is accurate or not.
Return a score between 0 and 1, where 1 means the description is accurate and 0 means it is not.
Be very critical. The pattern should be literal and specific, and vague or general descriptions should be rated low.
"""

class JudgeAnswer(BaseModel):
    score: float
    # justification: str

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_judge(image, pattern: list) -> dict:
    encoded_image = encode_image(image)

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": judge_template
        }, {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        }]
    }]

    for p in pattern:
        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": p
            }]
        })

    response = await acompletion(
        model="openai/google/gemini-2.0-flash-001",
        messages=messages,
        base_url="https://openrouter.ai/api/v1",
        response_format=JudgeAnswer
    )

    return json.loads(response.choices[0].message.content)

async def judge_activation_descriptions(images: list, pattern: str) -> list:
    calls = [call_judge(image, pattern) for image in images]

    calls = await gather(*calls)

    return calls


with open("results.json") as f:
    feature_image_counts = json.load(f)



async def all_in_one_judge(feature_id: int, n_non_matching=5):
    images_matching = list(islice(maxacts(feature_id, random_order=True), 0, 5))
    if not images_matching:
        return
    
    # Convert feature_image_counts to list of (id, count) tuples excluding current feature_id
    weighted_features = [(int(k), v) for k,v in enumerate(feature_image_counts) if int(k) != feature_id]
    
    # Extract just the weights for sampling
    features, weights = zip(*weighted_features)
    
    # Sample 5 other features using the weights
    sampled_features = np.random.choice(features, size=n_non_matching, p=np.array(weights)/sum(weights), replace=False)

    # Get images for the sampled features
    images_others = [next(iter(maxacts(f, random_order=True))) for f in sampled_features]

    # Generate descriptions for the images
    pattern = generate_activation_descriptions(images_matching)
    # print("Pattern", pattern)

    # Judge the descriptions
    labels = [1] * len(images_matching) + [0] * len(images_others)
    scores = await judge_activation_descriptions(images_matching + images_others, pattern)

    # Calculate the average score
    print(scores)
    avg_score = sum(score["score"] if matching else 1 - score["score"] for matching, score in zip(labels, scores)) / len(scores)

    return {
        "avg_score": avg_score, 
        "pattern": json.loads(pattern)
    }

#%%
# %%
import asyncio
interps_dir = "interps"
if not os.path.exists(interps_dir):
    os.makedirs(interps_dir)

existing_features = list(Path(interps_dir).glob("*.json"))

last_feature_processed = max(int(f.stem) for f in existing_features) if existing_features else 0

print(last_feature_processed)

semaphor = asyncio.Semaphore(5)
async def add_one_result(feature_id: int):
    async with semaphor:
        result = await all_in_one_judge(feature_id)
        if result:
            with open(f"{interps_dir}/{feature_id}.json", "w") as f:
                json.dump(result, f)

async def main():
    await asyncio.gather(*[add_one_result(i) for i in range(last_feature_processed, len(feature_image_counts))])
asyncio.run(main())
# %%
