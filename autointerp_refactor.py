import asyncio
from matplotlib import pyplot as plt
from scored_storage import ScoredStorage
import litellm
import base64
from pathlib import Path
from itertools import islice
from litellm.caching import Cache
from litellm import completion, acompletion
from hashlib import md5

from PIL import Image
import io

import json
from asyncio import gather
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
import matplotlib
import numpy as np
from tqdm.auto import tqdm

from diffusers import AutoencoderKL
import torch

matplotlib.use("Agg")


class ActivationDescription(BaseModel):
    pattern_descriptions: list[str]
    common_pattern: str


class JudgeAnswer(BaseModel):
    score: float


class AutoInterp:
    """
    A class to analyze and visualize neuron activations in neural networks.
    """
    
    def __init__(self, 
                 data_path,
                 images_folder,
                 image_activations_folder,
                 interps_dir,
                 latent_cache_path,
                 counts_path,
                 n_features,
                 height=16, 
                 width=16, 
                 top_k_activations=1024,
                 activaton_threshold=5):
        """
        Initialize the AutoInterp with paths and configurations.
        """
        # Setup paths
        self.images_folder = Path(images_folder)
        self.image_activations_folder = Path(image_activations_folder)
        self.interps_dir = Path(interps_dir)
        self.latent_cache_path = Path(latent_cache_path)
        self.counts_path = counts_path
        
        # Create directories if they don't exist
        self.latent_cache_path.mkdir(exist_ok=True)
        self.interps_dir.mkdir(exist_ok=True)
        
        # Setup dimensions
        self.HEIGHT = height
        self.WIDTH = width

        self.activation_threshold = activaton_threshold
        self.n_features = n_features
        
        # Initialize storage
        self.scored_storage = ScoredStorage(
            Path(data_path),
            3,
            top_k_activations,
            mode="r",
            use_backup=True,
        )
        
        # Initialize VAE model
        self.ae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16, 
            subfolder="vae"
        )
        
        # NF4 quantization table
        self.nf4 = np.asarray([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ])
        
        # Templates for LLM interaction
        self.activations_template = """You will be given a list of images.
Each image will have activations for a specific neuron highlighted in blue.
You should describe a common pattern or feature that the neuron is capturing.
First, write for each image, which parts are higlighted by the neuron.
Then, write a common pattern or feature that the neuron is capturing.
"""
        
        self.judge_template = """You will be given an image. And a neuron's activations description.
The image will have activations for the neuron highlighted in blue.
You should judge whether the description of the neuron's pattern is accurate or not.
Return a score between 0 and 1, where 1 means the description is accurate and 0 means it is not.
Be very critical. The pattern should be literal and specific, and vague or general descriptions should be rated low.
The activation pattern is {pattern}.
"""
        
        # Setup LiteLLM
        self.setup_litellm()
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(3)

    def setup_litellm(self):
        """Configure LiteLLM with caching settings"""
        DISK_CACHE_DIR = Path(".dspy_cache")
        DISK_CACHE_LIMIT = 1e10  # 10 GB

        litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")

        if litellm.cache.cache.disk_cache.size_limit != DISK_CACHE_LIMIT:
            litellm.cache.cache.disk_cache.reset("size_limit", DISK_CACHE_LIMIT)

        litellm.telemetry = False
        litellm.suppress_debug_info = True

    def latent_to_img(self, latent):
        """Convert latent representation to image"""
        latent = latent[None]
        latent = np.stack((latent & 0x0F, (latent & 0xF0) >> 4), -1).reshape(
            *latent.shape[:-1], -1
        )
        latent = self.nf4[latent]
        latent = latent * 5.0

        image = self.ae.decode(
            z=torch.from_numpy(
                latent / self.ae.config.scaling_factor + self.ae.config.shift_factor
            ).to(torch.bfloat16)
        ).sample
        image_np = (
            ((image + 1) * 127)
            .clip(0, 255)
            .to(torch.uint8)
            .numpy()
            .squeeze()
            .transpose(1, 2, 0)
        )

        return image_np

    def path_to_img(self, path):
        """Load image from path with caching"""
        cached_npz_path = (
            self.latent_cache_path / md5(str(path).encode()).hexdigest()
        ).with_suffix(".npz")
        if cached_npz_path.exists():
            return np.load(cached_npz_path)["arr_0"]
        arr = self.latent_to_img(np.load(path)["arr_0"])
        np.savez_compressed(cached_npz_path, arr)
        return arr

    def encode_image(self, image_pil):
        """Encode PIL image to base64 string"""
        png_bytes = io.BytesIO()
        image_pil.save(png_bytes, format="PNG")
        return base64.b64encode(png_bytes.getvalue()).decode("utf-8")

    def maxacts(self, feature_id, random_order=False, only_counts=False):
        """Get images with maximum activations for a given feature"""
        rows = self.scored_storage.get_rows(feature_id)

        # Group rows by idx
        grouped_rows = {}
        for (idx, h, w), score in rows:
            key = idx
            if key not in grouped_rows:
                grouped_rows[key] = np.zeros((self.HEIGHT, self.WIDTH), dtype=float)

            # Add score to the corresponding location in the grid
            grouped_rows[key][h, w] = score

        grouped_rows = sorted(grouped_rows.items(), key=lambda x: x[1].max(), reverse=True)
        if random_order:
            np.random.shuffle(grouped_rows)
            
        for idx, grid in grouped_rows:
            try:
                full_activations = np.load(self.image_activations_folder / f"{idx}.npz")
                img_path = self.images_folder / f"{idx}.npz"
                if not img_path.exists():
                    continue
            except FileNotFoundError:
                continue
                
            gravel = grid.ravel()
            k = full_activations["arr_0"].shape[1]
            for i, (f, w) in enumerate(
                zip(full_activations["arr_0"].ravel(), full_activations["arr_1"].ravel())
            ):
                if f == feature_id:
                    gravel[i // k] = w
            if (gravel > self.activation_threshold).sum() < 6:
                continue

            if only_counts:
                yield 1
                continue

            img = self.path_to_img(img_path)
            # Normalize the grid for color intensity
            normalized_grid = (
                (grid - grid.min()) / (grid.max() - grid.min())
                if grid.max() > grid.min()
                else grid
            )

            fig, ax = plt.subplots()
            ax.imshow(np.asarray(img) / 255, extent=[0, 1, 0, 1])
            grid_overlay = ax.imshow(
                grid,
                alpha=np.sqrt(normalized_grid) * 0.7,
                extent=[0, 1, 0, 1],
                cmap="Blues",
            )
            ax.axis("off")
            fig.colorbar(grid_overlay, ax=ax)
            
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
            yield img
            plt.close("all")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_activation_descriptions(self, images):
        """Generate descriptions for neuron activations using LLM"""
        encoded_images = [self.encode_image(img) for img in images]

        messages = [
            {"role": "user", "content": [{"type": "text", "text": self.activations_template}]}
        ]
        
        for encoded_image in encoded_images:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        }
                    ],
                }
            )

        response = completion(
            model="openai/google/gemini-2.0-flash-001",
            messages=messages,
            base_url="https://openrouter.ai/api/v1",
            response_format=ActivationDescription,
        )

        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def call_judge(self, image, pattern):
        """Call LLM to judge the accuracy of activation descriptions"""
        encoded_image = self.encode_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.judge_template.format(pattern=pattern["common_pattern"])},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ]

        response = await acompletion(
            model="openai/google/gemini-2.0-flash-001",
            messages=messages,
            base_url="https://openrouter.ai/api/v1",
            response_format=JudgeAnswer,
        )

        return json.loads(response.choices[0].message.content)

    async def judge_activation_descriptions(self, images, pattern):
        """Judge all images against pattern descriptions"""
        calls = [self.call_judge(image, pattern) for image in images]
        return await gather(*calls)

    async def all_in_one_judge(self, feature_id, feature_image_counts, n_non_matching=5):
        """Full pipeline: get images, generate descriptions, judge them"""
        images_matching = list(islice(self.maxacts(feature_id, random_order=False), 0, 5))
        if not images_matching:
            return

        # Convert feature_image_counts to list of (id, count) tuples excluding current feature_id
        weighted_features = [
            (int(k), v) for k, v in enumerate(feature_image_counts) if int(k) != feature_id
        ]

        # Extract just the weights for sampling
        features, weights = zip(*weighted_features)

        # Sample other features using the weights
        sampled_features = np.random.choice(
            features, size=n_non_matching, p=np.array(weights) / sum(weights), replace=False
        )

        # Get images for the sampled features
        images_others = [
            next(iter(self.maxacts(f, random_order=False))) for f in sampled_features
        ]

        # Generate descriptions for the images
        try:
            pattern = self.generate_activation_descriptions(images_matching)
            pattern = json.loads(pattern)
        except Exception as e:
            print(f"Failed to generate descriptions for feature {feature_id}: {e}")
            return

        # Judge the descriptions
        labels = [1] * len(images_matching) + [0] * len(images_others)
        scores = await self.judge_activation_descriptions(
            images_matching + images_others, pattern
        )

        # Calculate the average score
        avg_score = sum(
            score["score"] if matching else 1 - score["score"]
            for matching, score in zip(labels, scores)
        ) / len(scores)

        return {"avg_score": avg_score, "pattern": pattern}

    async def add_one_result(self, feature_id, feature_image_counts):
        """Process one feature and save results"""
        async with self.semaphore:
            result = await self.all_in_one_judge(feature_id, feature_image_counts)
            if result:
                with open(f"{self.interps_dir}/{feature_id}.json", "w") as f:
                    json.dump(result, f)

    async def process_features(self, feature_image_counts, start_from=None):
        """Process all features starting from the given index"""
        # Determine the starting point
        if start_from is None:
            existing_features = list(self.interps_dir.glob("*.json"))
            start_from = max(int(f.stem) for f in existing_features) if existing_features else 0
            
        # Process all remaining features
        await asyncio.gather(
            *[
                self.add_one_result(i, feature_image_counts)
                for i in range(start_from, len(feature_image_counts))
            ]
        )

    def calculate_feature_image_counts(self, feature_id):
        """Calculate the number of images for each feature"""
        return sum(1 for _ in self.maxacts(feature_id, only_counts=True))

    async def calculate_feature_image_counts_async(self, feature_id, semaphore, pbar, results, save_lock):
        """Calculate image counts with progress tracking"""
        async with semaphore:
            count = sum(1 for _ in self.maxacts(feature_id, only_counts=True))
            results[feature_id] = count
            pbar.update(1)
            
            # Save intermediate results every 500 items
            async with save_lock:
                if pbar.n % 500 == 0 and pbar.n > 0:
                    tmp_counts = np.array([results.get(i, 0) for i in range(self.n_features)])
                    np.save(self.counts_path, tmp_counts)
                    pbar.set_description(f"Saved progress at {pbar.n}/{self.n_features}")
            
            return count

    async def calculate_all_feature_image_counts_async(self, n_threads=5):
        """Calculate the number of images for all features using async with progress bar"""
        semaphore = asyncio.Semaphore(n_threads)
        save_lock = asyncio.Lock()  # Lock for saving to prevent concurrent writes
        results = {}  # Dictionary to store results as they complete
        
        with tqdm(total=self.n_features, desc="Calculating feature image counts") as pbar:
            counts = await asyncio.gather(
                *[self.calculate_feature_image_counts_async(i, semaphore, pbar, results, save_lock) 
                for i in range(self.n_features)]
            )
        
        return counts

    def calculate_all_feature_image_counts(self, n_threads=5):
        """Calculate the number of images for all features with progress tracking and periodic saves"""
        counts = asyncio.run(self.calculate_all_feature_image_counts_async(n_threads))
        counts = np.array(counts)

        # Save the final counts to a file
        np.save(self.counts_path, counts)
        print(f"Saved final feature image counts to {self.counts_path}")
        return counts
    

# Example usage
if __name__ == "__main__":
    # This would be imported/defined elsewhere in the real code
    feature_type = "mlp" 

    data_path=f"{feature_type}_data/feature_acts.db"
    images_folder=f"images_{feature_type}"
    image_activations_folder=f"image_activations_{feature_type}"
    interps_dir=f"interps_{feature_type}"
    latent_cache_path=Path(".latent_cache_mlp")
    counts_path=f"image_counts_{feature_type}.npy"

    if feature_type == "sae":
        data_path = "flux1-saes/maxacts_double_l18_img"

    thresholds = {
        "mlp": 1,
        "sae": 5,
        "itda": 5,
    }

    n_features = {
        "mlp": 12288,
        "sae": 2**16,
        "itda": 16000,
    }

    auto_interp = AutoInterp(
        data_path=data_path,
        images_folder=images_folder,
        image_activations_folder=image_activations_folder,
        interps_dir=interps_dir,
        latent_cache_path=latent_cache_path,
        counts_path=counts_path,
        activaton_threshold=thresholds[feature_type],
        n_features=n_features[feature_type],
    )

    do_counts = False

    if do_counts:
        counts = auto_interp.calculate_all_feature_image_counts(n_threads=5)
    else:
        counts = np.load(counts_path)

    asyncio.run(auto_interp.process_features(counts))