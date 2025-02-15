#%%
import torch

from optimum.quanto import freeze, qfloat8, quantize

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
from huggingface_hub import snapshot_download

dtype = torch.bfloat16

bfl_repo = "black-forest-labs/FLUX.1-dev"
revision = "main"
local_path = snapshot_download(repo_id="PrunaAI/FLUX.1-dev-8bit")

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
text_encoder_2 = torch.load(local_path + '/text_encoder_2.pt', weights_only=False)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer = torch.load(local_path + '/transformer.pt', weights_only=False)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
# pipe.enable_model_cpu_offload()
pipe.to('cuda')
print('done')
generator = torch.Generator().manual_seed(12345)
pipe(
"a cute apple smiling",
guidance_scale=0.0,
num_inference_steps=4,
max_sequence_length=256,
generator=torch.Generator("cpu").manual_seed(0)
  ).images[0]
#%%
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


#%%
from collections import OrderedDict
for m in pipe.transformer.modules():
    m._forward_hooks = OrderedDict()
pipe.transformer.transformer_blocks[18].register_forward_hook(lambda self, input, output: print(output[0].shape, output[1].shape))
# %%
height = 512
width = 512
prompt = "A cat holding a sign that says hello world"
latents = pipe(
    prompt,
    height=height,
    width=width,
    guidance_scale=3.5,
    num_inference_steps=1,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
    return_dict=False,
    callback_on_step_end=lambda *args, **kwargs: print(args, kwargs.keys()),
    output_type="latent"
)
# %%
latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
image = pipe.vae.decode(latents, return_dict=False)[0]
image = pipe.image_processor.postprocess(image, output_type="image")
