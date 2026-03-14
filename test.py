from pipeline.stablevsr_pipeline import StableVSRPipeline
from diffusers import DDPMScheduler, ControlNetModel
from accelerate.utils import set_seed
from PIL import Image
import os
import argparse
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pathlib import Path
import torch
from models.sft import SFT_Module
from pytorch_wavelets import DTCWTForward

def center_crop(im, size=128):
    width, height = im.size   # Get dimensions
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2
    return im.crop((left, top, right, bottom))

# get arguments
parser = argparse.ArgumentParser(description="Test code for StableVSR.")
parser.add_argument("--out_path", default='./StableVSR_results/', type=str, help="Path to output folder.")
parser.add_argument("--in_path", type=str, required=True, help="Path to input folder (containing sets of LR images).")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of sampling steps")
parser.add_argument("--controlnet_ckpt", type=str, default=None, help="Path to your folder with the controlnet checkpoint.")
parser.add_argument("--sft_ckpt", type=str, default=None, help="Path to trained sft_module.pth")
parser.add_argument("--part", type=int, default=1, help="Current part to process")
parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts to split the dataset")
args = parser.parse_args()

print("Run with arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")

# set parameters
set_seed(42)
device = torch.device('cuda')
model_id = 'claudiom4sir/StableVSR'
controlnet_model = ControlNetModel.from_pretrained(args.controlnet_ckpt if args.controlnet_ckpt is not None else model_id, subfolder='controlnet') # your own controlnet model
pipeline = StableVSRPipeline.from_pretrained(model_id, controlnet=controlnet_model)
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipeline.scheduler = scheduler
pipeline = pipeline.to(device)
pipeline.enable_xformers_memory_efficient_attention()
of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
of_model.requires_grad_(False)
of_model = of_model.to(device)

sft_module = SFT_Module(cond_channels=36, target_channels=320).to(device)
if args.sft_ckpt is not None:
    sft_module.load_state_dict(torch.load(args.sft_ckpt, map_location=device))
    print(f"✅ Loaded SFT Module weights from {args.sft_ckpt}")
else:
    print("⚠️ Warning: No SFT checkpoint provided! Using random weights.")
sft_module.eval()

dtcwt_xfm = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
dtcwt_xfm.requires_grad_(False)

# iterate for every video sequence in the input folder
seqs = sorted(os.listdir(args.in_path))
import math
chunk_size = math.ceil(len(seqs) / args.total_parts)
start_idx = (args.part - 1) * chunk_size
end_idx = min(start_idx + chunk_size, len(seqs))

assert 1 <= args.part <= args.total_parts, \
    f"--part must be between 1 and {args.total_parts}"
assert start_idx < len(seqs), \
    f"part {args.part} has no data (start_idx={start_idx} >= total={len(seqs)})"

seqs = seqs[start_idx:end_idx]
print(f"Processing part {args.part}/{args.total_parts}: {len(seqs)} sequences (index {start_idx} to {end_idx-1})")

for seq in seqs:
    frame_names = sorted(os.listdir(os.path.join(args.in_path, seq)))
    frames = []
    for frame_name in frame_names:
        frame = Path(os.path.join(args.in_path, seq, frame_name))
        frame = Image.open(frame)
        # frame = center_crop(frame)
        frames.append(frame)

    # upscale frames using StableVSR
    frames = pipeline('', frames, num_inference_steps=args.num_inference_steps, guidance_scale=0, of_model=of_model, sft_module=sft_module,dtcwt_xfm=dtcwt_xfm).images
    frames = [frame[0] for frame in frames]
    
    # save upscaled sequences
    seq = Path(seq)
    target_path = os.path.join(args.out_path, seq.parent.name, seq.name)
    os.makedirs(target_path, exist_ok=True)
    for frame, name in zip(frames, frame_names):
        frame.save(os.path.join(target_path, name))
