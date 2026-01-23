import argparse
import os
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL

from train import OUTPUT_MAX_LEN
from unet import UNetModel


class VaeDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, timesteps, context, y):
        return self.unet(x, None, timesteps, context, y)


def build_unet(args):
    model_args = SimpleNamespace(device=args.device, interpolation=False)
    return UNetModel(
        image_size=args.img_size,
        in_channels=4,
        model_channels=args.emb_dim,
        out_channels=4,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        context_dim=args.emb_dim,
        vocab_size=args.vocab_size,
        args=model_args,
        max_seq_len=args.max_seq_len,
    )


def export_unet(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    unet = build_unet(args).to(device)
    unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    unet.eval()
    wrapper = UNetWrapper(unet).to(device)
    wrapper.eval()

    batch = 1
    latent_h = args.img_size[0] // 8
    latent_w = args.img_size[1] // 8
    x = torch.randn(batch, 4, latent_h, latent_w, device=device, dtype=torch.float32)
    timesteps = torch.tensor([1], device=device, dtype=torch.int64)
    context = torch.zeros(batch, args.max_seq_len, device=device, dtype=torch.int64)
    labels = torch.zeros(batch, device=device, dtype=torch.int64)

    onnx_path = os.path.join(args.output_dir, "wordstylist_unet.onnx")
    torch.onnx.export(
        wrapper,
        (x, timesteps, context, labels),
        onnx_path,
        input_names=["x", "timesteps", "context", "y"],
        output_names=["predicted_noise"],
        dynamic_axes={
            "x": {0: "batch"},
            "timesteps": {0: "batch"},
            "context": {0: "batch"},
            "y": {0: "batch"},
            "predicted_noise": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Saved UNet ONNX to {onnx_path}")


def export_vae(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.eval()

    decoder = VaeDecoderWrapper(vae).to(device)
    decoder.eval()

    batch = 1
    latent_h = args.img_size[0] // 8
    latent_w = args.img_size[1] // 8
    latents = torch.randn(batch, 4, latent_h, latent_w, device=device, dtype=torch.float32)

    onnx_path = os.path.join(args.output_dir, "wordstylist_vae_decoder.onnx")
    torch.onnx.export(
        decoder,
        (latents,),
        onnx_path,
        input_names=["latents"],
        output_names=["images"],
        dynamic_axes={
            "latents": {0: "batch"},
            "images": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Saved VAE decoder ONNX to {onnx_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export WordStylist models to ONNX.")
    parser.add_argument("--output-dir", type=str, default="./onnx")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--img-size", type=int, nargs=2, default=(64, 256))
    parser.add_argument("--emb-dim", type=int, default=320)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=339)
    parser.add_argument("--vocab-size", type=int, default=53)
    parser.add_argument("--max-seq-len", type=int, default=OUTPUT_MAX_LEN)
    parser.add_argument("--unet-ckpt", type=str, required=True)
    parser.add_argument("--stable-dif-path", type=str, default="./stable-diffusion-v1-5")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-vae", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    export_unet(args)
    if not args.skip_vae:
        export_vae(args)


if __name__ == "__main__":
    main()
