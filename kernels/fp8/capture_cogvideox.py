"""Capture Q/K/V tensors from a CogVideoX attention layer.

Subclasses CogVideoXAttnProcessor2_0 with one that snapshots Q,K,V (after RoPE,
i.e. exactly what the attention kernel sees) for a chosen transformer block,
then runs a single denoising step and saves the tensors to disk.

Usage:
    python3 capture_cogvideox.py \
        --model THUDM/CogVideoX-2b \
        --layer 0 \
        --out captures/cogvideox.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="THUDM/CogVideoX-2b")
    parser.add_argument("--prompt", default=
        "A cat sitting on a windowsill watching rain fall on a quiet street.")
    parser.add_argument("--layer", type=int, default=0,
                        help="Index of the transformer block whose attention to dump")
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--out", default="captures/cogvideox.pt")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Truncate Q/K/V along the token dim before saving")
    return parser.parse_args()


def make_capturing_processor(captured, max_tokens):
    from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0

    class CapturingProcessor(CogVideoXAttnProcessor2_0):
        def __call__(self, attn, hidden_states, encoder_hidden_states,
                     attention_mask=None, image_rotary_emb=None):
            text_seq_length = encoder_hidden_states.size(1)
            hs = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            batch_size, sequence_length, _ = hs.shape

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(
                    batch_size, attn.heads, -1, attention_mask.shape[-1])

            query = attn.to_q(hs)
            key = attn.to_k(hs)
            value = attn.to_v(hs)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query[:, :, text_seq_length:] = apply_rotary_emb(
                    query[:, :, text_seq_length:], image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(
                        key[:, :, text_seq_length:], image_rotary_emb)

            # Snapshot Q/K/V exactly as the kernel would see them.
            if "Q" not in captured:
                q_save, k_save, v_save = query, key, value
                if max_tokens is not None and q_save.shape[2] > max_tokens:
                    q_save = q_save[..., :max_tokens, :]
                    k_save = k_save[..., :max_tokens, :]
                    v_save = v_save[..., :max_tokens, :]
                captured["Q"] = q_save.detach().to("cpu", dtype=torch.float32)
                captured["K"] = k_save.detach().to("cpu", dtype=torch.float32)
                captured["V"] = v_save.detach().to("cpu", dtype=torch.float32)
                captured["text_seq_length"] = text_seq_length
                captured["heads"] = attn.heads
                captured["head_dim"] = head_dim

            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)
            return hidden_states, encoder_hidden_states

    return CapturingProcessor()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from diffusers import CogVideoXPipeline

    torch_dtype = getattr(torch, args.dtype)
    print(f"Loading {args.model}...")
    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)
    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    blocks = pipe.transformer.transformer_blocks
    if not 0 <= args.layer < len(blocks):
        raise IndexError(f"--layer must be in [0, {len(blocks)}), got {args.layer}")

    captured = {}
    blocks[args.layer].attn1.processor = make_capturing_processor(captured, args.max_tokens)

    print(f"Running {args.num_inference_steps} step(s) to capture layer {args.layer}...")
    with torch.inference_mode():
        pipe(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            guidance_scale=6.0,
        )

    if "Q" not in captured:
        raise RuntimeError("No Q/K/V captured. The hook never fired.")

    print(f"Captured Q={tuple(captured['Q'].shape)} "
          f"K={tuple(captured['K'].shape)} V={tuple(captured['V'].shape)}")
    torch.save(captured, out_path)
    print(f"Saved capture to {out_path}")


if __name__ == "__main__":
    main()
