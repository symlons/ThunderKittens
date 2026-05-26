import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from functools import partial
import argparse
from typing import cast

hidden_size = 1024
mlp_ratio = 4
mlp_hidden_dim = int(hidden_size * mlp_ratio)


h200 = False
batch_size, seqlen = 64, 2048
if h200: batch_size = batch_size * 2

input_tensor = torch.rand(batch_size, seqlen, hidden_size)
mlp = Mlp(
    in_features=hidden_size,
    hidden_features=mlp_hidden_dim,
    act_layer=cast(type[nn.GELU], partial(nn.GELU, approximate="tanh")),
    drop=0,
)

def parse_shape(text: str) -> tuple[int, int]:
    batch, tokens = text.lower().replace("b", "").replace("t", "").split("x")
    return int(batch), int(tokens)

def profile_fwd(batch, tokens, dim, warmup, iters, eps):
    with torch.no_grad(): # todo, inference mode
    for i in warmup:
        output = mlp(input_tensor)
        print(mlp)

    for i in iters:
        output = mlp(input_tensor)
        print(mlp)
    print("mlp output: ", output.shape)
    print("timing, tflops")
    print("correctness")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["64x1024", "80x1024", "16x4096", "20x4096"])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    for shape in args.shapes:
        batch, tokens = parse_shape(shape)
        profile_fwd(batch, tokens, args.dim, args.warmup, args.iters, args.eps)


if __name__ == "__main__":
    main()
