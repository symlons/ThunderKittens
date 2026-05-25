from __future__ import annotations


TOKEN_SHAPES = {
    "t256": ("4", "8", "8"),
    "t512": ("8", "8", "8"),
    "t1024": ("8", "8", "16"),
    "t2048": ("8", "16", "16"),
    "t4096": ("16", "16", "16"),
    "t8192": ("16", "16", "32"),
    "t16384": ("16", "32", "32"),
    "t32768": ("32", "32", "32"),
    "t60000": ("30", "40", "50"),
}

BATCH_PARTS = ("b1", "b2", "b4", "b8", "b16", "b32", "b64", "b128", "b256", "b512", "b1024")


def dit_command(
    *,
    model: str,
    tokens: list[int] | None = None,
    batches: list[int] | None = None,
    spatial: tuple[int, int, int] | None = None,
    sweep: bool = False,
    compile_model: bool = False,
    fa3: bool = False,
    probe_memory: bool = False,
    warmup: int = 5,
    iters: int = 5,
) -> list[str]:
    command = ["python3", "dit3d_e2e_bench.py", "--model", model.upper()]
    if sweep or tokens:
        command.append("--sweep")
        if tokens:
            command.extend(["--tokens", *(str(t) for t in tokens)])
    elif spatial:
        command.extend(["--spatial", *(str(v) for v in spatial)])
    if batches:
        command.extend(["--batches", *(str(b) for b in batches)])
    command.extend(["--warmup", str(warmup), "--iters", str(iters)])
    if compile_model:
        command.append("--compile")
    if fa3:
        command.append("--fa3")
    if probe_memory:
        command.append("--probe-memory")
    return command


def command_for_mode(mode: str) -> list[str]:
    if mode == "fused_input_check":
        return ["python3", "dit3d_e2e_bench.py", "--check-fused-input"]
    if mode == "residual":
        return [
            "python3", "dit3d_e2e_bench.py",
            "--bench-residual",
            "--tokens", "1024",
            "--batches", "2", "8",
            "--hidden-dim", "1024",
            "--warmup", "500",
            "--iters", "100",
        ]
    if mode.startswith("dit3d"):
        parts = mode.split("_")
        is_sweep = "sweep" in parts or "tokens" in parts
        model = parts[1].upper() if len(parts) > 1 else "S"
        command = ["python3", "dit3d_e2e_bench.py", "--model", model]
        if not is_sweep:
            for part in parts:
                if part in TOKEN_SHAPES:
                    command.extend(["--spatial", *TOKEN_SHAPES[part], "--batches", "1", "--warmup", "5", "--iters", "5"])
                    break
            if "fitb" in parts and "--batches" in command:
                batch_i = command.index("--batches")
                del command[batch_i:batch_i + 2]
                command.extend(["--batches", "1", "2", "4", "8", "16", "32", "64", "128"])
            for batch_part in BATCH_PARTS:
                if batch_part not in parts:
                    continue
                batch = batch_part[1:]
                if "--batches" in command:
                    command[command.index("--batches") + 1] = batch
                else:
                    command.extend(["--batches", batch])
        if "mem" in parts:
            command.append("--probe-memory")
        if is_sweep:
            batches = ["1"]
            warmup = "1"
            iters = "3"
            if "smallb" in parts:
                batches = ["1", "2", "4", "8"]
            elif "fitb" in parts:
                batches = ["1", "2", "4", "8", "16", "32", "64", "128"]
                warmup = "5"
                iters = "5"
            elif "highb" in parts:
                batches = ["256", "512", "1024"]
                warmup = "5"
                iters = "10"
            else:
                for batch_part in BATCH_PARTS:
                    if batch_part in parts:
                        batches = [batch_part[1:]]
                        break
            command.extend([
                "--sweep",
                "--tokens", "256", "512", "1024", "2048", "4096", "8192", "16384",
                "--batches", *batches,
                "--warmup", warmup,
                "--iters", iters,
            ])
        if "compile" in parts:
            command.append("--compile")
        if "fa3" in parts:
            command.append("--fa3")
        return command
    return ["python3", "harness.py", mode, "--report", "KERNEL_REPORT.md"]
