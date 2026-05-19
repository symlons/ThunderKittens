def forward_cases(quick):
    if quick:
        return [(1, 8, 1536, 128, False, 0), (1, 8, 1536, 64, False, 0)]
    cases = []
    for D in (64, 128):
        for N in (384, 768, 1536, 3072):
            for seed in (0, 1):
                cases.append((1, 8, N, D, False, seed))
    return cases + [(2, 16, 1536, 128, False, 0), (2, 16, 1536, 64, False, 0)]


def backward_ref_cases(quick):
    if quick:
        return [(1, 8, 1536, 128, False, 0)]
    return [(1, 8, 1536, 128, False, 0), (1, 8, 1536, 64, False, 0), (2, 16, 1536, 128, False, 1)]


def backward_kernel_cases(quick):
    if quick:
        return [(1, 8, 1536, 128, 0)]
    return [(1, 8, 1536, 128, 0), (1, 8, 1536, 64, 0), (2, 16, 1536, 128, 1)]

