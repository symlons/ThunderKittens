from .metrics import fmt_forward, fmt_grad, tensor_metrics


def print_forward_comparison(label, out, ref):
    print(f"  {label:<12} {fmt_forward(tensor_metrics(out, ref))}")


def print_grad_comparison(label, got, ref):
    dQ, dK, dV = got
    rQ, rK, rV = ref
    print(" ", fmt_grad(label, tensor_metrics(dQ, rQ), tensor_metrics(dK, rK), tensor_metrics(dV, rV)))

