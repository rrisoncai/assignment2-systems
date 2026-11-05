"""mixed_precision_accumulation and benchmarking_mixed_precision (a): run the handout snippets."""
import torch
import torch.nn as nn


def accumulation():
    print("== mixed_precision_accumulation")
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print("fp32 += fp32          :", s.item())

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp16 += fp16          :", s.item())

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("fp32 += fp16          :", s.item())

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print("fp32 += fp16->fp32    :", s.item())
    print("0.01 in fp16 is exactly", torch.tensor(0.01, dtype=torch.float16).item())


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("  fc1 output :", x.dtype)
        x = self.ln(x)
        print("  ln output  :", x.dtype)
        x = self.fc2(x)
        return x


def toy_dtypes(dtype):
    print(f"== ToyModel under autocast({dtype})")
    model = ToyModel(16, 4).cuda()
    x = torch.randn(8, 16, device="cuda")
    y = torch.randint(0, 4, (8,), device="cuda")
    with torch.autocast(device_type="cuda", dtype=dtype):
        print("  params     :", {p.dtype for p in model.parameters()})
        logits = model(x)
        print("  logits     :", logits.dtype)
        loss = nn.functional.cross_entropy(logits, y)
        print("  loss       :", loss.dtype)
    loss.backward()
    print("  grads      :", {p.grad.dtype for p in model.parameters()})


if __name__ == "__main__":
    accumulation()
    toy_dtypes(torch.float16)
    toy_dtypes(torch.bfloat16)
