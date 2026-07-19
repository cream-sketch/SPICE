import sys
import torch

d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("router captures:", len(d["router_logits"]))
print("logits shape:", tuple(d["router_logits"][0].shape))
print("module[0]:", d["router_module_names"][0])
print("module[-1]:", d["router_module_names"][-1])
hs = d.get("hidden_states")
print("hidden_states:", len(hs) if hs else None, tuple(hs[0].shape) if hs else "")
print("input_ids:", tuple(d["input_ids"].shape))
