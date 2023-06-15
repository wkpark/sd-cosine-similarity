#
# Original authors Nyanko Lepsoni and RcINS. Danke schön
#
# MIT License
#
# support input_blocks similarity.
#
from safetensors.torch import load_file
import sys
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

def cal_cross_attn(to_q, to_k, to_v, rand_input):
    hidden_dim, embed_dim = to_q.shape
    attn_to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_q.load_state_dict({"weight": to_q})
    attn_to_k.load_state_dict({"weight": to_k})
    attn_to_v.load_state_dict({"weight": to_v})
    
    return torch.einsum(
        "ik, jk -> ik", 
        F.softmax(torch.einsum("ij, kj -> ik", attn_to_q(rand_input), attn_to_k(rand_input)), dim=-1),
        attn_to_v(rand_input)
    )

def model_hash(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'
        
def load_model(path):
    if path.suffix == ".safetensors":
        return load_file(path, device="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        
def eval(model, n, input):
    qk = f"model.diffusion_model.input_blocks.{n}.1.transformer_blocks.0.attn1.to_q.weight"
    uk = f"model.diffusion_model.input_blocks.{n}.1.transformer_blocks.0.attn1.to_k.weight"
    vk = f"model.diffusion_model.input_blocks.{n}.1.transformer_blocks.0.attn1.to_v.weight"
    atoq, atok, atov = model[qk], model[uk], model[vk]

    attn = cal_cross_attn(atoq, atok, atov, input)
    return attn

def eval_middle(model, input):
    qk = f"model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"
    uk = f"model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight"
    vk = f"model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight"
    atoq, atok, atov = model[qk], model[uk], model[vk]

    attn = cal_cross_attn(atoq, atok, atov, input)
    return attn

def main():
    file1 = Path(sys.argv[1])
    files = sys.argv[2:]
    
    seed = 114514
    torch.manual_seed(seed)
    print(f"seed: {seed}") 
    
    model_a = load_model(file1)
    
    print()
    print(f"base: {file1.name} [{model_hash(file1)}]")
    print()

    map_attn_a = {}
    map_rand_input = {}
    for n in 1,2,4,5,7,8:
        hidden_dim, embed_dim = model_a[f"model.diffusion_model.input_blocks.{n}.1.transformer_blocks.0.attn1.to_q.weight"].shape
        rand_input = torch.randn([embed_dim, hidden_dim])

        map_attn_a[n] = eval(model_a, n, rand_input)
        map_rand_input[n] = rand_input

    # for middle block
    hidden_dim, embed_dim = model_a[f"model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"].shape
    rand_input = torch.randn([embed_dim, hidden_dim])

    map_attn_a[0] = eval_middle(model_a, rand_input)
    map_rand_input[0] = rand_input

    del model_a
     
    hdr = "| "
    for n in 1,2,4,5,7,8:
        hdr += f"  IN{n:02d}   | "
    hdr += "  MI00   |"

    for file2 in files:
        print(hdr)
        val = "| "
        file2 = Path(file2)
        model_b = load_model(file2)
        
        sims = []
        for n in 1,2,4,5,7,8:
            attn_a = map_attn_a[n]
            attn_b = eval(model_b, n, map_rand_input[n])
            
            sim = torch.mean(torch.cosine_similarity(attn_a, attn_b))
            val += f"{sim * 1e2:.4f}% | "
            sims.append(sim)

        # for middle block
        attn_a = map_attn_a[0]
        attn_b = eval_middle(model_b, map_rand_input[0])
        sim = torch.mean(torch.cosine_similarity(attn_a, attn_b))
        val += f"{sim * 1e2:.4f}% |"
        sims.append(sim)

        print(val)
        print("")
        print(f"{file2} [{model_hash(file2)}] - {torch.mean(torch.stack(sims)) * 1e2:.4f}%")
        print("")
        
if __name__ == "__main__":
    main()