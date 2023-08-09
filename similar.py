#
# Original authors Nyanko Lepsoni and RcINS. Danke schön
#
# MIT License
#
# ChangeLog:
# - support input_blocks similarity.
# - all in one script
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

def eval(model, block, n, input):
    if block in [ "input_blocks", "output_blocks" ]:
        keybase = f"model.diffusion_model.{block}.{n}"
    else:
        keybase = f"model.diffusion_model.{block}"
    qk = f"{keybase}.1.transformer_blocks.0.attn1.to_q.weight"
    uk = f"{keybase}.1.transformer_blocks.0.attn1.to_k.weight"
    vk = f"{keybase}.1.transformer_blocks.0.attn1.to_v.weight"
    atoq, atok, atov = model[qk], model[uk], model[vk]

    attn = cal_cross_attn(atoq, atok, atov, input)
    return attn

def get_block(blockname):
    if blockname[0:3] == "OUT":
        block = "output_blocks"
        n = blockname[3:].lstrip("0")
        n = int(n) if n else 0
    elif blockname[0:2] == "IN":
        block = "input_blocks"
        n = blockname[2:].lstrip("0")
        n = int(n) if n else 0
    else:
        block = "middle_block"
        n = 1

    return block, n


IN_BLOCKS = [ "IN01", "IN02", "IN04", "IN05", "IN07", "IN08" ];
MID_BLOCK = [ "MID00" ];
OUT_BLOCKS = [ "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11" ];

def main():
    seed = 114514

    # simple argv parser
    selected = []
    args = sys.argv[1:]
    remains = []
    for i, arg in enumerate(args):
        if arg is None:
            continue
        elif arg == "-a":
            selected.append("inputs")
            selected.append("middle")
            selected.append("outputs")
        elif arg == "-i":
            selected.append("inputs")
        elif arg == "-m":
            selected.append("middle")
        elif arg == "-o":
            selected.append("outputs")
        elif arg == "-s":
            try:
                seed = int(args[i+1])
                args[i+1] = None
            except ValueError:
                args[i+1] = None
                pass
        else:
            remains.append(arg)

    file1 = Path(remains[0])
    files = remains[1:]

    if len(files) == 0:
        print("Usage: python similar.py [-a] [-i] [-m] [-o] [-s seed] model_a model_b")
        exit(1)

    torch.manual_seed(seed)
    print(f"seed: {seed}") 

    # setup blocks
    blocks = []
    if "inputs" in selected:
        blocks = blocks + IN_BLOCKS
    if "middle" in selected:
        blocks = blocks + MID_BLOCK
    if "outputs" in selected:
        blocks = blocks + OUT_BLOCKS

    if len(blocks) == 0:
        blocks = IN_BLOCKS + MID_BLOCK + OUT_BLOCKS

    model_a = load_model(file1)
    
    print()
    print(f"base: {file1.name} [{model_hash(file1)}]")
    print()

    map_attn_a = {}
    map_rand_input = {}
    for b in blocks:
        block, n = get_block(b)
        if block in ["input_blocks", "output_blocks"]:
            hidden_dim, embed_dim = model_a[f"model.diffusion_model.{block}.{n}.1.transformer_blocks.0.attn1.to_q.weight"].shape
        else: # middle_block
            hidden_dim, embed_dim = model_a[f"model.diffusion_model.{block}.1.transformer_blocks.0.attn1.to_q.weight"].shape
        rand_input = torch.randn([embed_dim, hidden_dim])

        map_attn_a[b] = eval(model_a, block, n, rand_input)
        map_rand_input[b] = rand_input

    del model_a
     
    hdr = "| "
    for n in blocks:
        hdr += f"  {n.rjust(5, ' ')}   | "

    for file2 in files:
        print(hdr)
        val = "| "
        file2 = Path(file2)
        model_b = load_model(file2)
        
        sims = []
        for b in blocks:
            block, n = get_block(b)
            attn_a = map_attn_a[b]
            attn_b = eval(model_b, block, n, map_rand_input[b])
            
            sim = torch.mean(torch.cosine_similarity(attn_a, attn_b))
            sims.append(sim)
            val += f"{sim * 1e2:8.4f}% | "

        print(val)
        print("")
        print(f"{file2} [{model_hash(file2)}] - {torch.mean(torch.stack(sims)) * 1e2:8.4f}%")
        print("")
        
if __name__ == "__main__":
    main()
