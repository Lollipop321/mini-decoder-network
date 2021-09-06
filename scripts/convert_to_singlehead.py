import torch
import math
import time
import struct
import argparse
import numpy as np
from collections import OrderedDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True, help="trained model prefix, also include dir, e.g. ../data/model-100")

    args = parser.parse_args()

    model_path = args.model

    checkpoint = torch.load(model_path, map_location='cpu')
    assert 'args' in checkpoint
    assert 'model' in checkpoint
    args = checkpoint['args']
    model = checkpoint['model']

    checkpoint_new = {}
    model_new = {}

    e = [0, 0, 0, 0, 0, 0]
    d = [0, 0, 0, 0, 0, 0]

    for name, w in model.items():
        if "decoder" in name:
            if "self_attn.in_proj" in name:
                layer = eval(name.split(".")[2])
                wq, wk, wv = w.chunk(3, dim=0)
                assert args.encoder_embed_dim == args.decoder_embed_dim
                model_new[name] = torch.cat([wq[(args.encoder_embed_dim // 8 * e[layer]): (args.encoder_embed_dim // 8 * (e[layer] + 1))],
                                             wk[(args.encoder_embed_dim // 8 * e[layer]): (args.encoder_embed_dim // 8 * (e[layer] + 1))],
                                             wv[(args.encoder_embed_dim // 8 * e[layer]): (args.encoder_embed_dim // 8 * (e[layer] + 1))]], dim=0)
            elif "encoder_attn.in_proj" in name:
                layer = eval(name.split(".")[2])
                wq, wk, wv = w.chunk(3, dim=0)
                assert args.encoder_embed_dim == args.decoder_embed_dim
                model_new[name] = torch.cat([wq[(args.encoder_embed_dim // 8 * d[layer]): (args.encoder_embed_dim // 8 * (d[layer] + 1))],
                                             wk[(args.encoder_embed_dim // 8 * d[layer]): (args.encoder_embed_dim // 8 * (d[layer] + 1))],
                                             wv[(args.encoder_embed_dim // 8 * d[layer]): (args.encoder_embed_dim // 8 * (d[layer] + 1))]], dim=0)
            elif "self_attn.out_proj.weight" in name:
                layer = eval(name.split(".")[2])
                assert args.encoder_embed_dim == args.decoder_embed_dim
                model_new[name] = w[:, (args.encoder_embed_dim // 8 * e[layer]): (args.encoder_embed_dim // 8 * (e[layer] + 1))]
            elif "encoder_attn.out_proj.weight" in name:
                layer = eval(name.split(".")[2])
                assert args.encoder_embed_dim == args.decoder_embed_dim
                model_new[name] = w[:, (args.encoder_embed_dim // 8 * d[layer]): (args.encoder_embed_dim // 8 * (d[layer] + 1))]
            else:
                model_new[name] = w
        else:
            model_new[name] = w

    checkpoint_new['args'] = args
    checkpoint_new['args'].arch = "transformer_singlehead_t2t_wmt_en_de"
    checkpoint_new['model'] = model_new
    # print(checkpoint_new['args'].arch)

    torch.save(checkpoint_new, 'checkpoint_singlehead.pt')

    print("finished!")