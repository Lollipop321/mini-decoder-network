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
                if "decoder.layers.0" in name:
                    layer = eval(name.split(".")[2])
                    wq, wk, wv = w.chunk(3, dim=0)
                    assert args.encoder_embed_dim == args.decoder_embed_dim
                    model_new[name] = torch.cat([wq[(args.encoder_embed_dim // 8 * e[layer]): (args.encoder_embed_dim // 8 * (e[layer] + 1))],
                                                 wk[(args.encoder_embed_dim // 8 * e[layer]): (args.encoder_embed_dim // 8 * (e[layer] + 1))],
                                                 wv[(args.encoder_embed_dim // 8 * e[layer]): (args.encoder_embed_dim // 8 * (e[layer] + 1))]], dim=0)
                else:
                    continue
            elif "encoder_attn.in_proj" in name:
                if "decoder.layers.0" in name:
                    layer = eval(name.split(".")[2])
                    wq, wk, wv = w.chunk(3, dim=0)
                    assert args.encoder_embed_dim == args.decoder_embed_dim
                    model_new[name] = torch.cat([wq[(args.encoder_embed_dim // 8 * d[layer]): (args.encoder_embed_dim // 8 * (d[layer] + 1))],
                                                 wk[(args.encoder_embed_dim // 8 * d[layer]): (args.encoder_embed_dim // 8 * (d[layer] + 1))],
                                                 wv[(args.encoder_embed_dim // 8 * d[layer]): (args.encoder_embed_dim // 8 * (d[layer] + 1))]], dim=0)
                else:
                    continue
            elif "self_attn.out_proj.weight" in name:
                if "decoder.layers.0" in name:
                    layer = eval(name.split(".")[2])
                    assert args.encoder_embed_dim == args.decoder_embed_dim
                    model_new[name] = w[:, (args.encoder_embed_dim // 8 * e[layer]): (args.encoder_embed_dim // 8 * (e[layer] + 1))]
                else:
                    continue
            elif "self_attn.out_proj.bias" in name:
                if "decoder.layers.0" in name:
                    model_new[name] = w
                else:
                    continue
            elif "encoder_attn.out_proj.weight" in name:
                if "decoder.layers.0" in name:
                    layer = eval(name.split(".")[2])
                    assert args.encoder_embed_dim == args.decoder_embed_dim
                    model_new[name] = w[:, (args.encoder_embed_dim // 8 * d[layer]): (args.encoder_embed_dim // 8 * (d[layer] + 1))]
                else:
                    continue
            elif "encoder_attn.out_proj.bias" in name:
                if "decoder.layers.0" in name:
                    model_new[name] = w
                else:
                    continue
            elif "self_attn_layer_norm" in name:
                if "decoder.layers.0" in name:
                    model_new[name] = w
                else:
                    continue
            elif "encoder_attn_layer_norm" in name:
                if "decoder.layers.0" in name:
                    model_new[name] = w
                else:
                    continue
            elif "final_layer_norm" in name:
                if "decoder.layers.0" in name:
                    model_new[name] = w
                else:
                    continue
            elif "fc" in name:
                continue
            elif "embed_out" in name:
                continue
            else:
                model_new[name] = w
        elif "encoder" in name:
            if "layers." in name:
                for i in range(0, 4):
                    for j in range(0, 6):
                        if "layers."+str(j)+"." in name:
                            if i*6+j < 22:
                                name_new = name.replace("layers."+str(j)+".", "layers."+str(i*6+j)+".")
                                model_new[name_new] = w
            else:
                model_new[name] = w
        else:
            model_new[name] = w

    checkpoint_new['args'] = args
    checkpoint_new['args'].arch = "transformer_mdn"
    checkpoint_new['model'] = model_new

    torch.save(checkpoint_new, 'checkpoint_mdn.pt')

    print("finished!")