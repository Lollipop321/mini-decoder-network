import torch
import argparse

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

    for key in model.keys():
        # print(key)
        if 'decoder' in key and 'fc' in key:
            continue
        else:
            model_new[key] = model[key]

    checkpoint_new['args'] = args
    checkpoint_new['args'].arch = "transformer_noffn_t2t_wmt_en_de"
    checkpoint_new['model'] = model_new

    torch.save(checkpoint_new, 'checkpoint_noffn.pt')

    print("finished!")