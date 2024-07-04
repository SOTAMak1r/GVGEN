import argparse
import json

def restore_string_to_list_in_a_dict(dictionary):
    for key in dictionary.keys():
        try:
            evaluated = eval(dictionary[key])
            if isinstance(evaluated, list):
                dictionary[key] = evaluated
        except:
            pass
        if isinstance(dictionary[key], dict):
            dictionary[key] = restore_string_to_list_in_a_dict(dictionary[key])
    return dictionary

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--recon_bg', type=str, default='white', help='recon bg color')
    parser.add_argument('--dm_bg',    type=str, default='white', help='dm bg color')

    parser.add_argument('--save_ply', type=bool, default=False, help='save point cloud ply')
    
    parser.add_argument('--dm_ckpt_path',    type=str, default='ckpts/text_dm.ckpt',      help='pretrained checkpoint path to load')
    parser.add_argument('--recon_ckpt_path', type=str, default='ckpts/text_recon.ckpt',   help='pretrained checkpoint path to load')
    parser.add_argument('--dm_cfg_path',     type=str, default='configs/text_dm.json',    help='configs for dm')
    parser.add_argument('--recon_cfg_path',  type=str, default='configs/text_recon.json', help='configs for recon')

    parser.add_argument('--text_input',  type=str, default='a green truck', help='text condition')
    parser.add_argument('--seed',  type=int, default=0)

    return parser.parse_args()

def get_cfgs(cfg_path):
    with open(cfg_path) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
    return config

if __name__ == '__main__':
    hparams = get_opts()
    print(type(hparams))
    print(hparams.cfg_path)
    cfg = get_cfgs(hparams.cfg_path)
    print(cfg)