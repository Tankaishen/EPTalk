import numpy as np
import argparse
import os
import torch
import time
from data_loader_MEAD import get_dataloaders
from model.EPTalk import EPTalk
from omegaconf import OmegaConf, DictConfig
from model.Classify import classifyNet
import random

def seed_everything(seed: int): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    # Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False causes 
    # cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    torch.backends.cudnn.benchmark = False # -> Might want to set this to True if it's too slow

def get_yaml_config(yaml_file):
    config = OmegaConf.load(yaml_file)
    return config

def init_affectnet_feature_extractor(config_path, model_path):
    cfg = get_yaml_config(config_path)
    model = classifyNet(cfg)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    from DEE.utils.utils import compare_checkpoint_model
    compare_checkpoint_model(checkpoint, model)
    return cfg, model

@torch.no_grad()
def test(args, model, test_loader, epoch):
    model.load_state_dict(
        torch.load(os.path.join(args.ckpt), map_location=torch.device('cpu')))
    model = model.to(args.device)
    model.eval()

    result_path = os.path.join(args.dataset, "test", args.result_path)
    result_path = result_path + '_' + str(args.feature_dim) + '_' + str(time.strftime("%m_%d_%H_%M", time.localtime()))
    os.makedirs(result_path, exist_ok=True)

    for audio, vertice, template, file_name, sids,labels, emo_level in test_loader:
        # to gpu
        audio, vertice, template, labels, emo_level = audio.to(args.device), vertice.to(args.device), template.to(args.device), labels.to(args.device), emo_level.to(args.device)
        prediction,vertice,_,_,_,_,_,_ = model(audio, template, vertice, labels)
        prediction = prediction.squeeze()  # (seq_len, V*3)
        prediction = prediction.reshape(prediction.shape[0], -1, 3)
        np.save(os.path.join(result_path, file_name[0].split(".")[0] + ".npy"), prediction.detach().cpu().numpy())

    # for audio, vertice, template, file_name, sids,_,_ in test_loader:
    #     # to gpu
    #     audio, vertice, template = audio.to(args.device), vertice.to(args.device), template.to(args.device)
    #     prediction = model.predict(audio, template)
    #     prediction = prediction.squeeze()  # (seq_len, V*3)
    #     prediction = prediction.reshape(prediction.shape[0], -1, 3)
    #     np.save(os.path.join(result_path, file_name[0].split(".")[0] + ".npy"), prediction.detach().cpu().numpy())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # VOCASET
    parser = argparse.ArgumentParser(
        description='EPTalk')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="MEAD", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023 * 3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=1024, help='512 for vocaset; 1024 for BIWI')
    parser.add_argument("--num_heads", type=int, default=4, help='transformers head')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="Audio_25", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices", help='path of the ground truth')
    parser.add_argument("--batch_size", type=str, default=1, help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=200, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:1", help='cuda or cpu')
    parser.add_argument("--last_train", type=int, default=0, help='last train')
    parser.add_argument("--load_path", type=str, default='save_temp', help='path to the trained models')
    parser.add_argument("--template_file", type=str, default="templates/template.npy", help='path of the personalized templates')
    parser.add_argument("--sampling_rate", type=int, default=16000, help='path of the personalized templates')
    parser.add_argument("--ckpt", type=str, default="ckpt/EPTalk.pth", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="test_result", help='path to the predictions')
    parser.add_argument("--stage", type=str, default="test", help='path to the predictions')
    parser.add_argument("--emo_full_face", type=bool, default=False, help='path to the predictions')
    parser.add_argument("--wav2vec_trainable", type=bool, default=False, help='path to the predictions')
    parser.add_argument("--wav2vec_with_processor", type=bool, default=False, help='path to the predictions')
    parser.add_argument("--wav2vec_expected_fps", type=int, default=50, help='path to the predictions')
    parser.add_argument("--wav2vec_target_fps", type=int, default=25, help='path to the predictions')
    parser.add_argument("--wav2vec_freeze", type=bool, default=True, help='path to the predictions')
    parser.add_argument("--wav2vec_dim", type=int, default=1024, help='path to the predictions')
    parser.add_argument("--wav2vec_model", type=str, default="/data/code/wav2vec2-large-xlsr-53-english", help='path to the predictions')
    parser.add_argument("--subspace", type=bool, default=True, help='path to the predictions')
    parser.add_argument("--gather", type=bool, default=True, help='path to the predictions')
    parser.add_argument("--model_path", type=str, default="FER/save_dir/base_BN_class_dropout0.3_leaky_weightloss_aug0.5_std0.022025-05-24-07-37-13/model_best.pth", help='loss function')
    parser.add_argument("--config_path", type=str, default="FER/save_dir/base_BN_class_dropout0.3_leaky_weightloss_aug0.5_std0.022025-05-24-07-37-13/config.yaml", help='loss function')
    parser.add_argument("--cosine_similarity", type=bool, default=True, help='path to the predictions')
    parser.add_argument("--workers", type=int, default=4, help='path to the predictions')
    args = parser.parse_args()

    seed_everything(42)
    # load FER
    model_path = args.model_path
    config_path = args.config_path
    _, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path, model_path)
    affectnet_feature_extractor.to(args.device)
    affectnet_feature_extractor.eval()
    affectnet_feature_extractor.requires_grad_(False)
    # build model
    model = EPTalk(args, FER=affectnet_feature_extractor)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)

    # load data
    dataset = get_dataloaders(args)
    test(args, model, dataset["test"], epoch=args.max_epoch)


if __name__ == "__main__":
    main()
