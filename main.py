import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import torch
import json
import torch.nn as nn
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

def json2args(json_path):
    with open(json_path) as f:
        args = argparse.Namespace(**json.load(f))
    return args

def get_DEE_from_json(json_path):
    """_summary_
    input : json path
    output : DEE model
    """
    args = json2args(json_path)
    DEE = None
    if args.loss in ['soft_contrastive','csd']:
        print('using prob model')
        DEE = ProbDEE(args)
    return DEE,args

def get_yaml_config(yaml_file):
    config = OmegaConf.load(yaml_file)
    return config

def init_affectnet_feature_extractor(config_path, model_path):
    cfg = get_yaml_config(config_path)
    model = None
    if cfg.model.name == 'MLP':
        model = MLP(input_dim = cfg.model.input_dim, # if cfg.model.input_dim is 53, we are using jaw pose
                    layers = cfg.model.layers, 
                    output_dim = cfg.model.output_dim, # cfg.model.output_dim is same as label num
                    dropout = cfg.model.dropout, 
                    batch_norm = cfg.model.batch_norm, 
                    activation = cfg.model.activation)
    else: 
        model = classifyNet(cfg)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    from DEE.utils.utils import compare_checkpoint_model
    compare_checkpoint_model(checkpoint, model)
    return cfg, model

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch, last_train):
    save_path = os.path.join(args.dataset, "ckpt", args.save_path)
    save_path = save_path + '_' + str(args.feature_dim) + '_' + str(time.strftime("%m_%d_%H_%M", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    if last_train != 0:
        model.load_state_dict(torch.load(os.path.join(args.load_path, '{}_model.pth'.format(last_train)),
                                         map_location=torch.device('cpu')))
        model = model.to(args.device)
    writer = SummaryWriter(log_dir="log/{}".format(save_path.split('/')[-1]))
    processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_model)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.wav2vec_model)
    iteration = 0
    for e in range(last_train+1, epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()
        train_loss_log = {}
        train_loss_log["loss1"] = []
        train_loss_log["loss2"] = []
        train_loss_log["loss3"] = []
        train_loss_log["loss4"] = []
        train_loss_log["loss5"] = []
        train_loss_log["loss"] = []
        for i, (audio, vertice, template, file_name, sids, labels, emo_level) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, labels, emo_level = audio.to(args.device), vertice.to(args.device), template.to(args.device), labels.to(args.device), emo_level.to(args.device)
            # vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits, audio_loss, face_loss = model(audio, template,
            vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits, motion, motion_gt = model(audio, template,
                                                                                                vertice, labels)
            if(args.clsloss==False):
                loss5 = torch.tensor(0.0).to(args.device)
            else:
                feat, feat_gt, loss5= model.emo_loss(motion, motion_gt)
            # loss5 = criterion(feat, feat_gt)
            loss1 = criterion(vertice_out, vertice)
            gt_vel = vertice[:, 1:, :] - vertice[:, :-1, :]
            pred_vel = vertice_out[:, 1:, :] - vertice_out[:, :-1, :]
            loss2 = criterion(pred_vel, gt_vel)
            loss3 = criterion(lip_features, text_hidden_states)
            text_logits = torch.argmax(text_logits, dim=-1)
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            text_logits = processor.batch_decode(text_logits)
            text_logits = tokenizer(text_logits, return_tensors="pt", padding=True).input_ids
            text_logits = text_logits.to(args.device)

            batch_size = log_probs.shape[1]
            input_lengths = torch.full((batch_size,), fill_value=log_probs.shape[0], dtype=torch.long)  # 所有输入长度相同
            target_lengths = torch.sum(text_logits != tokenizer.pad_token_id, dim=1)  # 计算每个目标序列的实际长度（排除填充）
            loss4 = nn.functional.ctc_loss(
                log_probs,
                text_logits,
                # torch.tensor([log_probs.shape[0]]),
                # torch.tensor([text_logits.shape[1]]),
                input_lengths,
                target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
            
            # loss = torch.mean(1000 * loss1 + 1000 * loss2 + 0.001 * loss3 + 0.001 * loss4) # + 0.01*audio_loss + 0.01*face_loss)
            loss = torch.mean(1000 * loss1 + 1000 * loss2 + 0.001 * loss3 + 0.001 * loss4 + 0.00001 * loss5) # + 0.01*audio_loss + 0.01*face_loss)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}, loss1:{:.7f}, loss2:{:.7f}, loss3:{:.7f}, loss4:{:.7f}, loss5:{:.7f}".format(
                        e, iteration, loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()))
            train_loss_log["loss1"].append(loss1.item())
            train_loss_log["loss2"].append(loss2.item())
            train_loss_log["loss3"].append(loss3.item())
            train_loss_log["loss4"].append(loss4.item())
            train_loss_log["loss5"].append(loss5.item())
            train_loss_log["loss"].append(loss.item())
            # pbar.set_description(
            #     "(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}, loss1:{:.7f}, loss2:{:.7f}, loss3:{:.7f}, loss4:{:.7f}, audio_loss:{:.7f}, face_loss{:.7f}".format(
            #         e, iteration, loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(),audio_loss.item(),face_loss.item()))
        writer.add_scalar("train/loss1", loss1.item(), e)
        writer.add_scalar("train/loss2", loss2.item(), e)
        writer.add_scalar("train/loss3", loss3.item(), e)
        writer.add_scalar("train/loss4", loss4.item(), e)
        writer.add_scalar("train/loss5", loss5.item(), e)
        writer.add_scalar("train/loss", loss.item(), e)
        
        writer.add_scalar("train/loss1_mean", sum(train_loss_log["loss1"])/len(train_loss_log["loss1"]), e)
        writer.add_scalar("train/loss2_mean", sum(train_loss_log["loss2"])/len(train_loss_log["loss2"]), e)
        writer.add_scalar("train/loss3_mean", sum(train_loss_log["loss3"])/len(train_loss_log["loss3"]), e)
        writer.add_scalar("train/loss4_mean", sum(train_loss_log["loss4"])/len(train_loss_log["loss4"]), e)
        writer.add_scalar("train/loss5_mean", sum(train_loss_log["loss5"])/len(train_loss_log["loss5"]), e)
        writer.add_scalar("train/loss_mean", sum(train_loss_log["loss"])/len(train_loss_log["loss"]), e)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], e)



        # validation
        valid_loss_log = []
        model.eval()
        with torch.no_grad():

            for audio, vertice, template, file_name, sids, labels, emo_level in dev_loader:
                # to gpu
                audio, vertice, template, labels, emo_level = audio.to(args.device), vertice.to(args.device), template.to(args.device), labels.to(args.device), emo_level.to(args.device)
                train_subject = "_".join(file_name[0].split("_")[:-1])
                vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits, motion, motion_gt = model(audio, template,
                                                                                                            vertice,labels)
                feat, feat_gt, loss5 = model.emo_loss(motion, motion_gt)                                                                     
                loss = criterion(vertice_out, vertice)
                # loss5 = criterion(feat, feat_gt)
                valid_loss_log.append(loss.item())
            writer.add_scalar("val_loss/loss_mean", sum(valid_loss_log)/len(valid_loss_log), e)
            writer.add_scalar("val_loss/loss", loss.item(), e)
            writer.add_scalar("val_loss", loss.item(), e)

        current_loss = np.mean(valid_loss_log)

        if (e > 0 and e % 5 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.7f}".format(e + 1, current_loss))
        torch.cuda.empty_cache()
    writer.close()
    return model


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
    parser.add_argument("--num_layers", type=int, default=2, help='transformers head')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="Audio_25", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices", help='path of the ground truth')
    parser.add_argument("--batch_size", type=str, default=4, help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=105, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:6", help='cuda or cpu')
    parser.add_argument("--last_train", type=int, default=0, help='last train')
    parser.add_argument("--load_path", type=str, default='', help='path to the trained models')
    parser.add_argument("--template_file", type=str, default="template.npy", help='path of the personalized templates')
    parser.add_argument("--sampling_rate", type=int, default=16000, help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="EPTalk", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--stage", type=str, default="train", help='path to the predictions')
    parser.add_argument("--emo_full_face", type=bool, default=False, help='path to the predictions')
    parser.add_argument("--wav2vec_trainable", type=bool, default=False, help='path to the predictions')
    parser.add_argument("--wav2vec_with_processor", type=bool, default=False, help='path to the predictions')
    parser.add_argument("--wav2vec_expected_fps", type=int, default=50, help='path to the predictions')
    parser.add_argument("--wav2vec_target_fps", type=int, default=25, help='path to the predictions')
    parser.add_argument("--wav2vec_freeze", type=bool, default=True, help='path to the predictions')
    parser.add_argument("--wav2vec_dim", type=int, default=1024, help='path to the predictions')
    parser.add_argument("--wav2vec_model", type=str, default="/data/code/wav2vec2-large-xlsr-53-english", help='path to the predictions')
    parser.add_argument("--subspace", type=bool, default=True, help='path to the predictions')
    parser.add_argument("--cosine_similarity", type=bool, default=True, help='path to the predictions')
    parser.add_argument("--clsloss", type=bool, default=True, help='path to the predictions')
    # add args for DEE
    parser.add_argument("--model_path", type=str, default="FER/checkpoint/FER_model_best.pth", help='loss function')
    parser.add_argument("--config_path", type=str, default="FER/checkpoint/config.yaml", help='loss function')
    args = parser.parse_args()
    
    seed_everything(42)
    # load FER
    model_path = args.model_path
    config_path = args.config_path
    _, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path, model_path)
    affectnet_feature_extractor.to(args.device)
    affectnet_feature_extractor.eval()
    affectnet_feature_extractor.requires_grad_(False)

    model = EPTalk(args, FER=affectnet_feature_extractor)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)

    # load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    trainer(args, dataset["train"], dataset["val"], model, optimizer, criterion, epoch=args.max_epoch,
            last_train=args.last_train)


if __name__ == "__main__":
    main()