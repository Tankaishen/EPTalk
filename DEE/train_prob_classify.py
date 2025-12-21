import os
import torch
os.environ["WANDB__SERVICE_WAIT"] = "300"
import argparse
from config import get_args_parser
import models
import datasets
from utils.utils import seed_everything, generate_date_time
from utils.loss import ClosedFormSampledDistanceLoss, create_matching_matrix
from utils.prob_eval import compute_csd_sims,compute_matching_prob_sims
from utils.pcme import sample_gaussian_tensors
from torch.utils.data import DataLoader
import tqdm
import time
import json
import random
import glob
from torch.utils.tensorboard import SummaryWriter
from FER.get_model import init_affectnet_feature_extractor
import numpy as np
from sklearn.metrics import confusion_matrix
# import seaborn as sns
import matplotlib.pyplot as plt
import evaluation
# import visualize
import utils 
from transformers import Wav2Vec2Processor
from datetime import datetime
torch.cuda.empty_cache()

# this code is for only training probDEE
# speaker normalization is not allowed in this code
def list_to(list_,device):
    """move a list of tensors to device
    """
    for i in range(len(list_)):
        list_[i] = list_[i].to(device)
    return list_

def train_one_epoch(args, model, optimizer,scheduler, train_dataloader, device, processor, total_step, writer, 
                    affectnet_feature_extractor=None, epoch=None, log_wandb = True):
    cumulative_loss = 0
    step = 0
    criterion = model.criterion 
    model.train()
    model.to(device)
    total_steps = len(train_dataloader)
    loss_dict = {}
    for samples, labels in tqdm.tqdm(train_dataloader):
        '''
        samples : [audio_processed,expression_processed]
        labels : [emotion, intensity, gender, actor_id] ->[ int, int, int, int]
        '''
        labels = list_to(labels,device)

        audio_samples = processor(samples[0],sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
        expression_samples = samples[1].to(device)
        expression_feat = affectnet_feature_extractor.extract_feature_from_layer(expression_samples, layer_num = -2)
        audio_embedding, expression_embedding = model(audio_samples, expression_feat)
        # while setting up the matched matrix, various methods can be used
        # PCME++ used mixup augmentation and label smoothing but for now
        # we use a vanila match matrix 
        batch_size = len(audio_samples)
        # matched = create_matching_matrix(labels[0]).to(device)
        # matched = torch.eye(batch_size).to(device)
        # 
        loss_audio = criterion(audio_embedding, labels[0])
        loss_exp = criterion(expression_embedding,labels[0])
        loss_dict['loss_audio'] = loss_audio.item()
        loss_dict['loss_exp'] = loss_exp.item()
        loss = loss_audio + loss_exp
        if log_wandb:
            # wandb.log({k: (v.item() if hasattr(v, 'item') else v) for k, v in loss_dict.items()})
            # wandb.log({"learning rate": optimizer.param_groups[0]['lr']})
            for k,v in loss_dict.items():
                writer.add_scalar(f'train/{k} (step)',v , total_step + step)
            writer.add_scalar(f'train/learning rate',v , total_step + step)
        if step % 50 == 0:
            print(f"epoch {epoch}, step {step} loss_audio : {loss_audio} loss_exp: {loss_exp}")
            
        step+=1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        audio_samples = audio_samples.detach().cpu()
        expression_samples = expression_samples.detach().cpu()
        cumulative_loss += loss.item() # culminative loss is only for CLIP loss
        del samples
        del audio_samples
        del expression_samples
        torch.cuda.empty_cache()
            
    scheduler.step()
    batch_num_per_epoch = len(train_dataloader)
    print(f"Epoch {epoch} loss: {cumulative_loss/batch_num_per_epoch}") # train_dataloader
    total_step+=total_steps
    return total_step
    
@torch.no_grad()
def val_one_epoch(args, model, val_dataloader , device, processor, total_step, writer, affectnet_feature_extractor=None, epoch=None, log_wandb = True):
    cumulative_loss = 0
    step = 0
    model.eval()
    criterion = model.criterion
    emotion_list = []
    intensity_list = []
    gender_list = []
    actor_list = []
    total = 0
    correct_audio = 0
    correct_exp = 0
    totalsample = len(val_dataloader)
    loss_dict = {}
    loss_dict['loss_audio'] = 0
    loss_dict['loss_exp'] = 0
    for samples, labels_ in tqdm.tqdm(val_dataloader):
        '''
        samples : [audio_processed,expression_processed]
        labels : [emotion, intensity, gender, actor_id] ->[ int, int, int, int]
        '''

        labels_ = list_to(labels_, device)

        audio_samples = processor(samples[0],sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
        expression_samples = samples[1].to(device)
        expression_feat = affectnet_feature_extractor.extract_feature_from_layer(expression_samples, layer_num = -2)
        audio_embedding, expression_embedding = model(audio_samples, expression_feat)
        _, audio_output = torch.max(audio_embedding.data, 1)
        _, exp_output = torch.max(expression_embedding.data, 1)

        emotion, intensity, gender, actor_name = labels_
        labels = labels_[0]
        total += labels.size(0) # 获取总标签个数
        correct_audio += (audio_output == labels.long()).sum().item() # 音频分类正确个数
        correct_exp += (exp_output == labels.long()).sum().item() # 表情分类正确个数

        batch_size = len(audio_samples)
        # matched = torch.eye(batch_size).to(device)
        
        loss_audio = criterion(audio_embedding, labels)
        loss_exp = criterion(expression_embedding, labels)
        loss_dict['loss_audio'] = loss_audio.item()
        loss_dict['loss_exp'] = loss_exp.item()

        if step % 50 == 0:
            print(f"epoch {epoch}, step {step} loss_audio : {loss_audio} loss_exp: {loss_exp}")
            
        step+=1

        audio_samples = audio_samples.detach().cpu()
        expression_samples = expression_samples.detach().cpu()
        # cumulative_loss += loss.item() # culminative loss is only for CLIP loss
        del samples
        del audio_samples
        del expression_samples
        torch.cuda.empty_cache()

        if log_wandb:
            # wandb.log({"expression accuracy loss": exp_retrieve_accuracy,
            #             "audio accuracy loss": audio_retrieve_accuracy})
            # wandb.log({"val loss": cumulative_loss/batch_num_per_epoch})
            writer.add_scalar('val/loss_audio',loss_dict['loss_audio'] , total_step + step)
            writer.add_scalar('val/loss_exp',loss_dict['loss_exp'] , total_step + step)
        # writer.add_scalar('val/val loss', cumulative_loss/batch_num_per_epoch , epoch)
    writer.add_scalar('val/expression accuracy loss',correct_exp/total , epoch)
    writer.add_scalar('val/audio accuracy loss',correct_audio/total , epoch)
        # mean_acc = (audio_retrieve_accuracy + exp_retrieve_accuracy)/2.
    total_step +=totalsample
    return total_step

    
    
    
def main(args): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training start using {device}...")
    seed_everything(42)
    print("Loading models...")
    if args.affectnet_model_path:
        model_path = args.affectnet_model_path
        config_path = os.path.dirname(model_path) + '/config.yaml'
        _, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path, model_path)
        affectnet_feature_extractor.to(device)
        affectnet_feature_extractor.eval()
        affectnet_feature_extractor.requires_grad_(False)
    DEE = models.ProbDEE(args)
    DEE = DEE.to(device)
    processor = Wav2Vec2Processor.from_pretrained("/data/code/wav2vec2-large-xlsr-53-english")
    
    print("Loading dataset...")
    val_dataset = datasets.AudioExpressionDataset(args, dataset = 'MEAD', split = 'val')
    print("length of val dataset: ", len(val_dataset))
    start_time = time.time()
    train_dataset = datasets.AudioExpressionDataset(args, dataset=args.dataset, split = 'train') # debug
    print(f"Dataset loaded in {time.time() - start_time} seconds")
    print("length of train dataset: ", len(train_dataset))
    datum = train_dataset[0]
    print("audio slice shape: ", datum[0][0].shape)
    print("expression parameter slice shape:", datum[0][1].shape)
    # jb - batch size 10 -> changed to args.batch_size
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(DEE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    print("Training loop...")
    best_val_acc = 0
    log_path = args.save_dir.split('/')[-1] + "_ProbDEE_" + str(time.strftime("%m_%d_%H_%M", time.localtime()))
    writer = SummaryWriter(log_dir="log/{}".format(log_path))
    total_step=0
    total_step_val=0
    for epoch in range(1, args.epochs+1):
        training_time = time.time()
        total_step = train_one_epoch(args, DEE, optimizer,scheduler, train_dataloader, device,processor, 
                                     affectnet_feature_extractor=affectnet_feature_extractor,
                                     total_step=total_step, writer=writer, epoch=epoch, log_wandb = True)
        print('training time for this epoch :', time.time() - training_time)
        
        validation_time = time.time()
        total_step_val = val_one_epoch(args, DEE, val_dataloader, device,processor, affectnet_feature_extractor=affectnet_feature_extractor, total_step=total_step_val,
                      writer=writer, epoch=epoch, log_wandb = True)
        print('validation time for this epoch :', time.time() - validation_time)
    
        if epoch % args.val_freq == 0 :
            torch.save(DEE.state_dict(), f"{args.save_dir}/model_{epoch}.pt")
            print(f"Model saved at {args.save_dir}/model_{epoch}.pt")
            print("Traininig DONE!")

# get retrival accuraccy on the fly

def save_arguments_to_file(args, filename='arguments.json'):
    save_path = args.save_dir + '/' + filename
    with open(save_path, 'w') as file:
        json.dump(vars(args), file)

def json2args(json_path):
    with open(json_path) as f:
        args = argparse.Namespace(**json.load(f))
    return args

if __name__ == '__main__':

    # parser = argparse.ArgumentParser('train', parents=[get_args_parser()])
    # args = parser.parse_args()
    # time_stamp = generate_date_time()
    # args.save_dir = args.save_dir+ '_' +time_stamp
    json_path = "DEE/checkpoint/arguments_my.json"
    args = json2args(json_path)
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    # wandb.init(project = args.project_name,
    #            name = args.wandb_name,
    #            config = args)
    
    print(args.save_dir)
    save_arguments_to_file(args)
    main(args)

    print("DONE!")
