import argparse
import os
def get_args_parser():
    parser = argparse.ArgumentParser('train DEE', add_help=False)

    parser.add_argument('--dataset', default='MEAD', type=str, help='choose dataset', choices=['RAVDESS', 'MEAD', 'CELEBV'])
    parser.add_argument('--device', default='cuda', type=str, help='choose dataset', choices=['RAVDESS', 'MEAD', 'CELEBV'])
    parser.add_argument('--normalize_exp', type=bool, default=False, help='normalize expression')
    parser.add_argument('--normalize_affectnet_features', action='store_true', help='normalize affectnet features')
    parser.add_argument('--random_slice', type=bool, default=True, help='randomly slice audio and expression')
    parser.add_argument('--disable_padding',type=bool, default=False, help='disable padding for audio and expression')
    # directory for MEAD and RAVDESS should have exactly same structure except for the name of the dataset
    # ex) '.../RAVDESS/Audio_samples' and '.../MEAD/Audio_samples'
    parser.add_argument('--audio_feature_dir', type=str, default="MEAD/Audio_25",help='audio feature directory')  
    parser.add_argument('--expression_feature_dir', type=str,default="MEAD/flame_params", help='expression feature directory')
    parser.add_argument('--use_30fps', type=bool, default=False, help='use 30fps for expression')
    parser.add_argument('--smooth_expression', type=bool, default=False, help='use savgol filter')
    # 1600 audio samples -> 0.1s
    # 3 frame -> 0.1s
    parser.add_argument('--audio_feature_len', default=32000, type=int, help='max sequence length of audio encoder') 
    parser.add_argument('--expression_feature_len', default=60, type=int, help='max sequence length of expression encoder')
    # if clip_length is 0, then do not clip
    # if clip_length is 1, then clip 0.1 seconds from front and back
    # if clip_length is 2, then clip 0.2 seconds from front and back
    parser.add_argument('--clip_length', default=5, type=int ,help='clip videos for clip_length* 0.1 seconds (front and back)')
    parser.add_argument('--stride_length', default=5, type=int ,help='stride length -> only works for trainset, valset stride is fixed to 0.2 seconds')
    # for training
    parser.add_argument('--batch_size', default=64, type=int, help='batch size for training, official=1024')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs for training')
    parser.add_argument('--scheduler', default='cosine', type=str, help='scheduler for learning rate')
    parser.add_argument('--val_freq', default=2, type=int, help='validation frequency')
    parser.add_argument('--save_dir', default='./checkpoint/ProbDEE', type=str, help='directory to save model')
    parser.add_argument('--project_name', type=str, default="ProbDEE_train", help='project name for wandb')    
    parser.add_argument('--wandb_name', default='test', type=str, help='wnadb run name')
    parser.add_argument('--loss', default='csd', type=str, help='which loss', choices=['infoNCE', 'RECO', 'emotion_guided_loss_gt', 'CLIP_loss_with_expression_guide', 'csd', 'soft_contrastive'])
    parser.add_argument('--gt_guide_weight', default=0.1, type=float, help='weight for ground truth guide loss')
    parser.add_argument('--exp_guide_weight', default=0.1, type=float, help='weight for expression guide loss')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for clip loss')
    parser.add_argument('--temperature_fix', type=bool, default=False, help='whether to use fixed temperature or not')
    parser.add_argument('--param_initialize', type=bool, default=False, help='whether to initialize parameters or not')
    parser.add_argument('--SID_first',type=bool, default=False, help='train SID first for few epochs and train with whole DEE later')
    parser.add_argument('--SID_epochs', default=0, type=int, help='epoch for SID_first')
    parser.add_argument('--gender_norm', type=bool, default=False, help='for SID, choose between normalizing gender or actor')
    parser.add_argument('--SID_lambda', default=1., type=float, help='lambda for SID loss')
    parser.add_argument('--use_embeddings', type=bool, default=True, help='use only wav2vec2 embeddings')
    parser.add_argument('--freeze_audio_encoder', type=bool, default=False, help='freeze audio transformer based encoder')
    parser.add_argument('--process_type', default='layer_norm', type=str, help='process type', choices=['wav2vec2', 'layer_norm'])
    # for model
    parser.add_argument('--num_parameter_heads', default=8, type=int, help='number of heads for parameter transformer')
    parser.add_argument('--num_audio_heads', default=8, type=int, help='number of heads for audio transformer')
    parser.add_argument('--num_audio_layers', default=2, type=int, help='number of audio encoder layer')
    parser.add_argument('--num_parameter_layers', default=6, type=int, help='number of parameter encoder layer')
    parser.add_argument('--feature_dim', default=128, type=int, help='output feature dimension for both audio and parameter')
    parser.add_argument('--parameter_dim', default=128, type=int, help='dimension of input parameter')
    parser.add_argument('--parameter_feature_dim', type=int, default=512, help='dimension of input parameter for encoder')
    parser.add_argument('--pos_encoding', default = 'sinusoidal', type=str, help='positional encoding for encoders', choices=['alibi', 'sinusoidal'])
    parser.add_argument('--max_seq_len', default=10, type=int, help='maximum sequence length for positional encoding -> this is in seconds')
    # if max_seq_len is 10-> then the max input sequenc is 10 seconds. -> this is set becasue the frequency of audio and exp is different
    # the audio feature is 50Hz and exp is 30Hz
    parser.add_argument('--use_audio_grl', type=bool, default=False, help='use gradient reversal layer for audio encoder')
    parser.add_argument('--use_exp_grl', type=bool, default=False, help='use gradient reversal layer for parameter encoder')
    
    parser.add_argument('--use_SER_encoder', type=bool, default=False, help='initialize wav2vec with emotion recognition model')
    parser.add_argument('--use_speaker_norm', type=bool, default=False, help='use speaker ID normalization')
    parser.add_argument('--freeze_FE', type=bool, default=False, help='freeze feature extractor of wav2vec')
    parser.add_argument('--use_emo2vec', type=bool, default=True, help='use DEE v2')
    # use_emo2vec allows us to use DEEv2m but for probDEE, we always use emo2vec
    parser.add_argument('--unfreeze_emo2vec', type=bool, default=True, help='unfreeze emo2vec')
    parser.add_argument('--use_finetuned_emo2vec', type=bool, default=True, help='use finetuned emo2vec')
    parser.add_argument('--unfreeze_block_layer', default=6, type=int, help='unfreeze block layer')
    # pooling strategy for model !!!!!READ 'help' carefully!!!!!!
    parser.add_argument('--exp_pool', default='gpo', type=str, help='pooling strategy for expression', choices=['cls', 'max', 'avg', 'gpo'])
    parser.add_argument('--audio_pool', default='gpo', type=str, help='pooling strategy for audio', choices=['cls', 'max', 'avg', 'gpo'])
    # parser.add_argument('--no_cls', action='store_true', help='not using cls token for audio embedding')
    # parser.add_argument('--max_pool', action='store_true', help='max pool for both audio and expression')
    # parser.add_argument('--gpo', action='store_true', help='use generalized pooling operatot for both audio and expression')
    # parser.add_argument('--avg_pool', action='store_true', help='average pool for both audio and expression')
    parser.add_argument('--no_audio_PE', type=bool, default=False, help='not using positional encoding for audio')
    parser.add_argument('--affectnet_model_path', default="FER/checkpoint/FER.pth", type=str, help='model path for affectnet feature extractor')
    # for evaluation.py
    parser.add_argument('--split', default='val', type=str, help='choose which data to use at visualization')
    parser.add_argument('--num_ckpt', default=None, type=int, help='index of checkpoint for validation')
    parser.add_argument('--last_ckpt', type=bool, default=False, help='use last checkpoint for validation')
    parser.add_argument('--best_ckpt', type=bool, default=False, help='use best checkpoint for validation')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize validation result')
    parser.add_argument('--val_save_dir', default='./', type=str, help='directory of saving visualized validation png')
    parser.add_argument('--cosine_sim',type=bool, default=False, help='visualize cosine similarity')
    parser.add_argument('--actor_name', default=None, type=str, help='actor name for validation')
    parser.add_argument('--full_length', type=bool, default=False, help='use full length of audio and expression')
    parser.add_argument('--repetition', default=None, type=int, help='choose what repetition to use for RAVDESS dataset, if None, use all')
    parser.add_argument('--statement', default=None, type=int, help='choose what statement to use for RAVDESS dataset, if None, use all')
    parser.add_argument('--key_emotions', type=bool, default=False, help='use calm, happy, sad, angry emotions for visualization')
    # check_for_errors(parser)
    
    # for prob_models
    parser.add_argument('--use_selfCL', action='store_true', help='use selfCL')
    parser.add_argument('--use_gt_matching_prob', type=bool, default=True, help='use ground truth matching probability')
    parser.add_argument('--input_embeddings', type=bool, default=False, help='not using raw audio as input but embeddings')
    parser.add_argument('--vib_beta', default=1.0e-6, type=float, help='beta for variational inference bottleneck')
    parser.add_argument('--smoothness_alpha', default=0., type=float, help='alpha for smoothness loss of pseudo labels')
    parser.add_argument('--inference_method', default='csd', type=str, help='inference method for variational inference', choices=['csd', 'sampling'])
    parser.add_argument('--num_samples', default=8, type=int, help='number of samples for variational inference')
    parser.add_argument('--add_n_mog', default=0, type=int, help='number of mixture of gaussian to add')
    return parser

def check_for_errors(args):
    if not args.use_RAVDESS and not args.use_MEAD:
        raise Exception('You have to use at least one dataset')
    if args.full_length and args.batch_size != 1:
        raise Exception('You have to use batch size 1 if you use full length')
    if args.num_audio_layers==0 and not args.no_cls:
        raise Exception('You have to use no cls token if you use no audio layers')
    # if args.temperature_fix == True and args.temperature == 0.:
    #     raise Exception('You have to give temperature value if temperature_fix is True')
    
    print("arguments are valid")