# Adapt from DEEPTalk: https://github.com/whwjdqls/DEEPTalk
from DEEPTalk.models.flame_models import flame
# from render_copy import single_render
import argparse
import torch
import json
import numpy as np
import os
from tqdm import tqdm
from PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer, get_vertices_from_FLAME,save_video

def main(args, verts_path, DEEPTalk_config):
    # DEEPTalk_config = "DEEPTalk/checkpoint/DEEPTalk/config_stage1_copy.json"
    # verts_path = "MEAD/Download_npy/M003/angry_level_1_001.npy"
    audio_path = verts_path.replace("Download_npy","Audio").replace("npy","wav")
    file_name = os.path.basename(verts_path).split('.')[0]

    save_dir = 'outputs'
    video_save_dir = os.path.join(save_dir, 'sample')
    subject = audio_path.split("/")[-2]
    os.makedirs(video_save_dir, exist_ok=True)
    output_name = f'{file_name}'
    out_video_path = os.path.join(video_save_dir, f'{args.model_name}_{subject}_{output_name}.mp4')
    
    with open(DEEPTalk_config) as f:
        config = json.load(f)
    device = torch.device("cpu")
    FLAME = flame.FLAME(config, batch_size=1).to(device).eval()
    FLAME.requires_grad_(False)
    verts = torch.tensor(np.load(verts_path)).to(device)

    print('getting flame model and calculating vertices...')
    B,F,P = verts.shape
    exp_param_pred = verts[:,:, :50]
    jaw_pose_pred = verts[:,:, 50:53]
    predicted_vertices = flame.get_vertices_from_flame(
            config, FLAME, exp_param_pred, jaw_pose_pred, device)
    predicted_vertices = predicted_vertices.reshape(B*F,-1,3) # (F,5023,3)

    print('initializing renderer...')
    template_mesh_path = 'DEEPTalk/models/flame_models/FLAME_sample.ply'
    width = 800
    height = 800
    renderer = PyRenderMeshSequenceRenderer(template_mesh_path,
                                            width=width,
                                            height=height)
    print(f'vertices shape : {predicted_vertices.shape}')
    print('rendering...')
    T = len(predicted_vertices)
    pred_images = []
    for t in tqdm(range(T)):
        pred_vertices = predicted_vertices[t].detach().cpu().view(-1,3).numpy()
        pred_image = renderer.render(pred_vertices)
        pred_images.append(pred_image) 
    
    print('saving video...')
    pred_images = np.stack(pred_images, axis=0)  
    
    save_video(out_video_path, pred_images, fourcc="mp4v", fps=25)
    if audio_path.endswith('.wav'):
        print('adding audio...')
        # audio_path = os.path.join(config.audio_path, f'{file_name}.wav')
        command = f'ffmpeg -y -i {out_video_path} -i {audio_path} -c:v copy -c:a aac -strict experimental {out_video_path.split(".mp4")[0]}_audio.mp4'
        os.system(command)
        os.remove(out_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='GT', help='model_name')
    parser.add_argument('--DEEPTalk_config', type=str, default='render/config.json', help='path to DEE checkpoint')
    parser.add_argument('--verts_path', type=str, default='vertice_path/angry_level_1_001.npy', help='path to DEE checkpoint')
    args = parser.parse_args()
    main(args, args.verts_path,args.DEEPTalk_config)