# Borrowed from https://github.com/EvelynFan/FaceFormer/blob/main/data_loader.py
import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
# from utilities import *
from random import randint
import torchaudio

vocaset_sid = {
    "FaceTalk_170728_03272_TA":0,
    "FaceTalk_170904_00128_TA":1,
    "FaceTalk_170725_00137_TA":2,
    "FaceTalk_170915_00223_TA":3,
    "FaceTalk_170811_03274_TA":4,
    "FaceTalk_170913_03279_TA":5,
    "FaceTalk_170904_03276_TA":6,
    "FaceTalk_170912_03278_TA":7,
    "FaceTalk_170811_03275_TA":8,
    "FaceTalk_170908_03277_TA":9,
    "FaceTalk_170809_00138_TA":10,
    "FaceTalk_170731_00024_TA":11,
}

BIWI_sid = {
    "F1":0,
    "F2":1,
    "F3":2,
    "F4":3,
    "F5":4,
    "F6":5,
    "F7":6,
    "F8":7,
    "M1":8,
    "M2":9,
    "M3":10,
    "M4":11,
    "M5":12,
    "M6":13,
}

# training_ids = ['M003', 'M005', 'M007'#, 'M009', 'M011'#, 'M012' , 'M013'
#                 ] # 16 ids
# val_ids = ['M032'#, 'M033', 'M034'
# ]  # 5 ids

# test_ids = ['M037'#, 'M039', 'M040'
# ] # 8 ids

training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                ] # 32 ids
val_ids = ['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036']  # 7 ids

test_ids = ['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040'] # 8 ids

                    # 32 train_ids
MEAD_ACTOR_DICT = {'M003': 0, 'M005': 1, 'M007': 2, 'M009': 3, 'M011': 4, 'M012': 5, 'M013': 6, 'M019': 7, 'M022': 8, 'M023': 9, 'M024': 10, 'M025': 11, 'M026': 12, 'M027': 13, 'M028': 14, 'M029': 15, 'M030': 16, 'M031': 17, 'W009': 18, 'W011': 19, 'W014': 20, 'W015': 21, 'W016': 22, 'W018': 23, 'W019': 24, 'W021': 25, 'W023': 26, 'W024': 27, 'W025': 28, 'W026': 29, 'W028': 30, 'W029': 31, 
                   'M032': 32, 'M033': 33, 'M034': 34, 'M035': 35, 'W033': 36, 'W035': 37, 'W036': 38, # 7 val_ids
                   'M037': 39, 'M039': 40, 'M040': 41, 'M041': 42, 'M042': 43, 'W037': 44, 'W038': 45, 'W040': 46} # 8 test_ids

# EMOTION_DICT = {'neutral': 1, 'calm': 2, 'happy': 3, 'sad': 4, 'angry' :  5, 'fear': 6, 'disgusted': 7, 'surprised': 8, 'contempt' : 9}
# calm for RAVDESS
EMOTION_DICT = {'neutral': 0, 'happy': 1, 'sad': 2, 'surprised': 3, 'fear': 4, 'disgusted': 5, 'angry': 6, 'contempt': 7}
STRENGTH_DICT = {'1': 1, '2': 2, '3': 3}
# modify DICT to match inferno's original emotion label
# modify_DICT = {1:1, 3:2, 4:3, 5:7, 6:5, 7:6, 8:4, 9:8}
GENDER_DICT = {'M' : 0, 'W' : 1}
sentences = range(0,45,3)

class DatasetProperty():
    fps = None
    audio_rate = None

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, args, data, subjects_dict, subjects_idx, data_type="train"): 
        self.data = data   
        self.args = args
        self.dataset = args.dataset
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.subjects_idx = subjects_idx
        self.data_type = data_type
        self.args.sampling_rate = 16000
        # Need to test
        if self.dataset == "vocaset":
            DatasetProperty.fps = 30
        else:
            DatasetProperty.fps = 25
        DatasetProperty.audio_rate = self.args.sampling_rate

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        sids = self.data[index]["sids"]
        label = torch.tensor(self.data[index]['label'], dtype=torch.long)
        emo_level = torch.tensor(self.data[index]['emo_level'], dtype=torch.long)
        # refer_vert = self.get_refer(file_name, sids)
        return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), file_name, sids, label, emo_level
        # return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), file_name, torch.tensor(sids)

    def __len__(self):
        return self.len
    
    def get_refer(self,file_name,sids):
        subject_id = "_".join(file_name.split("_")[:-1])
        rand_seq = np.random.choice(self.subjects_idx[self.data_type][subject_id])
        assert self.data[rand_seq]["sids"] == sids
        return self.data[rand_seq]["vertice"]
    
    def get_rand(self):
        rand_subj = np.random.choice(self.subjects_idx[self.data_type])
        rand_seq = np.random.choice(self.subjects_idx[self.data_type][rand_subj])
        return self.data[rand_seq]["vertice"], self.data[rand_seq]["template"]
    
def custom_collate_fn(batch_data):
    min_mesh_len = None
    min_audio_len = None
    for audio, mesh, _, _, _, _, _ in batch_data:
        min_mesh_len = len(mesh) if min_mesh_len is None else min(min_mesh_len, len(mesh))
        min_audio_len = len(audio) if min_audio_len is None else min(min_audio_len, len(audio))

    min_mesh_len = min(min_mesh_len, int(min_audio_len * DatasetProperty.fps / DatasetProperty.audio_rate))
    min_audio_len = int(min_mesh_len * DatasetProperty.audio_rate / DatasetProperty.fps)

    mesh_tensor = []
    audio_tensor = []
    mesh_id_tensor = []
    mesh_template_tensor = []
    file_names = []
    label_tensor = []
    emo_level_tensor = []


    for audio, mesh, template, file_name, subj_id, label, emo_level in batch_data:
        rand_bound = min(len(mesh), int(len(audio) * DatasetProperty.fps / DatasetProperty.audio_rate))
        mesh_start_index = randint(0, rand_bound - min_mesh_len)
        mesh_end_index = mesh_start_index + min_mesh_len

        audio_start_idx = int(mesh_start_index * DatasetProperty.audio_rate / DatasetProperty.fps)
        audio_end_idx = audio_start_idx + min_audio_len

        mesh_tensor.append(mesh[mesh_start_index: mesh_end_index])
        audio_crop = audio[audio_start_idx: audio_end_idx]
        audio_tensor.append(audio_crop)
        mesh_id_tensor.append(subj_id)
        mesh_template_tensor.append(template)
        file_names.append(file_name)
        label_tensor.append(label)
        emo_level_tensor.append(emo_level)



    mesh_tensor = torch.stack(mesh_tensor, dim = 0)
    audio_tensor = torch.stack(audio_tensor, dim = 0)
    mesh_id_tensor = torch.LongTensor(mesh_id_tensor)
    label_tensor = torch.LongTensor(label_tensor)
    emo_level_tensor = torch.LongTensor(emo_level_tensor)
    mesh_template_tensor = torch.stack(mesh_template_tensor, dim = 0)

    return audio_tensor, mesh_tensor, mesh_template_tensor, file_names, mesh_id_tensor, label_tensor, emo_level_tensor

def custom_collate_fn_refer(batch_data):
    min_mesh_len = None
    min_audio_len = None
    for audio, mesh, refer_vert, _, _, _ in batch_data:
        min_mesh_len = len(mesh) if min_mesh_len is None else min(min_mesh_len, len(mesh))
        min_audio_len = len(audio) if min_audio_len is None else min(min_audio_len, len(audio))
        min_mesh_len = min(min_mesh_len,len(refer_vert))

    min_mesh_len = min(min_mesh_len, int(min_audio_len * DatasetProperty.fps / DatasetProperty.audio_rate))
    min_audio_len = int(min_mesh_len * DatasetProperty.audio_rate / DatasetProperty.fps)

    mesh_tensor = []
    refer_vert_tensor = []
    audio_tensor = []
    mesh_id_tensor = []
    mesh_template_tensor = []
    file_names = []

    for audio, mesh, refer_vert, template, file_name, subj_id in batch_data:
        rand_bound = min(min(len(mesh), int(len(audio) * DatasetProperty.fps / DatasetProperty.audio_rate)),len(refer_vert))
        mesh_start_index = randint(0, rand_bound - min_mesh_len)
        mesh_end_index = mesh_start_index + min_mesh_len

        audio_start_idx = int(mesh_start_index * DatasetProperty.audio_rate / DatasetProperty.fps)
        audio_end_idx = audio_start_idx + min_audio_len

        mesh_tensor.append(mesh[mesh_start_index: mesh_end_index])
        refer_vert_tensor.append(refer_vert[mesh_start_index: mesh_end_index])
        audio_crop = audio[audio_start_idx: audio_end_idx]
        audio_tensor.append(audio_crop)
        mesh_id_tensor.append(subj_id)
        mesh_template_tensor.append(template)
        file_names.append(file_name)

    mesh_tensor = torch.stack(mesh_tensor, dim = 0)
    refer_vert_tensor = torch.stack(refer_vert_tensor, dim = 0)
    audio_tensor = torch.stack(audio_tensor, dim = 0)
    mesh_id_tensor = torch.LongTensor(mesh_id_tensor)
    mesh_template_tensor = torch.stack(mesh_template_tensor, dim = 0)

    return audio_tensor, mesh_tensor, refer_vert_tensor, mesh_template_tensor, file_names, mesh_id_tensor

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_model)
    
    template_file = "MEAD/templates/template.npy"
    # template_file = os.path.join(args.dataset, args.template_file)
    template = np.load(template_file)
    # with open(template_file, 'rb') as fin:
    #     templates = pickle.load(fin, encoding='latin1')

    i=0
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs[:]): # If you want to test the pipeline, add "[:num]" after the "fs" eg.-->"fs[:16]",so that you can save time.
            if f.endswith("wav"):
                wav_path = os.path.join(r, f)
                key = f.replace("wav", "npy")
                subject_id = wav_path.split("/")[-2]
                key = "_".join([subject_id,key])
                emo = wav_path.split("/")[-1].split('_')[0]
                emo_level = STRENGTH_DICT[wav_path.split("/")[-1].split('_')[2]]
                label = EMOTION_DICT[emo]
                if args.stage == "train":
                    if (subject_id not in training_ids) and (subject_id not in val_ids):
                        continue
                else:
                    if subject_id not in test_ids:
                        continue
                if int(wav_path.split("/")[-1][-6:-4]) not in sentences:
                    continue
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                audio = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                data[key]["audio"] = audio
                temp = template
                data[key]["name"] = key
                data[key]["label"] = label
                data[key]["emo_level"] = emo_level
                data[key]["template"] = temp.reshape((-1))
                data[key]["sids"] = MEAD_ACTOR_DICT[subject_id]
                vertice_path = os.path.join(vertices_path, subject_id, f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)[::2, :]
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)
                    else:
                        data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)

    print(f"Total data: {len(data)}")
    subjects_dict = {}
    subjects_dict["train"] = training_ids
    subjects_dict["val"] = val_ids
    subjects_dict["test"] = test_ids
    

    train_index = {}
    valid_index = {}
    test_index = {}
    for i in subjects_dict["train"]:
        train_index[i] = []
    for i in subjects_dict["val"]:
        valid_index[i] = []
    for i in subjects_dict["test"]:
        test_index[i] = []

    # splits = {'vocaset': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)},
    #           'BIWI': {'train': range(1, 33), 'val': range(33, 37), 'test': range(37, 41)}}

    for k, v in data.items():
        subject_id = k.split("_")[0]
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
            train_index[subject_id].append(len(train_data)-1)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
            valid_index[subject_id].append(len(valid_data)-1)
        if subject_id in subjects_dict["test"]:
            test_data.append(v)
            test_index[subject_id].append(len(test_data)-1)

    subjects_idx = {'train': train_index,'val': valid_index,'test': test_index}
    
    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict, subjects_idx

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict, subjects_idx = read_data(args)
    # train_data, valid_data, test_data, subjects_dict = fake_data(args)
    train_data = Dataset(args, train_data, subjects_dict, subjects_idx, "train")
    if len(train_data)>0:
        if args.batch_size > 1:
            dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True,num_workers=4)
        else:
            dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True,pin_memory=True)
    valid_data = Dataset(args, valid_data, subjects_dict, subjects_idx, "val")
    if len(valid_data)>0:
        if args.batch_size > 1:
            dataset["val"] = data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True,num_workers=4)
        else:
            dataset["val"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True,pin_memory=True)
    test_data = Dataset(args, test_data, subjects_dict, subjects_idx, "test")
    if len(test_data)>0:
        if args.batch_size > 1:
            dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True,num_workers=4)
        else:
            dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True,pin_memory=True)

    return dataset


if __name__ == "__main__":
    args = get_parser()
    get_dataloaders(args)
