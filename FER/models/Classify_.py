import torch
import torch.nn as nn
import sys
import pickle
sys.path.append('FER/models')
from MLP import MLP

class ResidualBlock(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, out_dim, dropout=0.2, batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.skip_connection = nn.Linear(in_dim, out_dim)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        if args.model.activation == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        identity = x
        x = self.linear1(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.linear2(x)
        if self.batch_norm:
            x = self.bn(x)
            
        identity = self.skip_connection(identity)
        x += identity
        return x

class classifyNet(nn.Module):
    def __init__(self, args):
        super(classifyNet, self).__init__()
        with open('MEAD/flame_models/FLAME_masks.pkl', 'rb') as f:
            mask = pickle.load(f, encoding='latin1')
            self.lip = mask["lips"]
            self.eye_region = mask["eye_region"]
            self.forehead = mask["forehead"]
            self.nose = mask["nose"]
        self.lip_linear = nn.Linear(self.lip.shape[0]*3, 256)
        self.eye_region_linear = nn.Linear(self.eye_region.shape[0]*3, 256)
        self.forehead_linear = nn.Linear(self.forehead.shape[0]*3, 256)
        self.nose_linear = nn.Linear(self.nose.shape[0]*3, 256)
        self.face_linear = nn.Linear(args.model.input_dim, 512)
        self.total_linear = nn.Linear(256*4 + 512, 1024)

        self.mlp = MLP(input_dim = 1024, # if cfg.model.input_dim is 53, we are using jaw pose
                    layers = args.model.layers, 
                    output_dim = args.model.output_dim, # cfg.model.output_dim is same as label num
                    dropout = args.model.dropout, 
                    batch_norm = args.model.batch_norm, 
                    activation = args.model.activation)
        
    def forward(self, face):
        face = face.reshape(-1,5023,3)
        lip = face[:, self.lip, :].reshape(-1, self.lip.shape[0]*3)
        eye_region = face[:, self.eye_region, :].reshape(-1, self.eye_region.shape[0]*3)
        forehead = face[:, self.forehead, :].reshape(-1, self.forehead.shape[0]*3)
        nose = face[:, self.nose, :].reshape(-1, self.nose.shape[0]*3)
        face = face.reshape(-1, 5023*3)

        lip_out = self.lip_linear(lip)
        eye_region_out = self.eye_region_linear(eye_region)
        forehead_out = self.forehead_linear(forehead)
        nose_out = self.nose_linear(nose)
        face_out = self.face_linear(face)
        
        total_out = torch.cat((lip_out, eye_region_out, forehead_out, nose_out, face_out), dim=1)
        
        total_out = self.total_linear(total_out)

        out = self.mlp(total_out)

        return out
    
    def extract_feature_from_layer(self, face, layer_num):
        face = face.reshape(-1,5023,3)
        lip = face[:, self.lip, :].reshape(-1, self.lip.shape[0]*3)
        eye_region = face[:, self.eye_region, :].reshape(-1, self.eye_region.shape[0]*3)
        forehead = face[:, self.forehead, :].reshape(-1, self.forehead.shape[0]*3)
        nose = face[:, self.nose, :].reshape(-1, self.nose.shape[0]*3)
        face = face.reshape(-1, 5023*3)

        lip_out = self.lip_linear(lip)
        eye_region_out = self.eye_region_linear(eye_region)
        forehead_out = self.forehead_linear(forehead)
        nose_out = self.nose_linear(nose)
        face_out = self.face_linear(face)
        
        total_out = torch.cat((lip_out, eye_region_out, forehead_out, nose_out, face_out), dim=1)
        
        total_out = self.total_linear(total_out)

        feat = self.mlp.extract_feature_from_layer(total_out, layer_num)
        return feat

