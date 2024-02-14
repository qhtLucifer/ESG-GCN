from typing import List,Dict

from torch import nn
import torch
import numpy as np

from utils.constants import paris
from utils.tools import load_model_configs
from utils.tools import calculate_adjacency


from einops import rearrange, repeat

from model.RGCN import RGCN
class ESRGCN(nn.Module):


    def __init__(self, args:Dict):
        '''
        semantics_type: `joint`, `bone`, `motion`

        '''
        super(ESRGCN, self).__init__()


        # Initialization.
        self.dataset_type = args.dataset_name
        self.device = args.device
        config:Dict = load_model_configs(args.config_path)
        self.semantics_type = config['semantics_type']

        self.input_channels:int = len(self.semantics_type) * 3 + 2 # 3-dims for each semantics and 2 dims for spatial and frame index semantics.

        # Model definition.

        self.rgcn = RGCN(input_size = self.input_channels, 
                         hidden_size = config['rgcn']['hidden_size'], 
                         A = calculate_adjacency('ntu'),
                         num_joints = config['rgcn']['num_joints'],
                         hidden_output = config['rgcn']['hidden_output']
                         )
  
        self.fusion_layer = nn.Sequential(
                                        nn.Mish(),
                                        nn.Linear(in_features = config['fusion_layer']['input_shape'], 
                                                    out_features = config['fusion_layer']['out_features']),
                                        nn.Softmax()
                                      )
        

        
    def forward(self, inputs : torch.Tensor) -> torch.Tensor:
        '''
        The shape of inputs: [batch_size, seq_length, num_joints, channels, head_count]
        The shape of outputs: [batch_size,num_classes, 1]
        '''

        torch._pad_packed_sequence()
        B, C, T, V, M = inputs.shape 

        inputs = rearrange(inputs, 'b c t v m -> (b m) t v c')

        inputs_concat = inputs

        if 'joint' not in self.semantics_type:

            inputs_concat = torch.Tensor().to(self.device)


        if 'bone' in self.semantics_type:

            bone_features = self.calculate_bone(inputs)
            inputs_concat = torch.cat([inputs_concat, bone_features],dim = 3) # [B, T, V, C] -> [B, T, V, 2*C]

        if 'motion' in self.semantics_type:

            motion_features = self.calculate_motion(inputs)

            inputs_concat = torch.cat([inputs_concat, motion_features], dim = 3) # [B, T, V, C] -> [B, T, V, 2*C] or  [B, T, V, 2*C] -> [B, T, V, 3*C]


        # Position features.  
        if self.dataset_type in ['ntu','ntu120']:

            joints_index = torch.arange(1,26).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(B,T,1,1)   # [25] -> [B, T, 25, 1]

        else:
            # TODO: Support other datasets.
            pass
        
        inputs_concat = torch.cat([inputs_concat, joints_index], dim = 3)

        #Frame features.
        if self.dataset_type in ['ntu','ntu120']:

            frame_index = torch.arange(1,T + 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat(B,1,25,1)   # [T] -> [B, T, 25, 1]

        else:
            # TODO: Support other datasets.
            pass
        
        inputs_concat = torch.cat([inputs_concat, frame_index], dim = 3)

        print(inputs_concat.shape)

        x1 = self.rgcn(inputs_concat)
        print(x1.shape)
        # Adjust dims: [B, T, V, C] -> [B, V * C]
        x2 = torch.flatten(x1, 1, 2)
        print(x2.shape)
        y = self.fusion_layer(x2)

        y = rearrange(y, '(b m) c t -> b m c t', m = M).mean(1)

        return y


    def calculate_bone(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        calculate bone vectors.
        The shape of inputs: [batch_size, seq_length, num_joints, channels]
        The shape of bone vectors: [batch_size, seq_length, num_joints, channels]
        '''

        B, T, V, C = inputs.shape()
        bone_inputs = torch.zeros((inputs.shape), dtype = torch.float32).to(self.device)

        for indexes in paris[self.dataset_type]:

            bone_inputs[:,:,indexes[0] - 1,:] = inputs[:, :, indexes[0] - 1, :] - inputs[:, :, indexes[1] - 1, :]

        return bone_inputs

    def calculate_motion(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
        calculate motion features.
        The shape of inputs: [batch_size, seq_length, num_joints, channels]
        The shape of bone vector: [batch_size, seq_length, num_joints, channels]

        **According to the motion calculation formula, the last frame will padding with zero.**
        '''

        B, T, V, C = inputs.shape()
        motion_inputs = torch.zeros((inputs.shape), dtype = torch.float32).to(self.device)

        for t in range(T - 1):

            motion_inputs[:,t,:,:] = inputs[:,t+1,:,:] - inputs[:,t,:,:]

        return motion_inputs
