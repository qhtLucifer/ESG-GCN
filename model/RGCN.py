from torch import nn
import torch
import numpy as np

class RGCNCell(nn.Module):

    def __init__(self, hidden_size: int, input_shape: int, A: torch.Tensor):
        super(RGCNCell, self).__init__()

        self.A = A
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_size + input_shape, input_shape),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size + input_shape, input_shape),
            nn.Sigmoid()
        )
        self.out_gate = nn.Sequential(
            nn.Linear(hidden_size + input_shape, input_shape),
            nn.Tanh()
        )

    def forward(self, inputs, prev_state):

        # data size is [batch, num_joint, feature_shape]
        
        batch_size = inputs.shape[0]
        
        inputs = torch.bmm(self.A.unsqueeze(0).repeat(batch_size,1,1), inputs) # [B, N, V] -> [B, N, V]
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            
            if torch.cuda.is_available():
                
                prev_state = torch.zeros(inputs.shape).cuda()
                
            else:
                
                prev_state = torch.zeros(inputs.shape)
        
        stacked_inputs = torch.cat([inputs, prev_state], dim=2)
        update =self.update_gate(stacked_inputs)
        reset = self.reset_gate(stacked_inputs)
        out_inputs = self.out_gate(torch.cat([inputs, prev_state * reset], dim=2))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

class RGCN(nn.Module):
    '''
    input size: num of the last dims.
    
    The shape of the inputs should be [Batch_size, seq_length, num_joints, feature_dim]
    
    And the outputs is `y` and `hidden_state`.
    
    The shape of `y`: [batch_size, seq_length, num_joints, hidden_size].
    
    The shape of `hidden_state`: [batch_size, num_joints, hidden_size].
    '''
    
    def __init__(self, input_size: int, hidden_size: int, A: np.array, num_joints: int = 25, hidden_output: bool = False):
        super(RGCN, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.cell = RGCNCell(self.input_size, self.hidden_size, A)
        self.num_joints = num_joints
        
        self.hidden_output = hidden_output


    def forward(self, X: torch.Tensor, h0 = None):
        # x.shape = (batch_size,seq_length, num_joint, coordinate_dims)
        # h0.shape = (batch_size, num_joint, hidden_size)
        # output.shape = (batch_size, seq_length,num_joint, hidden_size)
        batch_size: int = X.shape[0]
        seq_length: int = X.shape[1]
   
        if h0 is None:
        
            prev_h = torch.zeros([batch_size, self.num_joints, self.hidden_size])

        output = torch.zeros([batch_size, seq_length,self.num_joints, self.hidden_size])
        print(X[:,0].shape)
        
        for i in range(seq_length):
            
            prev_h = self.cell(X[:,i], prev_h)
            output[:, i] = prev_h
            
        if self.hidden_output:
            
            return output, prev_h
        
        return output