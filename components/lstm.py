import torch
import torch.nn as nn
import torch.nn.functional as F
from components.quat import QuaternionOps

class PointLSTMCell(nn.Module):
    def __init__(self, pts_num, in_channels, hidden_dim, offset_dim, bias=True, use_quat=True):
        super().__init__()
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.offset_dim = offset_dim
        self.use_quat = use_quat
        self.bias = bias
        
        # Feature transformation
        self.pool = nn.AdaptiveMaxPool2d((None, 1))
        
        # Calculate total input channels
        self.total_input_dim = in_channels + hidden_dim
        
        if use_quat:
            self.quat_ops = QuaternionOps()
            
            # Quaternion feature layers
            self.quat_conv = nn.Conv2d(4, offset_dim, kernel_size=1, bias=bias)
            self.consistency_conv = nn.Conv2d(1, offset_dim, kernel_size=1, bias=bias)
            
            # Update total input dimensions
            self.total_input_dim += offset_dim * 2
        else:
            # Position feature layer
            self.pos_conv = nn.Conv2d(3, offset_dim, kernel_size=1, bias=bias)
            self.total_input_dim += offset_dim
        
        # Main gates
        self.conv = nn.Conv2d(
            in_channels=self.total_input_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g gates
            kernel_size=1,
            bias=bias
        )

    def init_hidden(self, batch_size):
        """Initialize hidden states"""
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_dim, 1, 1, device=device),
            torch.zeros(batch_size, self.hidden_dim, 1, 1, device=device)
        )

    def forward(self, input_tensor, hidden_state, cell_state):
        device = input_tensor.device
        B, C, P, K = input_tensor.shape
        
        # Extract position info (first 3 channels for xyz)
        if C >= 3:
            pos_curr = input_tensor[:, :3]
            pos_prev = hidden_state[:, :3]
        else:
            pos_curr = torch.zeros(B, 3, P, K, device=device)
            pos_prev = torch.zeros(B, 3, 1, 1, device=device)
        
        # Collect features
        features = [input_tensor]
        
        # Match dimensions before concatenation
        hidden_expanded = hidden_state.expand(-1, -1, P, K)
        features.append(hidden_expanded)
        
        if self.use_quat:
            # Compute quaternion features
            with torch.cuda.amp.autocast(enabled=False):
                pos_prev_expanded = pos_prev.expand(-1, -1, P, K)
                consistency_error, rotation_quat = self.quat_ops.compute_motion_consistency(pos_prev_expanded, pos_curr)
            
            # Process quaternion features
            quat_features = self.quat_conv(rotation_quat)
            error_features = self.consistency_conv(consistency_error.unsqueeze(1))
            
            features.extend([quat_features, error_features])
        else:
            # Compute position-based features
            pos_prev_expanded = pos_prev.expand(-1, -1, P, K)
            pos_diff = pos_curr - pos_prev_expanded
            motion_features = self.pos_conv(pos_diff)
            features.append(motion_features)
        
        # Combine all features
        combined = torch.cat(features, dim=1)
        
        # Apply convolution to get gates
        gates = self.conv(combined)
        
        # Split into individual gates
        i, f, o, g = gates.chunk(4, dim=1)
        
        # Apply gate activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        # Update cell and hidden states
        c_next = f * cell_state + i * g
        h_next = o * torch.tanh(c_next)
        
        # Apply pooling if needed
        if self.pool is not None:
            h_next = self.pool(h_next)
            c_next = self.pool(c_next)
        
        return h_next, c_next

class PointLSTM(nn.Module):
    def __init__(self, pts_num, in_channels, hidden_dim, offset_dim, num_layers, 
                 topk=4, offsets=False, batch_first=True, bias=True, 
                 return_all_layers=False, use_quat=True):
        super().__init__()
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.use_quat = use_quat

        cell_list = []
        for i in range(num_layers):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_dim
            cell_list.append(PointLSTMCell(pts_num=self.pts_num,
                                         in_channels=cur_in_channels,
                                         hidden_dim=self.hidden_dim,
                                         offset_dim=offset_dim,
                                         bias=bias,
                                         use_quat=use_quat))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3)
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Process each point
            for t in range(input_tensor.size(2)):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, :, t:t+1],
                    hidden_state=h,
                    cell_state=c
                )
                output_inner.append(h)
            
            layer_output = torch.cat(output_inner, dim=2)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list, None

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states