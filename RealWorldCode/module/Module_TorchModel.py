###########################################################################################################
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


###########################################################################################################
# Basic Block of DirectEnsemble
class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(),
                batchNorm=True,  dropout=0.2):
        super().__init__()
        ff_block = [nn.Linear(input_dim, output_dim)]
        if activation:
            ff_block.append(activation)
        if batchNorm:
            ff_block.append(nn.BatchNorm1d(output_dim))
        if dropout > 0:
            ff_block.append(nn.Dropout(dropout))
        self.ff_block = nn.Sequential(*ff_block)
    
    def forward(self, x):
        return self.ff_block(x)


###########################################################################################################
class CategoricalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, n_layers=1, **kwargs):
        super().__init__()
        self.embedding_layers = nn.ModuleList([])
        self.n_layers = n_layers

        for _ in range(self.n_layers):
            self.embedding_layers.append(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, **kwargs))
        
    def forward(self, x):
        outputs_list = []

        for il in range(self.n_layers):
            emb = self.embedding_layers[il](x[..., il:il+1])
            outputs_list.append(emb)
        # return torch.cat(outputs_list, dim=-2)
        return torch.cat(outputs_list, dim=-1).squeeze(-2)

###########################################################################################################
# Temporal Embedding
class PeriodicEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # Linear Component
        self.linear_layer = nn.Linear(input_dim , 1)
        if embed_dim % 2 == 0:
            self.linear_layer2 = nn.Linear(input_dim , 1)
        else:
            self.linear_layer2 = None
        
        # Periodic Components
        self.periodic_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.periodic_bias = nn.Parameter(torch.randn(1, (embed_dim - 1)//2 ))

        # NonLinear Purse Periodic Component
        self.nonlinear_weights = nn.Parameter(torch.randn(input_dim, (embed_dim - 1)//2 ))
        self.nonlinear_bias = nn.Parameter(torch.randn(1, (embed_dim - 1)//2 ))

    def forward(self, x):
        # Linear Component
        linear_term = self.linear_layer(x)
        
        # Periodic Component
        periodic_term = torch.sin(x @ self.periodic_weights + self.periodic_bias)

        # NonLinear Purse Periodic Component
        nonlinear_term = torch.sign(torch.sin(x @ self.nonlinear_weights + self.nonlinear_bias))
        
        # Combine All Components
        if self.linear_layer2 is None:
            return torch.cat([linear_term, periodic_term, nonlinear_term], dim=-1)
        else:
            linear_term2 = self.linear_layer2(x)
            return torch.cat([linear_term, linear_term2, periodic_term, nonlinear_term], dim=-1)


# -------------------------------------------------------------------------------------------
# ★ Main Embedding
class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = input_dim * embed_dim

        if hidden_dim is None:
            self.temporal_embed_layers = nn.ModuleList([PeriodicEmbedding(input_dim=1, embed_dim=embed_dim) for _ in range(input_dim)])
        else:
            self.temporal_embed_layers = nn.ModuleList([PeriodicEmbedding(input_dim=1, embed_dim=hidden_dim) for _ in range(input_dim)])
            self.hidden_layer = nn.Linear(input_dim*hidden_dim, embed_dim)
            self.embed_dim = embed_dim
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (2,) -> (1, 2)
        emb_outputs = [layer(x[:,i:i+1]) for i, layer in enumerate(self.temporal_embed_layers)]
        output = torch.cat(emb_outputs, dim=1)
        if self.hidden_dim is not None:
            output = self.hidden_layer(output)

        return output


###########################################################################################################
# Spatial Embedding
class CoordinateEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, depth=1):
        super().__init__()
        self.embedding_block = nn.ModuleDict({'in_layer':FeedForwardBlock(input_dim, hidden_dim, batchNorm=False, dropout=0)})

        for h_idx in range(depth):
            if h_idx < depth-1:
               self.embedding_block[f'hidden_layer{h_idx+1}'] = FeedForwardBlock(hidden_dim, hidden_dim, batchNorm=False, dropout=0)
            else:
                self.embedding_block['out_layer'] = FeedForwardBlock(hidden_dim, embed_dim, activation=False, batchNorm=False, dropout=0)

    def forward(self, x):
        for layer_name, layer in self.embedding_block.items():
            if layer_name == 'in_layer' or layer_name == 'out_layer':
                x = layer(x)
            else:
                x = layer(x) + x
        return x

# -------------------------------------------------------------------------------------------
class GridEmbedding(nn.Module):
    def __init__(self, grid_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(grid_size**2, embed_dim)
        self.grid_size = grid_size

    def forward(self, x):
        # 좌표를 그리드로 매핑
        x_grid = (x * self.grid_size).long()  # 좌표를 그리드 인덱스로 변환
        x_index = x_grid[:, 0] * self.grid_size + x_grid[:, 1]  # 인덱스화
        # print(x_grid, x_index)
        return self.embedding(x_index)

# -------------------------------------------------------------------------------------------
def positional_encoding(coords, d_model):
    # coords: [N, 2]
    N, dim = coords.shape
    pe = []
    for i in range(d_model // 4):
        freq = 10000 ** (2 * i / d_model)
        pe.append(np.sin(coords * freq))
        pe.append(np.cos(coords * freq))
    pe = np.concatenate(pe, axis=1)  # [N, 2*d_model//2]
    return torch.tensor(pe, dtype=torch.float32)


# -------------------------------------------------------------------------------------------
# ★ Main Embedding Block
class SpatialEmbedding(nn.Module):
    def __init__(self, embed_dim=None, coord_hidden_dim=32, coord_embed_dim=8, coord_depth=2,
                grid_size=10, grid_embed_dim=8, periodic_embed_dim=5, 
                relative=True, euclidean_dist=True, angle=True):
        """
        embed_dim : (None) end with combined result
        coord_embed_dim : (None) not use coordinate embedding
        grid_embed_dim : (None) not use grid embedding
        periodic_embed_dim : (None) not use periodic embedding
        relative : (False) not use relative coordinate, (True) use relative coordinate
        euclidean_dist : (False) not use euclidean distance, (True) use euclidean distance
        angle : (False) not use angle, (True) use angle

        """
        super().__init__()
        self.coord_hidden_dim = coord_hidden_dim        ## 32
        self.coord_embed_dim = coord_embed_dim          ## 4
        self.grid_size = grid_size                      ## 10
        self.grid_embed_dim = grid_embed_dim            ## 4
        self.periodic_embed_dim = periodic_embed_dim    ## 3

        self.relative = relative                    ## True: 2
        self.euclidean_dist = euclidean_dist        ## True : 1
        self.angle = angle                          ## True : 1
        
        self.embed_dim = 0

        if self.coord_embed_dim is not None:
            self.coord_embedding = CoordinateEmbedding(input_dim=2, hidden_dim=coord_hidden_dim, embed_dim=coord_embed_dim, depth=coord_depth)
            self.embed_dim += self.coord_embed_dim * 2

        if self.grid_embed_dim is not None:
            self.grid_embedding = GridEmbedding(grid_size=grid_size, embed_dim=grid_embed_dim)
            self.embed_dim += self.grid_embed_dim * 2

        if self.periodic_embed_dim is not None:
            self.periodic_embedding = PeriodicEmbedding(input_dim=2, embed_dim=periodic_embed_dim)
            self.embed_dim += self.periodic_embed_dim * 2
        
        if self.relative:
            self.embed_dim += 2
        
        if self.euclidean_dist:
            self.embed_dim += 1

        if self.angle:
            self.embed_dim += 1

        if embed_dim is not None:
            self.hidden_dim = self.embed_dim
            self.embed_dim = embed_dim
            self.hidden_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.hidden_dim = None

    def forward(self, coord1, coord2):
        spatial_embeddings = []
        # [embed_1, embed_2, grid_1, grid_2, relative, euclidean_dist, angle, period_1, period_2]

        # embed
        if self.coord_embed_dim is not None:
            embed_1 = self.coord_embedding(coord1)
            embed_2 = self.coord_embedding(coord2)
            spatial_embeddings.append(embed_1)
            spatial_embeddings.append(embed_2)

        # grid
        if self.grid_embed_dim is not None:
            grid_1 = self.grid_embedding(coord1)
            grid_2 = self.grid_embedding(coord1)
            spatial_embeddings.append(grid_1)
            spatial_embeddings.append(grid_2)
        
        # periodic
        if self.periodic_embed_dim is not None:
            period_1 = self.periodic_embedding(coord1)
            period_2 = self.periodic_embedding(coord2)
            spatial_embeddings.append(period_1)
            spatial_embeddings.append(period_2)

        # norm
        if self.relative:
            relative = coord2 - coord1 
            spatial_embeddings.append(relative)

        if self.euclidean_dist:
            euclidean_dist = torch.norm(coord2 - coord1, p=2, dim=1, keepdim=True)
            spatial_embeddings.append(euclidean_dist)

        # angle
        if self.angle:
            relative = coord2 - coord1
            angle = torch.atan2(relative[:,1], relative[:,0]).unsqueeze(1)  
            spatial_embeddings.append(angle)

        # combine
        output = torch.cat(spatial_embeddings, dim=1)
        # embed_dim = coord_embed_dim * 2 + grid_embed_dim * 2 + periodic_embed_dim * 2 + 2(relative) + 1(euclidean_dist) + 1(angle)

        if self.hidden_dim is not None:
            output = self.hidden_layer(output)
        return output









###########################################################################################################
# Combined Embedding
class CombinedEmbedding(nn.Module):
    def __init__(self, categorical_emb_dim = None, numerical_emb_dim = None, temporal_emb_dim = None, spatial_emb_dim = None,
                categorical_input_dim = None, numerical_input_dim = None, temporal_input_dim = None,
                categorical_num_embedding = None,
                categorical_params = {}, numerical_params = {'batchNorm':False, 'dropout':0}, temporal_params = {}, spatial_params = {}
                ):
        super().__init__()

        self.embedding_layers = nn.ModuleDict({})
        self.embed_dim = 0

        self.categorical_input_dim = 0
        self.numerical_input_dim = 0
        self.temporal_input_dim = 0
        self.spatial_input_dim = 0

        self.categorical_emb_dim = 0
        self.numerical_emb_dim = 0
        self.temporal_emb_dim = 0
        self.spatial_emb_dim = 0

        # categorical
        if (categorical_emb_dim is not None) and (categorical_input_dim is not None) and (categorical_num_embedding is not None):
            self.embedding_layers['categorical_layer'] = CategoricalEmbedding(num_embeddings=categorical_num_embedding, embedding_dim=categorical_emb_dim, n_layers=categorical_input_dim, **categorical_params)
            self.embed_dim += categorical_emb_dim
            self.categorical_input_dim = categorical_input_dim
            self.categorical_emb_dim = categorical_emb_dim
        
        # numerical
        if (numerical_emb_dim is not None) and (numerical_input_dim is not None):
            self.embedding_layers['numerical_layer'] = FeedForwardBlock(input_dim=numerical_input_dim, output_dim=numerical_emb_dim, **numerical_params)
            self.embed_dim += numerical_emb_dim
            self.numerical_input_dim = numerical_input_dim
            self.numerical_emb_dim = numerical_emb_dim

        # temporal
        if (temporal_emb_dim is not None) and (temporal_input_dim is not None):
            self.embedding_layers['temporal_layer'] = TemporalEmbedding(input_dim=temporal_input_dim, embed_dim=temporal_emb_dim, **temporal_params)
            self.embed_dim += temporal_emb_dim
            self.temporal_input_dim = temporal_input_dim
            self.temporal_emb_dim = temporal_emb_dim

        # spatial
        if (spatial_emb_dim is not None):
            self.embedding_layers['sptial_layer'] = SpatialEmbedding(embed_dim=spatial_emb_dim, **spatial_params)
            self.embed_dim += spatial_emb_dim
            self.spatial_input_dim = 4
            self.spatial_emb_dim = spatial_emb_dim
        
        self.embed_group_dims = (self.categorical_input_dim, self.numerical_input_dim, self.temporal_input_dim, self.spatial_input_dim)

    def input_split(self, x):
        splited_input = []
        init_idx = 0
        xr = torch.tensor([]).type(torch.float32)
        for ir, r in enumerate(self.embed_group_dims):
            if r > 0:
                xr = x[..., init_idx:init_idx+r].type(torch.float32)
                if ir == 0:
                    xr = xr.type(torch.int64)
            splited_input.append(xr)
            init_idx += r
        return splited_input

    def forward(self, x=None, categorical_x=None, numerical_x=None, temporal_x=None, spatial_x=None):
        if x is not None:
            categorical_x, numerical_x, temporal_x, spatial_x = self.input_split(x)
        else:
            categorical_x = torch.tensor([]).type(torch.int64) if categorical_x is None else categorical_x.type(torch.int64)
            numerical_x = torch.tensor([]).type(torch.float32) if numerical_x is None else numerical_x.type(torch.float32)
            temporal_x = torch.tensor([]).type(torch.float32) if temporal_x is None else temporal_x.type(torch.float32)
            spatial_x = torch.tensor([]).type(torch.float32) if spatial_x is None else spatial_x.type(torch.float32)

        embed_output_list = []
        if ('categorical_layer' in self.embedding_layers.keys()) and (len(categorical_x) > 0):
            categorical_embedding = self.embedding_layers['categorical_layer'](categorical_x)
            embed_output_list.append(categorical_embedding)

        if ('numerical_layer' in self.embedding_layers.keys()) and (len(numerical_x) > 0):
            numerical_embedding = self.embedding_layers['numerical_layer'](numerical_x)
            embed_output_list.append(numerical_embedding)
        
        if ('temporal_layer' in self.embedding_layers.keys()) and (len(temporal_x) > 0):
            temporal_embedding = self.embedding_layers['temporal_layer'](temporal_x)
            embed_output_list.append(temporal_embedding)

        if ('sptial_layer' in self.embedding_layers.keys()) and (len(spatial_x) > 0):
            spatial_embedding = self.embedding_layers['sptial_layer'](spatial_x[...,:2], spatial_x[...,2:])
            embed_output_list.append(spatial_embedding)

        embed_ouput = torch.cat(embed_output_list, dim=-1)
        return embed_ouput



















###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################




###########################################################################################################
###########################################################################################################
###########################################################################################################


# BaroNowModel
class BaroNowResNet_V2(nn.Module):
    def __init__(self, output_dim=1, n_layers=3, n_output=10, 
                base_embedding_params={}, timemachine_embedding_params={}, api_embedding_params={}, train_mode='forward'):
        super().__init__()

        # embedding_layer
        self.base_embedding_layer = CombinedEmbedding(**base_embedding_params)
        self.timemahcine_embedding_layer = CombinedEmbedding(**timemachine_embedding_params)
        self.api_embedding_layer = CombinedEmbedding(**api_embedding_params)

        # embedding_dimension
        self.base_embed_dim = self.base_embedding_layer.embed_dim
        self.timemahcine_embed_dim = self.base_embed_dim + self.timemahcine_embedding_layer.embed_dim
        self.api_embed_dim = self.timemahcine_embed_dim + self.api_embedding_layer.embed_dim
        out_dim = output_dim*n_output
        
        # final_layer
        self.base_fc_block = nn.ModuleDict({})
        self.timemachine_fc_block =  nn.ModuleDict({})
        self.api_fc_block =  nn.ModuleDict({})

        for h_idx in range(n_layers):
            self.base_fc_block[f'hidden_layer{h_idx}'] = FeedForwardBlock(self.base_embed_dim, self.base_embed_dim, batchNorm=False, dropout=0)
            self.timemachine_fc_block[f'hidden_layer{h_idx}'] = FeedForwardBlock(self.timemahcine_embed_dim, self.timemahcine_embed_dim, batchNorm=False, dropout=0)
            self.api_fc_block[f'hidden_layer{h_idx}'] = FeedForwardBlock(self.api_embed_dim, self.api_embed_dim, batchNorm=False, dropout=0)
        
        self.base_mu_layer = FeedForwardBlock(self.base_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.base_std_layer = FeedForwardBlock(self.base_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        self.timemachine_mu_layer = FeedForwardBlock(self.timemahcine_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.timemachine_std_layer = FeedForwardBlock(self.timemahcine_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        self.api_mu_layer = FeedForwardBlock(self.api_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.api_std_layer = FeedForwardBlock(self.api_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        # params
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_output = n_output
        self.train_mode = train_mode

    # weight_activate
    def weight_activate(self, activate=None, deactivate=None, all_activate=None):
        # all weight activate
        if (all_activate is True) or (all_activate is False):
            for param in self.parameters():
                param.requires_grad = all_activate

        if (activate is not None) and isinstance(activate, nn.Module):
            for name, param in activate.named_parameters():
                param.requires_grad = True

        if (deactivate is not None)and isinstance(deactivate, nn.Module):
            for name, param in deactivate.named_parameters():
                param.requires_grad = False

    # requires_grad_status
    def requires_grad_status(self, layer=None, verbose=0):
        if layer is None:
            layer = self

        require_grad_list = {}
        if isinstance(layer, nn.Module):
            for name, param in layer.named_parameters():
                require_grad_list[name] = param.requires_grad
                
                if verbose > 0:
                    print(f"({name}) {param.requires_grad}")
            return require_grad_list
    
    # forward_base
    def forward_structure(self, x_list=[], embedding_layer_dict={}, fc_block=None, final_layer=None, activate_fc_block=True):
        # weight freezing / activate
        self.weight_activate(all_activate=False)
        if activate_fc_block:
            self.weight_activate(activate=fc_block)
        self.weight_activate(activate=final_layer)

        embed_output_list = []
        for x_input, (layer, require_grad) in zip(x_list, embedding_layer_dict.items()):
            if require_grad is True:
                self.weight_activate(activate=layer)
            embed_output_list.append(layer(x_input))

        # embedding concat
        x_hidden = torch.cat(embed_output_list, dim=-1)

        # fc block + residual connection
        for layer_name, layer in fc_block.items():
                x_hidden = layer(x_hidden) + x_hidden    # residual connection
        return final_layer(x_hidden)
    
    # forward_freeze_forward
    def forward_freeze_forward(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu'):
        x_list = []
        embedding_layer_dict = {}
        fc_block = None
        final_layer = None
        activate_fc_block = True if pred_type=='mu' else False
        if (base_x is not None) and (timemachine_x is None) and (api_x is None):
            x_list = [base_x]
            embedding_layer_dict[self.base_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.base_fc_block
            final_layer = self.base_mu_layer if pred_type == 'mu' else self.base_std_layer 

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is None):
            x_list=[base_x, timemachine_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.timemachine_fc_block
            final_layer = self.timemachine_mu_layer if pred_type == 'mu' else self.timemachine_std_layer

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is not None):
            x_list=[base_x, timemachine_x, api_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = False
            embedding_layer_dict[self.api_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.api_fc_block
            final_layer = self.api_mu_layer if pred_type == 'mu' else self.api_std_layer
        
        return self.forward_structure(x_list, embedding_layer_dict, fc_block, final_layer=final_layer, activate_fc_block=activate_fc_block)
    
    # forward_freeze_backward
    def forward_freeze_backward(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu', deactivate_all=False):
        x_list = []
        embedding_layer_dict = {}
        fc_block = None
        final_layer = None
        activate_fc_block = True if pred_type=='mu' else False
        if (base_x is not None) and (timemachine_x is None) and (api_x is None):
            x_list = [base_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            fc_block = self.base_fc_block
            final_layer = self.base_mu_layer if pred_type == 'mu' else self.base_std_layer 

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is None):
            x_list=[base_x, timemachine_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = False
            fc_block = self.timemachine_fc_block
            final_layer = self.timemachine_mu_layer if pred_type == 'mu' else self.timemachine_std_layer

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is not None):
            x_list=[base_x, timemachine_x, api_x]
            embedding_layer_dict[self.base_embedding_layer] = True if pred_type=='mu' else False
            embedding_layer_dict[self.timemahcine_embedding_layer] = True if pred_type=='mu' else False
            embedding_layer_dict[self.api_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.api_fc_block
            final_layer = self.api_mu_layer if pred_type == 'mu' else self.api_std_layer
        
        return self.forward_structure(x_list, embedding_layer_dict, fc_block, final_layer=final_layer, activate_fc_block=activate_fc_block)

    # forward_pred_type
    def forward_pred_type(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu'):
        if self.train_mode == 'forward':
            output = self.forward_freeze_forward(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type=pred_type)
        elif self.train_mode == 'backward':
            output = self.forward_freeze_backward(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type=pred_type)
        
        if pred_type == 'mu':
            return output.mean(dim=-1, keepdims=True)
        if pred_type == 'std':
            return output.std(dim=-1, keepdims=True)

    # forward
    def forward(self, base_x=None, timemachine_x=None, api_x=None, pred_type=None):
        if pred_type is None:
            mu = self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='mu')
            std = self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='std')
            return (mu, std)
        
        elif pred_type == 'mu':
            return self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='mu')

        elif pred_type =='std':
            return self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='std')
        





# ###########################################################################################################


# BaroNowTransformer
class BaroNowTransformer_V2(nn.Module):
    def __init__(self, output_dim=1, n_layers=3, n_output=10, n_head=8, transformer_n_layers=2,
                base_embedding_params={}, timemachine_embedding_params={}, api_embedding_params={}, train_mode='forward'):
        super().__init__()

        # embedding_layer
        self.base_embedding_layer = CombinedEmbedding(**base_embedding_params)
        self.timemahcine_embedding_layer = CombinedEmbedding(**timemachine_embedding_params)
        self.api_embedding_layer = CombinedEmbedding(**api_embedding_params)

        # embedding_dimension
        self.base_embed_dim = self.base_embedding_layer.embed_dim
        self.timemahcine_embed_dim = self.base_embed_dim + self.timemahcine_embedding_layer.embed_dim
        self.api_embed_dim = self.timemahcine_embed_dim + self.api_embedding_layer.embed_dim
        out_dim = output_dim*n_output
        
        # final_layer
        self.base_block = nn.ModuleDict({})
        self.base_block['transforemr_encoder_layer'] = nn.TransformerEncoderLayer(d_model=self.base_embed_dim, nhead=n_head, dim_feedforward=self.base_embed_dim, batch_first=True)
        self.base_block['transforemr_encoder'] = nn.TransformerEncoder(self.base_block['transforemr_encoder_layer'], num_layers=transformer_n_layers)
        self.base_mu_layer = FeedForwardBlock(self.base_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.base_std_layer = FeedForwardBlock(self.base_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        self.timemachine_block = nn.ModuleDict({})
        self.timemachine_block['transforemr_encoder_layer'] = nn.TransformerEncoderLayer(d_model=self.timemahcine_embed_dim, nhead=n_head, dim_feedforward=self.timemahcine_embed_dim, batch_first=True)
        self.timemachine_block['transforemr_encoder'] = nn.TransformerEncoder(self.timemachine_block['transforemr_encoder_layer'], num_layers=transformer_n_layers)
        self.timemachine_mu_layer = FeedForwardBlock(self.timemahcine_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.timemachine_std_layer = FeedForwardBlock(self.timemahcine_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        self.api_block = nn.ModuleDict({})
        self.api_block['transforemr_encoder_layer'] = nn.TransformerEncoderLayer(d_model=self.api_embed_dim, nhead=n_head, dim_feedforward=self.api_embed_dim, batch_first=True)
        self.api_block['transforemr_encoder'] = nn.TransformerEncoder(self.api_block['transforemr_encoder_layer'], num_layers=transformer_n_layers)
        self.api_mu_layer = FeedForwardBlock(self.api_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.api_std_layer = FeedForwardBlock(self.api_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        # params
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_output = n_output
        self.train_mode = train_mode

    # weight_activate
    def weight_activate(self, activate=None, deactivate=None, all_activate=None):
        # all weight activate
        if (all_activate is True) or (all_activate is False):
            for param in self.parameters():
                param.requires_grad = all_activate

        if (activate is not None) and isinstance(activate, nn.Module):
            for name, param in activate.named_parameters():
                param.requires_grad = True

        if (deactivate is not None)and isinstance(deactivate, nn.Module):
            for name, param in deactivate.named_parameters():
                param.requires_grad = False

    # requires_grad_status
    def requires_grad_status(self, layer=None, verbose=0):
        if layer is None:
            layer = self

        require_grad_list = {}
        if isinstance(layer, nn.Module):
            for name, param in layer.named_parameters():
                require_grad_list[name] = param.requires_grad
                
                if verbose > 0:
                    print(f"({name}) {param.requires_grad}")
            return require_grad_list
    
    # forward_base
    def forward_structure(self, x_list=[], embedding_layer_dict={}, fc_block=None, final_layer=None, activate_fc_block=True):
        # weight freezing / activate
        self.weight_activate(all_activate=False)
        if activate_fc_block:
            self.weight_activate(activate=fc_block)
        self.weight_activate(activate=final_layer)

        embed_output_list = []
        for x_input, (layer, require_grad) in zip(x_list, embedding_layer_dict.items()):
            if require_grad is True:
                self.weight_activate(activate=layer)
            embed_output_list.append(layer(x_input))

        # embedding concat
        x_hidden = torch.cat(embed_output_list, dim=-1)

        # transformer_encoder & fc_layer
        x_hidden = fc_block['transforemr_encoder'](x_hidden)
        return final_layer(x_hidden)
       
    # forward_freeze_forward
    def forward_freeze_forward(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu'):
        x_list = []
        embedding_layer_dict = {}
        fc_block = None
        final_layer = None
        activate_fc_block = True if pred_type=='mu' else False
        if (base_x is not None) and (timemachine_x is None) and (api_x is None):
            x_list = [base_x]
            embedding_layer_dict[self.base_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.base_block
            final_layer = self.base_mu_layer if pred_type == 'mu' else self.base_std_layer

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is None):
            x_list=[base_x, timemachine_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.timemachine_block
            final_layer = self.timemachine_mu_layer if pred_type == 'mu' else self.timemachine_std_layer

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is not None):
            x_list=[base_x, timemachine_x, api_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = False
            embedding_layer_dict[self.api_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.api_block
            final_layer = self.api_mu_layer if pred_type == 'mu' else self.api_std_layer
        
        return self.forward_structure(x_list, embedding_layer_dict, fc_block, final_layer=final_layer, activate_fc_block=activate_fc_block)
    
    # forward_freeze_backward
    def forward_freeze_backward(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu', deactivate_all=False):
        x_list = []
        embedding_layer_dict = {}
        fc_block = None
        final_layer = None
        activate_fc_block = True if pred_type=='mu' else False
        if (base_x is not None) and (timemachine_x is None) and (api_x is None):
            x_list = [base_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            fc_block = self.base_block
            final_layer = self.base_mu_layer if pred_type == 'mu' else self.base_std_layer

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is None):
            x_list=[base_x, timemachine_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = False
            fc_block = self.timemachine_block
            final_layer = self.timemachine_mu_layer if pred_type == 'mu' else self.timemachine_std_layer

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is not None):
            x_list=[base_x, timemachine_x, api_x]
            embedding_layer_dict[self.base_embedding_layer] = True if pred_type=='mu' else False
            embedding_layer_dict[self.timemahcine_embedding_layer] = True if pred_type=='mu' else False
            embedding_layer_dict[self.api_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.api_block
            final_layer = self.api_mu_layer if pred_type == 'mu' else self.api_std_layer
        
        return self.forward_structure(x_list, embedding_layer_dict, fc_block, final_layer=final_layer, activate_fc_block=activate_fc_block)

    # forward_pred_type
    def forward_pred_type(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu'):
        if self.train_mode == 'forward':
            output = self.forward_freeze_forward(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type=pred_type)
        elif self.train_mode == 'backward':
            output = self.forward_freeze_backward(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type=pred_type)
        
        if pred_type == 'mu':
            return output.mean(dim=-1, keepdims=True)
        if pred_type == 'std':
            return output.std(dim=-1, keepdims=True)

    # forward
    def forward(self, base_x=None, timemachine_x=None, api_x=None, pred_type=None):
        if pred_type is None:
            mu = self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='mu')
            std = self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='std')
            return (mu, std)
        
        elif pred_type == 'mu':
            return self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='mu')

        elif pred_type =='std':
            return self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='std')





























############################################################################################################
############################################################################################################
############################################################################################################


# BaroNowModel
class BaroNowResNet_V3(nn.Module):
    def __init__(self, output_dim=1, n_layers=3, n_output=10, 
                base_embedding_params={}, timemachine_embedding_params={}, api_embedding_params={}):
        super().__init__()

        # embedding_layer
        self.base_embedding_layer = CombinedEmbedding(**base_embedding_params)
        self.distance_embedding_layer = SpatialEmbedding(embed_dim=None, coord_embed_dim=None, grid_embed_dim=None, periodic_embed_dim=None, 
                                                    relative=False, euclidean_dist=True, angle=False)
        
        self.timemahcine_embedding_layer = CombinedEmbedding(**timemachine_embedding_params)
        self.api_embedding_layer = CombinedEmbedding(**api_embedding_params)

        # embedding_dimension
        self.base_embed_dim = self.base_embedding_layer.embed_dim
        self.timemahcine_embed_dim = self.base_embed_dim + self.timemahcine_embedding_layer.embed_dim
        self.api_embed_dim = self.timemahcine_embed_dim + self.api_embedding_layer.embed_dim
        out_dim = output_dim*n_output
        
        # final_layer
        self.base_fc_block = nn.ModuleDict({})
        self.timemachine_fc_block =  nn.ModuleDict({})
        self.api_fc_block =  nn.ModuleDict({})

        for h_idx in range(n_layers):
            self.base_fc_block[f'hidden_layer{h_idx}'] = FeedForwardBlock(self.base_embed_dim, self.base_embed_dim, batchNorm=False, dropout=0)
            self.timemachine_fc_block[f'hidden_layer{h_idx}'] = FeedForwardBlock(self.timemahcine_embed_dim, self.timemahcine_embed_dim, batchNorm=False, dropout=0)
            self.api_fc_block[f'hidden_layer{h_idx}'] = FeedForwardBlock(self.api_embed_dim, self.api_embed_dim, batchNorm=False, dropout=0)
        
        self.base_mu_layer = FeedForwardBlock(self.base_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.base_std_layer = FeedForwardBlock(self.base_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        self.timemachine_mu_layer = FeedForwardBlock(self.timemahcine_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.timemachine_std_layer = FeedForwardBlock(self.timemahcine_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        self.api_mu_layer = FeedForwardBlock(self.api_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.api_std_layer = FeedForwardBlock(self.api_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        # params
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_output = n_output

    # weight_activate
    def weight_activate(self, activate=None, deactivate=None, all_activate=None):
        # all weight activate
        if (all_activate is True) or (all_activate is False):
            for param in self.parameters():
                param.requires_grad = all_activate

        if (activate is not None) and isinstance(activate, nn.Module):
            for name, param in activate.named_parameters():
                param.requires_grad = True

        if (deactivate is not None)and isinstance(deactivate, nn.Module):
            for name, param in deactivate.named_parameters():
                param.requires_grad = False

    # requires_grad_status
    def requires_grad_status(self, layer=None, verbose=0):
        if layer is None:
            layer = self

        require_grad_list = {}
        if isinstance(layer, nn.Module):
            for name, param in layer.named_parameters():
                require_grad_list[name] = param.requires_grad
                
                if verbose > 0:
                    print(f"({name}) {param.requires_grad}")
            return require_grad_list
    
    # forward_base
    def forward_structure(self, x_list=[], embedding_layer_dict={}, fc_block=None, final_layer=None, activate_fc_block=True):
        # weight freezing / activate
        self.weight_activate(all_activate=False)
        if activate_fc_block:
            self.weight_activate(activate=fc_block)
        self.weight_activate(activate=final_layer)

        embed_output_list = []
        for x_input, (layer, require_grad) in zip(x_list, embedding_layer_dict.items()):
            if require_grad is True:
                self.weight_activate(activate=layer)
            embed_output_list.append(layer(x_input))

        # embedding concat
        x_hidden = torch.cat(embed_output_list, dim=-1)

        # fc block + residual connection
        for layer_name, layer in fc_block.items():
                x_hidden = layer(x_hidden) + x_hidden    # residual connection
        return final_layer(x_hidden)
    
    # forward_freeze_forward
    def forward_freeze_forward(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu'):
        x_list = []
        embedding_layer_dict = {}
        fc_block = None
        final_layer = None
        activate_fc_block = True if pred_type=='mu' else False
        
        # base (for dynamic)
        if (base_x is not None) and (timemachine_x is None) and (api_x is None):
            x_list = [base_x]
            embedding_layer_dict[self.base_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.base_fc_block
            final_layer = self.base_mu_layer if pred_type == 'mu' else self.base_std_layer 

        # timemachine (for api_call)
        elif (base_x is not None) and (timemachine_x is not None) and (api_x is None):
            x_list=[base_x, timemachine_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.timemachine_fc_block
            final_layer = self.timemachine_mu_layer if pred_type == 'mu' else self.timemachine_std_layer

        # api_call (for leaving_time)
        elif (base_x is not None) and (timemachine_x is not None) and (api_x is not None):
            x_list=[base_x, timemachine_x, api_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = False
            embedding_layer_dict[self.api_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.api_fc_block
            final_layer = self.api_mu_layer if pred_type == 'mu' else self.api_std_layer
        
        return self.forward_structure(x_list, embedding_layer_dict, fc_block, final_layer=final_layer, activate_fc_block=activate_fc_block)
    
    # forward_pred_type
    def forward_pred_type(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu'):
        output = self.forward_freeze_forward(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type=pred_type)

        # dynamic case distance based prediction
        if (base_x is not None) and (timemachine_x is None) and (api_x is None):
            distant_output = self.distance_embedding_layer(base_x[..., -4:-2], base_x[..., -2:])
            output = output * distant_output
        
        if pred_type == 'mu':
            return output.mean(dim=-1, keepdims=True)
        if pred_type == 'std':
            return output.std(dim=-1, keepdims=True)

    # forward
    def forward(self, base_x=None, timemachine_x=None, api_x=None, pred_type=None):
        if pred_type is None:
            mu = self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='mu')
            std = self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='std')
            return (mu, std)
        
        elif pred_type == 'mu':
            return self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='mu')

        elif pred_type =='std':
            return self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='std')




# BaroNowTransformer
class BaroNowTransformer_V3(nn.Module):
    def __init__(self, output_dim=1, n_layers=3, n_output=10, n_head=8, transformer_n_layers=2,
                base_embedding_params={}, timemachine_embedding_params={}, api_embedding_params={}):
        super().__init__()

        # embedding_layer
        self.base_embedding_layer = CombinedEmbedding(**base_embedding_params)
        self.distance_embedding_layer = SpatialEmbedding(embed_dim=None, coord_embed_dim=None, grid_embed_dim=None, periodic_embed_dim=None, 
                                                    relative=False, euclidean_dist=True, angle=False)
        self.timemahcine_embedding_layer = CombinedEmbedding(**timemachine_embedding_params)
        self.api_embedding_layer = CombinedEmbedding(**api_embedding_params)

        # embedding_dimension
        self.base_embed_dim = self.base_embedding_layer.embed_dim
        self.timemahcine_embed_dim = self.base_embed_dim + self.timemahcine_embedding_layer.embed_dim
        self.api_embed_dim = self.timemahcine_embed_dim + self.api_embedding_layer.embed_dim
        out_dim = output_dim*n_output
        
        # final_layer
        self.base_block = nn.ModuleDict({})
        self.base_block['transforemr_encoder_layer'] = nn.TransformerEncoderLayer(d_model=self.base_embed_dim, nhead=n_head, dim_feedforward=self.base_embed_dim, batch_first=True)
        self.base_block['transforemr_encoder'] = nn.TransformerEncoder(self.base_block['transforemr_encoder_layer'], num_layers=transformer_n_layers)
        self.base_mu_layer = FeedForwardBlock(self.base_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.base_std_layer = FeedForwardBlock(self.base_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        self.timemachine_block = nn.ModuleDict({})
        self.timemachine_block['transforemr_encoder_layer'] = nn.TransformerEncoderLayer(d_model=self.timemahcine_embed_dim, nhead=n_head, dim_feedforward=self.timemahcine_embed_dim, batch_first=True)
        self.timemachine_block['transforemr_encoder'] = nn.TransformerEncoder(self.timemachine_block['transforemr_encoder_layer'], num_layers=transformer_n_layers)
        self.timemachine_mu_layer = FeedForwardBlock(self.timemahcine_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.timemachine_std_layer = FeedForwardBlock(self.timemahcine_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        self.api_block = nn.ModuleDict({})
        self.api_block['transforemr_encoder_layer'] = nn.TransformerEncoderLayer(d_model=self.api_embed_dim, nhead=n_head, dim_feedforward=self.api_embed_dim, batch_first=True)
        self.api_block['transforemr_encoder'] = nn.TransformerEncoder(self.api_block['transforemr_encoder_layer'], num_layers=transformer_n_layers)
        self.api_mu_layer = FeedForwardBlock(self.api_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)
        self.api_std_layer = FeedForwardBlock(self.api_embed_dim, out_dim, activation=False, batchNorm=False, dropout=0)

        # params
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_output = n_output

    # weight_activate
    def weight_activate(self, activate=None, deactivate=None, all_activate=None):
        # all weight activate
        if (all_activate is True) or (all_activate is False):
            for param in self.parameters():
                param.requires_grad = all_activate

        if (activate is not None) and isinstance(activate, nn.Module):
            for name, param in activate.named_parameters():
                param.requires_grad = True

        if (deactivate is not None)and isinstance(deactivate, nn.Module):
            for name, param in deactivate.named_parameters():
                param.requires_grad = False

    # requires_grad_status
    def requires_grad_status(self, layer=None, verbose=0):
        if layer is None:
            layer = self

        require_grad_list = {}
        if isinstance(layer, nn.Module):
            for name, param in layer.named_parameters():
                require_grad_list[name] = param.requires_grad
                
                if verbose > 0:
                    print(f"({name}) {param.requires_grad}")
            return require_grad_list
    
    # forward_base
    def forward_structure(self, x_list=[], embedding_layer_dict={}, fc_block=None, final_layer=None, activate_fc_block=True):
        # weight freezing / activate
        self.weight_activate(all_activate=False)
        if activate_fc_block:
            self.weight_activate(activate=fc_block)
        self.weight_activate(activate=final_layer)

        embed_output_list = []
        for x_input, (layer, require_grad) in zip(x_list, embedding_layer_dict.items()):
            if require_grad is True:
                self.weight_activate(activate=layer)
            embed_output_list.append(layer(x_input))

        # embedding concat
        x_hidden = torch.cat(embed_output_list, dim=-1)

        # transformer_encoder & fc_layer
        x_hidden = fc_block['transforemr_encoder'](x_hidden)
        return final_layer(x_hidden)
       
    # forward_freeze_forward
    def forward_freeze_forward(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu'):
        x_list = []
        embedding_layer_dict = {}
        fc_block = None
        final_layer = None
        activate_fc_block = True if pred_type=='mu' else False
        if (base_x is not None) and (timemachine_x is None) and (api_x is None):
            x_list = [base_x]
            embedding_layer_dict[self.base_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.base_block
            final_layer = self.base_mu_layer if pred_type == 'mu' else self.base_std_layer

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is None):
            x_list=[base_x, timemachine_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.timemachine_block
            final_layer = self.timemachine_mu_layer if pred_type == 'mu' else self.timemachine_std_layer

        elif (base_x is not None) and (timemachine_x is not None) and (api_x is not None):
            x_list=[base_x, timemachine_x, api_x]
            embedding_layer_dict[self.base_embedding_layer] = False
            embedding_layer_dict[self.timemahcine_embedding_layer] = False
            embedding_layer_dict[self.api_embedding_layer] = True if pred_type=='mu' else False
            fc_block = self.api_block
            final_layer = self.api_mu_layer if pred_type == 'mu' else self.api_std_layer
        
        return self.forward_structure(x_list, embedding_layer_dict, fc_block, final_layer=final_layer, activate_fc_block=activate_fc_block)
    
    # forward_pred_type
    def forward_pred_type(self, base_x=None, timemachine_x=None, api_x=None, pred_type='mu'):
        output = self.forward_freeze_forward(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type=pred_type)

        # dynamic case distance based prediction
        if (base_x is not None) and (timemachine_x is None) and (api_x is None):
            distant_output = self.distance_embedding_layer(base_x[..., -4:-2], base_x[..., -2:])
            output = output * distant_output

        if pred_type == 'mu':
            return output.mean(dim=-1, keepdims=True)
        if pred_type == 'std':
            return output.std(dim=-1, keepdims=True)

    # forward
    def forward(self, base_x=None, timemachine_x=None, api_x=None, pred_type=None):
        if pred_type is None:
            mu = self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='mu')
            std = self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='std')
            return (mu, std)
        
        elif pred_type == 'mu':
            return self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='mu')

        elif pred_type =='std':
            return self.forward_pred_type(base_x=base_x, timemachine_x=timemachine_x, api_x=api_x, pred_type='std')
        












###########################################################################################################
###########################################################################################################
###########################################################################################################


class PuclicDynamic(nn.Module):
    def __init__(self, spatial_embed = 16, spatial_hidden_dim=32, output_dim=10):
       super().__init__()
       self.spatial_embedding_layer = CombinedEmbedding(spatial_emb_dim=spatial_embed)
       self.sptial_fc_layer = nn.Sequential(
           nn.Linear(self.spatial_embedding_layer.embed_dim, spatial_hidden_dim)
           ,nn.ReLU()
           ,nn.Linear(spatial_hidden_dim, spatial_hidden_dim)
       )
       self.spatial_mu_layer = nn.Linear(spatial_hidden_dim, output_dim)
       self.spatial_std_layer = nn.Linear(spatial_hidden_dim, output_dim)

       self.distance_embedding_layer = SpatialEmbedding(embed_dim=None, coord_embed_dim=None, grid_embed_dim=None, periodic_embed_dim=None, 
                            relative=False, euclidean_dist=True, angle=False)

    # weight_activate
    def weight_activate(self, activate=None, deactivate=None, all_activate=None):
        # all weight activate
        if (all_activate is True) or (all_activate is False):
            for param in self.parameters():
                param.requires_grad = all_activate

        if (activate is not None) and isinstance(activate, nn.Module):
            for name, param in activate.named_parameters():
                param.requires_grad = True

        if (deactivate is not None)and isinstance(deactivate, nn.Module):
            for name, param in deactivate.named_parameters():
                param.requires_grad = False

    # requires_grad_status
    def requires_grad_status(self, layer=None, verbose=0):
        if layer is None:
            layer = self

        require_grad_list = {}
        if isinstance(layer, nn.Module):
            for name, param in layer.named_parameters():
                require_grad_list[name] = param.requires_grad
                
                if verbose > 0:
                    print(f"({name}) {param.requires_grad}")
            return require_grad_list
    
    def forward_mu(self, x):
        self.weight_activate(all_activate=True)
        sp_embed_ouput = self.sptial_fc_layer( self.spatial_embedding_layer(x) )
        mu_sample = self.spatial_mu_layer(sp_embed_ouput) * self.distance_embedding_layer(x[..., :2], x[..., 2:])
        return mu_sample.mean(dim=-1, keepdims=True)
    
    def forward_std(self, x):
        self.weight_activate(all_activate=False)
        self.weight_activate(activate=self.spatial_std_layer)

        sp_embed_ouput = self.sptial_fc_layer( self.spatial_embedding_layer(x) )
        std_sample = self.spatial_std_layer(sp_embed_ouput) * self.distance_embedding_layer(x[..., :2], x[..., 2:])
        return std_sample.std(dim=-1, keepdims=True)

    def forward(self, x, pred_type=None):
        if pred_type is None:
            mu = self.forward_mu(x)
            std = self.forward_std(x)
            return (mu, std)
        
        elif pred_type == 'mu':
            return self.forward_mu(x)
    
        elif pred_type =='std':
            return self.forward_std(x)



























