import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).

    Args:
        normalized_shape (int): Channel size
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True) # computes the mean across channels, (N, 1, H, W)
            s = (x - u).pow(2).mean(1, keepdim=True) # variance across channels
            x = (x - u) / torch.sqrt(s + self.eps) # normalize (N, C, H, W)

            # If self.bias has shape (C,), then self.bias[:, None, None] reshapes it to (C, 1, 1). 
            # This allows it to be broadcast across the height and width dimensions when added to or 
            # multiplied with a tensor of shape (N, C, H, W).
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv (depthwise) -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv (depthwise) -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        # dim is the number of input channels

        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # (C, 4C)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) # (4C, C)

        # A kind of Tensor that is to be considered a module parameter.
        # Parameters are ~torch.Tensor subclasses, that have a very special property 
        # when used with Module.
        # When they're assigned as Module attributes they are automatically added to 
        # the list of its parameters, and will appear e.g. in ~Module.parameters iterator.
        # (C,)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        # gamma is a learnable scaling parameter (a vector of shape (C,)) used to scale the output 
        # of the block before adding it back to the input (residual connection). 
        # It allows the network to control the contribution of the block’s output, and is initialized 
        # to a small value (layer_scale_init_value) so the block starts as an identity mapping and can 
        # learn to scale up during training.

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x) # (N, C, H, W)

        # a torch function that changes the order of dimensions of a tensor
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x) # (N, H, W, C)
        x = self.pwconv1(x) # (N, H, W, C) @ (C, 4C) -> (N, H, W, 4C)
        x = self.act(x) # (N, H, W, 4C)
        x = self.pwconv2(x) # (N, H, W, 4C) @ (4C, C) -> (N, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x # broadcasting (C,) -> (N, H, W, C)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        # stochastic depth might drop the entire block, so we have identity connection only
        x = input + self.drop_path(x) # residual connection
        return x
    
    
class ConvNeXt(nn.Module):
    r""" 
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=6, 
                 depths=[3, 6, 6, 3], dims=[96, 192, 384, 768], drop_path_rate=0.2, 
                 layer_scale_init_value=1e-6, head_init_scale=1., stage0=False,
                 ):
        super().__init__()

        self.stage0 = stage0

        # nn.ModuleList() is a PyTorch container that holds submodules (layers) just like a regular Python list, 
        # but registers them as part of the parent module (ConvNeXt). 
        # This means all modules inside a ModuleList will be properly tracked by PyTorch for parameter management, 
        # saving/loading, and moving to devices (to cuda, etc.).
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        # nn.Sequential is a PyTorch container that allows you to stack layers (modules) together in the order they are defined. 
        # When you pass input to an nn.Sequential, it will be fed through each layer in sequence. 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), # (N, 3, H, W) -> (N, 96, H/4, W/4)
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2), # (N, dim[i], H, W) -> (N, dim[i+1], H/2, W/2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        # 18 drop path rates, one for each block, could be just 18 0's. Stochastic depth decay rule
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # [0, 0.0118..., ..., 0.2]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                # unpacking operator * for iterables, ** for dictionaries
                # it is used to unpack the list of Block modules into individual arguments for nn.Sequential
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        if self.stage0:
            self.head = nn.Linear(dims[-1], num_classes) # (768, 6)

            # multiplies all elements of the tensor by head_init_scale (a scalar).
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        # apply is a method of nn.Module in PyTorch. It recursively applies a given function to every submodule/layers 
        # (including itself) in the module hierarchy. This is commonly used for custom initialization.
        # apply method works hierarchically: it applies the function to the top-level module, then recursively to all 
        # submodules (including layers inside ConvNeXt, Block, etc.). This ensures that every layer, no matter how deeply nested, 
        # is processed.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # m is a module (submodule, layer) in the network
        # this function must have this signature to be used with apply
        if isinstance(m, (nn.Conv2d, nn.Linear)):

            # initializes the weights of the layer m with values drawn from a truncated normal (Gaussian) distribution 
            # with a standard deviation of 0.02. "Truncated" means values far from the mean are cut off, so the weights 
            # stay within a certain range.
            # Could've just used the by default Xavier initialization since its based on normal too.
            trunc_normal_(m.weight, std=.02)

            # sets all elements of the bias tensor m.bias to zero.
            # Which is unnecessary since by pytorch default biases are initialized to zero for Linear and Conv2d layers.
            # For Conv2d, bias is a 1D tensor of shape (channel_out,) that gets broadcasted across all spatial positions.
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is (N, 3, 224, 224)
        for i in range(4): # for each of the 4 stages
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        x = self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

        if self.stage0:
            return self.head(x) # (N, 6)
        else:
            return x # (N, C)