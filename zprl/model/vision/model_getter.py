import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer. Borrowed from robomimic.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=32,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={})'.format(self._num_kp)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, x):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        x = self.nets(x) if self.nets is not None else x

        # (B,k,h,w) -> (B*k,h*w)
        x = x.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(x, dim=-1)  # (B*k,h*w)
        # (1, h*w) * (B*k, h*w) -> (B*k, 1)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        return feature_keypoints.flatten(start_dim=1)  # [B, K*2]


class SpatialLearnedEmbeddings(nn.Module):
    """
    Spatial Learned Embeddings Layer. This layer learns a set of spatial embeddings
    for keypoints, which can be used in place of spatial softmax.

    Args:
        input_shape (tuple): shape of the input feature (C, H, W)
        num_spatial_blocks (int): number of keypoints to learn embeddings for
        bottleneck_dim (int): dimension of the bottleneck layer
    """
    def __init__(self, input_shape, num_spatial_blocks=8, bottleneck_dim=256):
        super(SpatialLearnedEmbeddings, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim

        self.embeddings = nn.Parameter(
            torch.empty(self._in_c, self._in_h, self._in_w, self.num_spatial_blocks))
        self.bottleneck = nn.Linear(
            self._in_c * self.num_spatial_blocks,
            self.bottleneck_dim
        )
        self.ln = nn.LayerNorm(self.bottleneck_dim)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self._in_c
        variance = 1.0 / fan_in
        std = math.sqrt(variance) / .87962566103423978
        with torch.no_grad():
            nn.init.trunc_normal_(self.embeddings, std=std)
    
    def forward(self, x):
        x = torch.einsum('bchw,chwf->bcf', x, self.embeddings)
        x = x.flatten(start_dim=1)  # [B, C * num_spatial_blocks]
        x = self.bottleneck(x)
        x = self.ln(x)
        return F.tanh(x)  # [B, bottleneck_dim]
        

class ResNet18Pooling(nn.Module):
    def __init__(self, input_shape, weights=None, **kwargs):
        super(ResNet18Pooling, self).__init__()
        # 1. encode input (images) using convolutional layers
        ## ResNet18 layers: conv, bn, relu, maxpool(64,x/2), layer1 (64,x/4)
        ## layer2 (128,x/8), layer3(256,x/16), layer4(512,x/32), avgpool(512,), fc
        assert (
            len(input_shape) == 3
        ), "[error] input shape of resnet should be (C, H, W)"

        remove_layer_num = 2 # 2, 3, 4, TBD
        layers = list(torchvision.models.resnet18(weights=weights).children())[
            :-remove_layer_num
        ]
        self.remove_layer_num = remove_layer_num
        self.resnet_base = nn.Sequential(*layers)

        # 2. add pooling layer
        x = torch.zeros(1, *input_shape)
        y = self.resnet_base(x) # now should be (B,512,h/32,w/32)
        output_shape = y.shape

        pooling_type = kwargs.get('pooling_type', 'spatial_learned_embeddings')
        if pooling_type == 'spatial_softmax':
            num_kp = kwargs.get("num_kp", 32)
            self.pooling = SpatialSoftmax(
                input_shape=output_shape[1:], num_kp=num_kp
            )
        elif pooling_type == 'spatial_learned_embeddings':
            num_spatial_blocks = kwargs.get("num_spatial_blocks", 8)
            bottleneck_dim = kwargs.get("bottleneck_dim", 256)
            self.pooling = SpatialLearnedEmbeddings(
                input_shape=output_shape[1:],
                num_spatial_blocks=num_spatial_blocks,
                bottleneck_dim=bottleneck_dim
            )
        elif pooling_type == 'avg':
            bottleneck_dim = kwargs.get("bottleneck_dim", None)
            bottleneck = nn.Identity()
            if bottleneck_dim is not None:
                bottleneck = nn.Sequential(
                    nn.Linear(output_shape[1], bottleneck_dim),
                    nn.LayerNorm(bottleneck_dim),
                    nn.Tanh()
                )

            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                bottleneck
            )
        else:
            raise ValueError(
                f"Unsupported pooling type: {pooling_type}. "
                "Supported types: spatial_softmax, spatial_learned_embeddings."
            )
        self.output_shape = self.pooling(y).shape[-1]

    def forward(self, x):
        x = self.resnet_base(x)
        x = self.pooling(x)
        return x


class TimmPooling(nn.Module):
    def __init__(self, model_name, input_shape, **kwargs):
        super(TimmPooling, self).__init__()
        assert (
            len(input_shape) == 3
        ), "[error] input shape of timm model should be (C, H, W)"

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=input_shape[0]
        )

        x = torch.zeros(1, *input_shape)
        y = self.backbone.forward_features(x)
        output_shape = y.shape

        num_spatial_blocks = kwargs.get("num_spatial_blocks", 8)
        bottleneck_dim = kwargs.get("bottleneck_dim", 256)
        self.pooling = SpatialLearnedEmbeddings(
            input_shape=output_shape[1:],
            num_spatial_blocks=num_spatial_blocks,
            bottleneck_dim=bottleneck_dim
        )
        self.output_shape = self.pooling(y).shape[-1]
        print(f"Feature shape: {output_shape}")
        print(f"Backbone params: {sum(p.numel() for p in self.backbone.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Pooling params: {sum(p.numel() for p in self.pooling.parameters() if p.requires_grad) / 1e6:.2f}M")

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pooling(x)
        return x


if __name__ == "__main__":
    # Test Resnet w/ Spatial Projection
    input_shape = (3, 120, 160)
    sample = torch.randn(16, *input_shape)
    pooling_kwargs = {
        "pooling_type": "spatial_softmax",
        "num_kp": 32,  # Number of keypoints to project
    }
    model1 = ResNet18Pooling(input_shape, weights=None, **pooling_kwargs)
    
    output = model1(sample)
    print(output.shape, model1.output_shape)
    # Expected output shape: (16, 64) for num_kp=32

    pooling_kwargs = {
        "pooling_type": "spatial_learned_embeddings",
        "num_spatial_blocks": 8,
        "bottleneck_dim": 256,
    }
    model2 = ResNet18Pooling(input_shape, weights=None, **pooling_kwargs)
    output = model2(sample)
    print(output.shape, model2.output_shape)
    # Expected output shape: (16, 256) for bottleneck_dim=256

    pooling_kwargs = {
        "pooling_type": "avg",
        "bottleneck_dim": 256,
    }
    model3 = ResNet18Pooling(input_shape, weights=None, **pooling_kwargs)
    output = model3(sample)
    print(output.shape, model3.output_shape)
    # Expected output shape: (16, 512) for avg pooling

    # Test resnet34
    input_shape = (3, 224, 224)
    sample = torch.randn(16, *input_shape)
    pooling_kwargs = {
        "num_spatial_blocks": 8,
        "bottleneck_dim": 256,
    }
    model4 = TimmPooling("resnet34", input_shape, **pooling_kwargs)
    output = model4(sample)
    print(output.shape, model4.output_shape)
    # Expected output shape: (16, 256) for Timm convnext pooling
