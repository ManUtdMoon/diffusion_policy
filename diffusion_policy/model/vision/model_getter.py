import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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

        return feature_keypoints
    

class ResNet18SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=32, weights=None):
        super(ResNet18SpatialSoftmax, self).__init__()
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

        # 2. add spatial softmax layer
        x = torch.zeros(1, *input_shape)
        y = self.resnet_base(x) # now should be (B,512,h/32,w/32)
        output_shape = y.shape
        self.pooling = SpatialSoftmax(
            input_shape=output_shape[1:], num_kp=num_kp
        )
        self.output_shape = self.pooling(y).flatten(start_dim=1).shape[-1]

    def forward(self, x):
        x = self.resnet_base(x)
        x = self.pooling(x)
        return x.flatten(start_dim=1)


if __name__ == "__main__":
    # Test Resnet w/ Spatial Projection
    input_shape = (3, 120, 160)
    num_kp = 32
    model = ResNet18SpatialSoftmax(input_shape, num_kp, weights=None)

    sample = torch.randn(16, *input_shape)
    output = model(sample)
    print(output.shape, model.output_shape)
    # Expected output shape: (16, 64) for num_kp=32
