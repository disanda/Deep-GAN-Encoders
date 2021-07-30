#Conv类中定义了lr_equalization_coef属性，该属性为True, 则会进行特殊的初始化，且在optimizer中更新

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np


class Bool:
    def __init__(self):
        self.value = False

    def __bool__(self):
        return self.value
    __nonzero__ = __bool__

    def set(self, value):
        self.value = value


use_implicit_lreq = Bool()
use_implicit_lreq.set(True)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


def make_tuple(x, n):
    if is_sequence(x):
        return x
    return tuple([x for _ in range(n)])


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gain=np.sqrt(2.0), lrmul=1.0, implicit_lreq=use_implicit_lreq):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.std = 0
        self.gain = gain
        self.lrmul = lrmul
        self.implicit_lreq = implicit_lreq
        self.reset_parameters()

    def reset_parameters(self):
        self.std = self.gain / np.sqrt(self.in_features) * self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
            setattr(self.weight, 'lr_equalization_coef', self.std)
            if self.bias is not None:
                setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input):
        if not self.implicit_lreq:
            bias = self.bias
            if bias is not None:
                bias = bias * self.lrmul
            return F.linear(input, self.weight * self.std, bias)
        else:
            return F.linear(input, self.weight, self.bias)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transpose=False, transform_kernel=False, lrmul=1.0,
                 implicit_lreq=use_implicit_lreq):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_tuple(kernel_size, 2)
        self.stride = make_tuple(stride, 2)
        self.padding = make_tuple(padding, 2)
        self.output_padding = make_tuple(output_padding, 2)
        self.dilation = make_tuple(dilation, 2)
        self.groups = groups
        self.gain = gain
        self.lrmul = lrmul
        self.transpose = transpose
        self.fan_in = np.prod(self.kernel_size) * in_channels // groups # k*k*c
        self.transform_kernel = transform_kernel
        if transpose:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *self.kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.std = 0
        self.implicit_lreq = implicit_lreq
        self.reset_parameters()

    def reset_parameters(self):
        self.std = self.gain / np.sqrt(self.fan_in) #这里重置了std
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
            setattr(self.weight, 'lr_equalization_coef', self.std)
            if self.bias is not None:
                setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x):
        if self.transpose:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1, 1, 1), mode='constant')
                w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv_transpose2d(x, w * self.std, bias, stride=self.stride,
                                          padding=self.padding, output_padding=self.output_padding,
                                          dilation=self.dilation, groups=self.groups)
            else:
                return F.conv_transpose2d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                          output_padding=self.output_padding, dilation=self.dilation,
                                          groups=self.groups)
        else:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1, 1, 1), mode='constant')
                w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv2d(x, w * self.std, bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)
            else:
                return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)


class ConvTranspose2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transform_kernel=False, lrmul=1.0, implicit_lreq=use_implicit_lreq):
        super(ConvTranspose2d, self).__init__(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              output_padding=output_padding,
                                              dilation=dilation,
                                              groups=groups,
                                              bias=bias,
                                              gain=gain,
                                              transpose=True,
                                              transform_kernel=transform_kernel,
                                              lrmul=lrmul,
                                              implicit_lreq=implicit_lreq)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0), transpose=False):
        super(SeparableConv2d, self).__init__()
        self.spatial_conv = Conv2d(in_channels, in_channels, kernel_size, stride, padding, output_padding, dilation,
                                   in_channels, False, 1, transpose)
        self.channel_conv = Conv2d(in_channels, out_channels, 1, bias, 1, gain=gain)

    def forward(self, x):
        return self.channel_conv(self.spatial_conv(x))


class SeparableConvTranspose2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0)):
        super(SeparableConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, dilation, bias, gain, True)

# class _NormBase(Module):
#     """Common base of _InstanceNorm and _BatchNorm"""

#     _version = 2
#     __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
#     num_features: int
#     eps: float
#     momentum: float
#     affine: bool
#     track_running_stats: bool
#     # WARNING: weight and bias purposely not defined here.
#     # See https://github.com/pytorch/pytorch/issues/39670

#     def __init__(
#         self,
#         num_features: int,
#         eps: float = 1e-5,
#         momentum: float = 0.1,
#         affine: bool = True,
#         track_running_stats: bool = True,
#         device=None,
#         dtype=None
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(_NormBase, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
#             self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
#         else:
#             self.register_parameter("weight", None)
#             self.register_parameter("bias", None)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
#             self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
#             self.running_mean: Optional[Tensor]
#             self.running_var: Optional[Tensor]
#             self.register_buffer('num_batches_tracked',
#                                  torch.tensor(0, dtype=torch.long,
#                                               **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
#         else:
#             self.register_buffer("running_mean", None)
#             self.register_buffer("running_var", None)
#             self.register_buffer("num_batches_tracked", None)
#         self.reset_parameters()

#     def reset_running_stats(self) -> None:
#         if self.track_running_stats:
#             # running_mean/running_var/num_batches... are registered at runtime depending
#             # if self.track_running_stats is on
#             self.running_mean.zero_()  # type: ignore[union-attr]
#             self.running_var.fill_(1)  # type: ignore[union-attr]
#             self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

#     def reset_parameters(self) -> None:
#         self.reset_running_stats()
#         if self.affine:
#             init.ones_(self.weight)
#             init.zeros_(self.bias)

#     def _check_input_dim(self, input):
#         raise NotImplementedError

#     def extra_repr(self):
#         return (
#             "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
#             "track_running_stats={track_running_stats}".format(**self.__dict__)
#         )

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         version = local_metadata.get("version", None)

#         if (version is None or version < 2) and self.track_running_stats:
#             # at version 2: added num_batches_tracked buffer
#             #               this should have a default value of 0
#             num_batches_tracked_key = prefix + "num_batches_tracked"
#             if num_batches_tracked_key not in state_dict:
#                 state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

#         super(_NormBase, self)._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )

# class _InstanceNorm(_NormBase):
#     def __init__(
#         self,
#         num_features: int,
#         eps: float = 1e-5,
#         momentum: float = 0.1,
#         affine: bool = False,
#         track_running_stats: bool = False
#     ) -> None:
#         super(_InstanceNorm, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)

#     def _check_input_dim(self, input):
#         raise NotImplementedError

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         version = local_metadata.get('version', None)
#         # at version 1: removed running_mean and running_var when
#         # track_running_stats=False (default)
#         if version is None and not self.track_running_stats:
#             running_stats_keys = []
#             for name in ('running_mean', 'running_var'):
#                 key = prefix + name
#                 if key in state_dict:
#                     running_stats_keys.append(key)
#             if len(running_stats_keys) > 0:
#                 error_msgs.append(
#                     'Unexpected running stats buffer(s) {names} for {klass} '
#                     'with track_running_stats=False. If state_dict is a '
#                     'checkpoint saved before 0.4.0, this may be expected '
#                     'because {klass} does not track running stats by default '
#                     'since 0.4.0. Please remove these keys from state_dict. If '
#                     'the running stats are actually needed, instead set '
#                     'track_running_stats=True in {klass} to enable them. See '
#                     'the documentation of {klass} for details.'
#                     .format(names=" and ".join('"{}"'.format(k) for k in running_stats_keys),
#                             klass=self.__class__.__name__))
#                 for key in running_stats_keys:
#                     state_dict.pop(key)

#         super(_InstanceNorm, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)

#     def forward(self, input: Tensor) -> Tensor:
#         self._check_input_dim(input)

#         assert self.running_mean is None or isinstance(self.running_mean, Tensor)
#         assert self.running_var is None or isinstance(self.running_var, Tensor)
#         return F.instance_norm(
#             input, self.running_mean, self.running_var, self.weight, self.bias,
#             self.training or not self.track_running_stats, self.momentum, self.eps)

# class InstanceNorm2d(_InstanceNorm):
#     r"""Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs
#     with additional channel dimension) as described in the paper
#     `Instance Normalization: The Missing Ingredient for Fast Stylization
#     <https://arxiv.org/abs/1607.08022>`__.

#     .. math::

#         y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

#     The mean and standard-deviation are calculated per-dimension separately
#     for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
#     of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.
#     The standard-deviation is calculated via the biased estimator, equivalent to
#     `torch.var(input, unbiased=False)`.

#     By default, this layer uses instance statistics computed from input data in
#     both training and evaluation modes.

#     If :attr:`track_running_stats` is set to ``True``, during training this
#     layer keeps running estimates of its computed mean and variance, which are
#     then used for normalization during evaluation. The running estimates are
#     kept with a default :attr:`momentum` of 0.1.

#     .. note::
#         This :attr:`momentum` argument is different from one used in optimizer
#         classes and the conventional notion of momentum. Mathematically, the
#         update rule for running statistics here is
#         :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
#         where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
#         new observed value.

#     .. note::
#         :class:`InstanceNorm2d` and :class:`LayerNorm` are very similar, but
#         have some subtle differences. :class:`InstanceNorm2d` is applied
#         on each channel of channeled data like RGB images, but
#         :class:`LayerNorm` is usually applied on entire sample and often in NLP
#         tasks. Additionally, :class:`LayerNorm` applies elementwise affine
#         transform, while :class:`InstanceNorm2d` usually don't apply affine
#         transform.

#     Args:
#         num_features: :math:`C` from an expected input of size
#             :math:`(N, C, H, W)`
#         eps: a value added to the denominator for numerical stability. Default: 1e-5
#         momentum: the value used for the running_mean and running_var computation. Default: 0.1
#         affine: a boolean value that when set to ``True``, this module has
#             learnable affine parameters, initialized the same way as done for batch normalization.
#             Default: ``False``.
#         track_running_stats: a boolean value that when set to ``True``, this
#             module tracks the running mean and variance, and when set to ``False``,
#             this module does not track such statistics and always uses batch
#             statistics in both training and eval modes. Default: ``False``

#     Shape:
#         - Input: :math:`(N, C, H, W)`
#         - Output: :math:`(N, C, H, W)` (same shape as input)

#     Examples::

#         >>> # Without Learnable Parameters
#         >>> m = nn.InstanceNorm2d(100)
#         >>> # With Learnable Parameters
#         >>> m = nn.InstanceNorm2d(100, affine=True)
#         >>> input = torch.randn(20, 100, 35, 45)
#         >>> output = m(input)
#     """

#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))


# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================