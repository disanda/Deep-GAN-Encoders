import torch
from torch.nn.modules.utils import _pair
from numpy import sqrt,prod
from torch.nn.functional import conv2d, conv_transpose2d, linear, interpolate
from torch.nn import Conv2d, ConvTranspose2d, LeakyReLU, Embedding, AvgPool2d

# extending Conv2D and Deconv2D layers for equalized learning rate logic
class _equalized_conv2d(torch.nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        super().__init__()
        # define the weight and bias if to be used
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(c_out, c_in, *_pair(k_size))))
        self.use_bias = bias
        self.stride = stride
        self.pad = pad
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)
    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        return conv2d(input=x, weight=self.weight * self.scale, bias=self.bias if self.use_bias else None, stride=self.stride, padding=self.pad)
    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))

class _equalized_deconv2d(torch.nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        super().__init__()
        # define the weight and bias if to be used
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(c_in, c_out, *_pair(k_size))))
        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale the weight on runtime
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)
    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))

class _equalized_linear(torch.nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param c_in: number of input channels
            :param c_out: number of output channels
            :param bias: whether to use bias with the linear layer
    """
    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        super().__init__()

        self.weight = th.nn.Parameter(th.nn.init.normal_(
            th.empty(c_out, c_in)
        ))

        self.use_bias = bias

        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        return linear(x, self.weight * self.scale, self.bias if self.use_bias else None)


# ----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ----------------------------------------------------------------------------
class PixelwiseNorm(torch.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================
class GenInitialBlock(torch.nn.Module):
    """ Module implementing the initial block of the input """
    def __init__(self, in_channels, use_eql):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use equalized learning rate
        """
        super().__init__()
        if use_eql:
            self.conv_1 = _equalized_deconv2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (3, 3),pad=1, bias=True)
        else:
            self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()
        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)
    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        # apply pixel norm
        y = self.pixNorm(y)
        return y

class GenGeneralConvBlock(torch.nn.Module):
    """ Module implementing a general convolutional block """
    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use equalized learning rate
        """
        super().__init__()
        self.upsample = lambda x: interpolate(x, scale_factor=2)
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, out_channels, (3, 3),pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(out_channels, out_channels, (3, 3),pad=1, bias=True)
        else:
            self.conv_1 = Conv2d(in_channels, out_channels, (3, 3),padding=1, bias=True)
            self.conv_2 = Conv2d(out_channels, out_channels, (3, 3),padding=1, bias=True)
        # Pixelwise feature vector normalization operation
        self.pixNorm = PixelwiseNorm()
        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)
    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.upsample(x)
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))
        return y

class MinibatchStdDev(torch.nn.Module):
    #Minibatch standard deviation layer for the discriminator
    def __init__(self):
        super().__init__()
    def forward(self, x, alpha=1e-8):
        """
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)
        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)
        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)
        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)
        # return the computed values:
        return y

class DisFinalBlock(torch.nn.Module):
    """ Final block for the Discriminator """
    def __init__(self, in_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """
        super().__init__()
        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)
        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)
    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)
        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation
        # flatten the output raw discriminator scores
        return y.view(-1)

class DisGeneralConvBlock(torch.nn.Module):
    """ General block in the discriminator  """
    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """
        super().__init__()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3), pad=1, bias=True)
        else:
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.downSampler = AvgPool2d(2)
        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)
    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)
        return y

class ConDisFinalBlock(torch.nn.Module):
    """ Final block for the Conditional Discriminator
        Uses the Projection mechanism from the paper -> https://arxiv.org/pdf/1802.05637.pdf
    """
    def __init__(self, in_channels, num_classes, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param num_classes: number of classes for conditional discrimination
        :param use_eql: whether to use equalized learning rate
        """
        super().__init__()
        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)
        # we also need an embedding matrix for the label vectors
        self.label_embedder = Embedding(num_classes, in_channels, max_norm=1)
        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)
    def forward(self, x, labels):
        """
        forward pass of the FinalBlock
        :param x: input
        :param labels: samples' labels for conditional discrimination
                       Note that these are pure integer labels [Batch_size x 1]
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)  # [B x C x 4 x 4]
        # perform the forward pass
        y = self.lrelu(self.conv_1(y))  # [B x C x 4 x 4]
        # obtain the computed features
        y = self.lrelu(self.conv_2(y))  # [B x C x 1 x 1]
        # embed the labels
        labels = self.label_embedder(labels)  # [B x C]
        # compute the inner product with the label embeddings
        y_ = torch.squeeze(th.squeeze(y, dim=-1), dim=-1)  # [B x C]
        projection_scores = (y_ * labels).sum(dim=-1)  # [B]
        # normal discrimination score
        y = self.lrelu(self.conv_3(y))  # This layer has linear activation
        # calculate the total score
        final_score = y.view(-1) + projection_scores
        # return the output raw discriminator scores
        return final_score