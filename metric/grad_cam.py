# -*- coding: utf-8 -*-
# 不同库的图片格式转换: [h,w,c] -> [c,h,w] https://www.jianshu.com/p/dd08418c306f
# 这一版是每一张图片分别传梯度
import numpy as np
import cv2
import torchvision
import torch
import os

class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]
        print("gradient shape:{}".format(output_grad[0].size()))

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name: #对应层使用hook
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))
                #self.handlers.append(module.register_full_backward_hook(self._get_grads_hook))
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def call_per_img(self, inputs, index):
        """
        :param inputs: [N,3,H,W]
        :param index: class id
        :return:
        #  每张图片都计算一次梯度
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [N,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy(),axis=1) # [N]
        #index_max = max(index,key=index.count) #选择出现最多的index
        #index_max=np.argmax(np.bincount(index))
        cam_all=np.random.randn(inputs.size(0),inputs.size(2),inputs.size(3))
        for i,j in enumerate(index):
            target = output[i][j]
            target.backward(retain_graph=True)
            gradient = self.gradient[i].cpu().data.numpy()  # 第i个:[C,H,W]
            weight = np.mean(gradient, axis=(1, 2))  # [C]
            feature = self.feature[i].cpu().data.numpy()  # 第i个:[C,H,W]
            cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
            cam = np.sum(cam, axis=0)  # [C,H,W] -> [H,W]
            cam = np.maximum(cam, 0)  # ReLU
            # 数值归一化
            cam -= np.min(cam)
            cam /= np.max(cam)
            cam_all[i] = cv2.resize(cam, (inputs.size(3),inputs.size(2))) # [C,W,H] -> [C,H,W]
        return cam_all

    # 只计算一次梯度
    def __call__(self, inputs, index):
        """
        :param inputs: [N,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [N,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy(),axis=1) # [N]
        #index_max = max(index,key=index.count) #选择出现最多的index
        index_max=np.argmax(np.bincount(index))
        target = output[:,index_max] # [N]
        target = target.mean()
        target.backward(retain_graph=True)

        gradient = self.gradient.cpu().data.numpy()  # [N,C,H,W]
        weight = np.mean(gradient, axis=(2, 3))  # [N,C]
        feature = self.feature.cpu().data.numpy()  # [N,C,H,W]
        cam = feature * weight[:, :, np.newaxis, np.newaxis]  # [N,C,H,W]
        cam = np.sum(cam, axis=1)  # [N,H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # resize to 224*224
        cam_all=np.random.randn(inputs.size(0),inputs.size(2),inputs.size(3)) #[n,h,w]
        #print(cam_all.shape)
        for i,j in enumerate(cam):
            # j:[c,h,w] ,数值归一化
            j -= np.min(j)
            j /= np.max(j)
            cam_all[i] = cv2.resize(j, (inputs.size(3),inputs.size(2))) # cv2的resize宽高顺序相反，是W,H
        cam_all = cam_all.reshape(inputs.size(0),1,inputs.size(2),inputs.size(3))
        ts = torch.tensor(cam_all)
        return ts 


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def call_per_image(self, inputs, index):
        """

        :param inputs: [N,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [N,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy(),axis=1) # [N]
        cam_all=np.random.randn(inputs.size(0),inputs.size(2),inputs.size(3))
        for i,j in enumerate(index):
            target = output[i][j]
            target.backward(retain_graph=True)
            gradient = self.gradient[i].cpu().data.numpy() # 第i个:[C,H,W]
            gradient = np.maximum(gradient, 0.)  # ReLU
            indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
            norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
            for x in range(len(norm_factor)):
                norm_factor[x] = 1. / norm_factor[x] if norm_factor[x] > 0. else 0.  # 避免除零
            alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]
            weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)
            feature = self.feature[i].cpu().data.numpy()  # [C,H,W]
            cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
            cam = np.sum(cam, axis=0)  # [H,W]
            # cam = np.maximum(cam, 0)  # ReLU

            # 数值归一化
            cam -= np.min(cam)
            cam /= np.max(cam)
            # resize to 224*224
            cam_all[i] = cv2.resize(cam, (inputs.size(3),inputs.size(2)))
        return cam_all

    def __call__(self, inputs, index):
        """

        :param inputs: [N,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [N,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy(),axis=1) # [N]
        index_max=np.argmax(np.bincount(index))
        target = output[:,index_max] # [N]
        target = target.mean()
        target.backward(retain_graph=True)
        cam_all=np.random.randn(inputs.size(0),inputs.size(2),inputs.size(3)) #[n,w,h]
        for i,j in enumerate(index):
            gradient = self.gradient[i].cpu().data.numpy() # 第i个:[C,H,W]
            gradient = np.maximum(gradient, 0.)  # ReLU
            indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
            norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
            for x in range(len(norm_factor)):
                norm_factor[x] = 1. / norm_factor[x] if norm_factor[x] > 0. else 0.  # 避免除零
            alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]
            weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)
            feature = self.feature[i].cpu().data.numpy()  # [C,H,W]
            cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
            cam = np.sum(cam, axis=0)  # [H,W]
            # cam = np.maximum(cam, 0)  # ReLU

            # 数值归一化
            cam -= np.min(cam)
            cam /= np.max(cam)
            # resize to 224*224
            cam_all[i] = cv2.resize(cam, (inputs.size(3),inputs.size(2)))
        cam_all = cam_all.reshape(inputs.size(0),1,inputs.size(2),inputs.size(3))
        ts = torch.tensor(cam_all)
        return ts

class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(self.backward_hook)
                #module.register_full_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """

        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=None):
        """

        :param inputs: [1,3,H,W]
        :param index: class_id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [n,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy(),axis=1) #[N]
        index_max=np.argmax(np.bincount(index))
        target = output[:,index_max]
        target = target.mean()
        target.backward(retain_graph=True)
        return inputs.grad  # [N,3,H,W]

def mask2cam(mask,imgs): #mask: [n,1,h,w], imgs:[n,3,h,w] 
    imgs = imgs.detach().clone().cpu()
    mask = mask.detach().clone().cpu()
    heatmap = np.float32(imgs).copy() #[n,c,h,w]
    heatmap_cv2 = np.transpose(heatmap,(0,2,3,1)) # [n,c,h,w] -> [n,h,w,c]
    cam = np.float32(imgs).copy()
    for i,j in enumerate(mask[:,0]):
        heatmap_i = cv2.applyColorMap(np.uint8(255 * j), cv2.COLORMAP_JET) #[H,W,1]
        heatmap_i = np.float32(heatmap_i) / 255
        heatmap_i= heatmap_i[..., ::-1]  # gbr to rgb
        heatmap_i = np.transpose(heatmap_i,(2,0,1)) #[H,W,1] -> [1,H,W]
        heatmap[i] = heatmap_i
        flag = imgs[i].detach().cpu() #[C,H,W]
        #flag = flag.permute(1,2,0) #[C,H,W] -> [H,W,C]
        cam[i] = heatmap_i + np.float32(flag.numpy())
        cam[i] -= np.max(np.min(cam.copy()), 0)
        cam[i] /= np.max(cam[i])
    return torch.tensor(heatmap),torch.tensor(cam)



#---------test-----------
# from skimage import io
# i1 = io.imread('cat-1.jpg') #[h,w,c]
# i2 = io.imread('cat-2.jpg')
# i3 = io.imread('cat-3.jpg')
# size= (1024,512)
# i1 = cv2.resize(i1,size)
# i2 = cv2.resize(i2,size)
# i3 = cv2.resize(i3,size)
# i1 = i1[np.newaxis,:] #[1,h,w,c]
# i2 = i2[np.newaxis,:]
# i3 = i3[np.newaxis,:]
# i1 = i1/255.0 # pixel: 255->1
# i2 = i2/255.0
# i3 = i3/255.0
# i1 = torch.tensor(i1,requires_grad=True)
# i2 = torch.tensor(i2,requires_grad=True)
# i3 = torch.tensor(i3,requires_grad=True)
# i1 = i1.permute(0,3,1,2) # [1,c,h,w]
# i2 = i2.permute(0,3,1,2)
# i3 = i3.permute(0,3,1,2)
# imgs = torch.cat((i1,i2))
# imgs = torch.cat((imgs,i3)) #[n,c,h,w]
# torchvision.utils.save_image(imgs,'imgs-256.png',nrow=1)
# imgs = imgs.to(torch.float32)

# vgg16 = torchvision.models.vgg16(pretrained = True)

# layer_name = None
# for name, m in vgg16.named_modules():
#     if isinstance(m, torch.nn.Conv2d):
#         layer_name = name

# grad_cam = GradCAM(vgg16, layer_name)
# grad_cam_2 = GradCamPlusPlus(vgg16, layer_name)
# mask = grad_cam(imgs, None) # #[c,h,w]
# mask_once = grad_cam.call_once(imgs, None)
# mask_2 = grad_cam_2(imgs, None)
# mask_once_2 = grad_cam_2.call_once(imgs, None)

# cam = None
# heatmap = None
# for i,j in enumerate(mask):
#     heatmap = cv2.applyColorMap(np.uint8(255 * j), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = heatmap[..., ::-1]  # gbr to rgb
#     flag = imgs[i]
#     flag = flag.permute(1,2,0)
#     cam = heatmap + np.float32(flag.detach().numpy())
#     cam -= np.max(np.min(cam.copy()), 0)
#     cam /= np.max(cam)
#     io.imsave('./{}-cam.jpg'.format(i), np.uint8(cam*255.))
#     io.imsave('./{}-heatmap.jpg'.format(i), (heatmap * 255.).astype(np.uint8))
#     io.imsave('./{}-img.jpg'.format(i), np.uint8(flag.detach().numpy()*255.))

# for i,j in enumerate(mask_2):
#     heatmap = cv2.applyColorMap(np.uint8(255 * j), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = heatmap[..., ::-1]  # gbr to rgb
#     flag = imgs[i]
#     flag = flag.permute(1,2,0)
#     cam = heatmap + np.float32(flag.detach().numpy())
#     cam -= np.max(np.min(cam.copy()), 0)
#     cam /= np.max(cam)
#     io.imsave('./{}-cam++.jpg'.format(i), np.uint8(cam*255.))
#     io.imsave('./{}-heatmap++.jpg'.format(i), (heatmap * 255.).astype(np.uint8))

# for i,j in enumerate(mask_once):
#     heatmap = cv2.applyColorMap(np.uint8(255 * j), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = heatmap[..., ::-1]  # gbr to rgb
#     flag = imgs[i]
#     flag = flag.permute(1,2,0)
#     cam = heatmap + np.float32(flag.detach().numpy())
#     cam -= np.max(np.min(cam.copy()), 0)
#     cam /= np.max(cam)
#     io.imsave('./{}-cam_once.jpg'.format(i), np.uint8(cam*255.))
#     io.imsave('./{}-heatmap_once.jpg'.format(i), (heatmap * 255.).astype(np.uint8))


# for i,j in enumerate(mask_once_2):
#     heatmap = cv2.applyColorMap(np.uint8(255 * j), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = heatmap[..., ::-1]  # gbr to rgb
#     flag = imgs[i]
#     flag = flag.permute(1,2,0)
#     cam = heatmap + np.float32(flag.detach().numpy())
#     cam -= np.max(np.min(cam.copy()), 0)
#     cam /= np.max(cam)
#     io.imsave('./{}-cam++_once.jpg'.format(i), np.uint8(cam*255.))
#     io.imsave('./{}-heatmap++_once.jpg'.format(i), (heatmap * 255.).astype(np.uint8))

# gbp = GuidedBackPropagation(vgg16)
# imgs.retain_grad()
# grad = gbp(imgs)
# grads = grad.data.numpy() # [n,c,h,w]
# grads -= np.max(np.min(grads), 0)
# grads /= np.max(grads)
# torchvision.utils.save_image(torch.tensor(grads),'./gb.png',nrow=1)

# for i,j in enumerate(grads):
#   j -= np.max(np.min(j), 0)
#   j /= np.max(j)
#   j *= 255.
#   gb = np.transpose(np.uint8(j), (1, 2, 0))
#   io.imsave('./gb_{}.jpg'.format(i), gb)

# gb = np.transpose(grad, (1, 2, 0))
# print(gb.shape)
# cam_gb = gb * mask[..., np.newaxis]
# print(cam_gb.shape)
