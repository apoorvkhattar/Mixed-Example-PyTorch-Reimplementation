import torch

import torch.nn.functional as F
import torch.distributions.beta as beta


# Vertical Concat
def verticalConcatMask(batch_1, batch_2, n_outputs=10):
    b, c, h, w = batch_1[0].shape[0], batch_1[0].shape[1], batch_1[0].shape[2], batch_1[0].shape[3]
    
    lambda_vertical_beta = beta.Beta(torch.tensor([1.]), torch.tensor([1.]))
    lambda_vertical = lambda_vertical_beta.sample(torch.Size([b])).view(b,1)
    
    img_1, label_1 = batch_1[0], batch_1[1]
    img_2, label_2 = batch_2[0], batch_2[1]
    
    if len(label_1.shape) == 1:
        label_1 = F.one_hot(label_1, num_classes=n_outputs)
        
    if len(label_2.shape) == 1:
        label_2 = F.one_hot(label_2, num_classes=n_outputs)
    
    binary_mask = torch.ones(img_1.shape)
    for b_indx in range(b):
        binary_mask[b_indx,:,(lambda_vertical[b_indx]*h).long():,:] = 0
        
    vertical_img = binary_mask * img_1 + (1 - binary_mask) * img_2
    vertical_label = (lambda_vertical*h).long().repeat(1,n_outputs) / h * label_1 + \
                        (h - (lambda_vertical*h).long()).repeat(1,n_outputs) / h * label_2
    
    return vertical_img, vertical_label

# Horizontal Concat
def horizontalConcatMask(batch_1, batch_2, n_outputs=10):
    b, c, h, w = batch_1[0].shape[0], batch_1[0].shape[1], batch_1[0].shape[2], batch_1[0].shape[3]
    
    lambda_horizontal_beta = beta.Beta(torch.tensor([1.]), torch.tensor([1.]))
    lambda_horizontal = lambda_horizontal_beta.sample(torch.Size([b])).view(b,1)
    
    img_1, label_1 = batch_1[0], batch_1[1]
    img_2, label_2 = batch_2[0], batch_2[1]
    
    if len(label_1.shape) == 1:
        label_1 = F.one_hot(label_1, num_classes=n_outputs)
        
    if len(label_2.shape) == 1:
        label_2 = F.one_hot(label_2, num_classes=n_outputs)
    
    binary_mask = torch.ones(img_1.shape)
    for b_indx in range(b):
        binary_mask[b_indx,:,:,(lambda_horizontal[b_indx]*w).long():] = 0
        
    horizontal_img = binary_mask * img_1 + (1 - binary_mask) * img_2
    horizontal_label = (lambda_horizontal*w).long().repeat(1,n_outputs) / w * label_1 + \
                        (w - (lambda_horizontal*w).long()).repeat(1,n_outputs) / w * label_2
    
    return horizontal_img, horizontal_label


# VHMixup
def VHMixup(batch_1, batch_2, n_outputs=10):
    b, c, h, w = batch_1[0].shape[0], batch_1[0].shape[1], batch_1[0].shape[2], batch_1[0].shape[3]
    
    lambda_mixup_beta = beta.Beta(torch.tensor([1.]), torch.tensor([1.]))
    lambda_mixup = lambda_mixup_beta.sample(torch.Size([b])).view(b,1)
    
    vertical_concat, vertical_label = verticalConcatMask(batch_1, batch_2, n_outputs)
    horizontal_concat, horizontal_label = horizontalConcatMask(batch_1, batch_2, n_outputs)
    
    mixed_img = lambda_mixup.reshape(b,1,1,1).repeat(1,c,h,w) * vertical_concat + \
                    (1 - lambda_mixup.reshape(b,1,1,1).repeat(1,c,h,w)) * horizontal_concat
    mixed_label = lambda_mixup.repeat(1,n_outputs) * vertical_label + \
                    (1 - lambda_mixup.repeat(1,n_outputs)) * horizontal_label
    
    return mixed_img, mixed_label


# VHBC+
def VHBCplus(batch_1, batch_2, n_outputs=10):
    b, c, h, w = batch_1[0].shape[0], batch_1[0].shape[1], batch_1[0].shape[2], batch_1[0].shape[3]
    
    vertical_concat, vertical_label = verticalConcatMask(batch_1, batch_2, n_outputs)
    horizontal_concat, horizontal_label = horizontalConcatMask(batch_1, batch_2, n_outputs)
    
    lambda_uni = torch.rand(b)
    lambda_factor = (1 - lambda_uni) / lambda_uni
    
    vertical_std = torch.std(vertical_concat.view(b,-1),dim=1)
    horizontal_std = torch.std(horizontal_concat.view(b,-1),dim=1)
    std_factor = vertical_std / horizontal_std

    p = 1 / (1 + std_factor * lambda_factor)
    
    denom = torch.sqrt(p**2 + (1-p)**2)
    
    c, h, w = batch_1[0].shape[1], batch_1[0].shape[2], batch_1[0].shape[3]
    
    bcplus_img = (p.reshape(b,1,1,1).repeat(1,c,h,w) * vertical_concat + \
                      (1 - p).reshape(b,1,1,1).repeat(1,c,h,w) * horizontal_concat) / denom.reshape(b,1,1,1).repeat(1,c,h,w)
    bcplus_label = lambda_uni.reshape(b,1).repeat(1,n_outputs) * vertical_label + \
                        (1 - lambda_uni.reshape(b,1).repeat(1,n_outputs)) * horizontal_label
    
    return bcplus_img, bcplus_label