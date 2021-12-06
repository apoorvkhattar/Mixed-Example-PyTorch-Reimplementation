import torch

import torch.distributions.beta as beta


# Vertical Concat
def verticalConcat(pair_1, pair_2, n_outputs):
    lambda_vertical_beta = beta.Beta(torch.tensor([1.]), torch.tensor([1.]))
    lambda_vertical = lambda_vertical_beta.sample()
    
    img_1, label_1 = pair_1[0], pair_1[1]
    img_2, label_2 = pair_2[0], pair_2[1]
    
    h, w = img_1.shape[1], img_1.shape[2]
    
    if len(label_1.shape) == 0:
        label_1 = torch.zeros(n_outputs)
        label_1[pair_1[1]] = 1
        
    if len(label_2.shape) == 0:
        label_2 = torch.zeros(n_outputs)
        label_2[pair_2[1]] = 1
    
    vertical_concat = torch.zeros(img_1.shape)
    vertical_label = torch.zeros(n_outputs)

    vertical_concat[:,:int(lambda_vertical*h),:] = img_1[:,:int(lambda_vertical*h),:]
    vertical_concat[:,int(lambda_vertical*h):,:] = img_2[:,int(lambda_vertical*h):,:]
    
    vertical_label = int(lambda_vertical*h) / h * label_1 + (h - int(lambda_vertical*h)) / h * label_2
    
    return vertical_concat, vertical_label

# Horizontal Concat
def horizontalConcat(pair_1, pair_2, n_outputs):
    lambda_horizontal_beta = beta.Beta(torch.tensor([1.]), torch.tensor([1.]))
    lambda_horizontal = lambda_horizontal_beta.sample()
    
    img_1, label_1 = pair_1[0], pair_1[1]
    img_2, label_2 = pair_2[0], pair_2[1]
    
    h, w = img_1.shape[1], img_1.shape[2]
    
    if len(label_1.shape) == 0:
        label_1 = torch.zeros(n_outputs)
        label_1[pair_1[1]] = 1
        
    if len(label_2.shape) == 0:
        label_2 = torch.zeros(n_outputs)
        label_2[pair_2[1]] = 1
    
    horizontal_concat = torch.zeros(img_1.shape)
    horizontal_label = torch.zeros(n_outputs)
    
    horizontal_concat[:,:,:int(lambda_horizontal*w)] = img_1[:,:,:int(lambda_horizontal*w)]
    horizontal_concat[:,:,int(lambda_horizontal*w):] = img_2[:,:,int(lambda_horizontal*w):]
    
    horizontal_label = int(lambda_horizontal*w) / w * label_1 + (w - int(lambda_horizontal*w)) / w * label_2
    
    return horizontal_concat, horizontal_label


# VHMixup
def VHMixup(pair_1, pair_2, n_outputs):
    lambda_mixup_beta = beta.Beta(torch.tensor([1.]), torch.tensor([1.]))
    lambda_mixup = lambda_mixup_beta.sample()
    
    vertical_concat, vertical_label = verticalConcat(pair_1, pair_2, n_outputs)
    horizontal_concat, horizontal_label = horizontalConcat(pair_1, pair_2, n_outputs)
    
    mixed_img = lambda_mixup * vertical_concat + (1 - lambda_mixup) * horizontal_concat
    mixed_label = lambda_mixup * vertical_label + (1 - lambda_mixup) * horizontal_label
    
    return mixed_img, mixed_label


# VHBC+
def VHBCplus(pair_1, pair_2, n_outputs):
    vertical_concat, vertical_label = verticalConcat(pair_1, pair_2, n_outputs)
    horizontal_concat, horizontal_label = horizontalConcat(pair_1, pair_2, n_outputs)
    
    lambda_uni = torch.rand(1)
    lambda_factor = (1 - lambda_uni) / lambda_uni
    
    vertical_std = torch.std(vertical_concat)
    horizontal_std = torch.std(horizontal_concat)
    std_factor = vertical_std / horizontal_std

    p = 1 / (1 + std_factor * lambda_factor)
    
    denom = torch.sqrt(p**2 + (1-p)**2)
    
    bcplus_img = (p * vertical_concat + (1 - p) * horizontal_concat) / denom
    bcplus_label = lambda_uni * vertical_label + (1 - lambda_uni) * horizontal_label
    
    return bcplus_img, bcplus_label


# Augment pair-wise in a batch
def augmentBatch(pair_1, pair_2, augmentor, n_outputs):
    img_1, label_1 = pair_1[0], pair_1[1]
    img_2, label_2 = pair_2[0], pair_2[1]

    augment_batch = torch.zeros(img_1.shape)
    augment_labels = torch.zeros(label_1.shape[0],n_outputs)

    for b in range(img_1.shape[0]):
        p1 = img_1[b], label_1[b]
        p2 = img_2[b], label_2[b]

        augment_batch[b], augment_labels[b] = augmentor(p1, p2, n_outputs)

    return augment_batch, augment_labels