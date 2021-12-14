# Mixed-Example-PyTorch-Reimplementation

This is a reimplementation of [Improved Mixed-Example Data Augmentation](https://arxiv.org/pdf/1805.11272.pdf) in PyTorch.

### Training Arguments

The arguments for the dataloader, optimizer and choice of augmentation method are defined in the fifth block of the notebook for easy access.

### Results

In the following table, we report the results of the baseline model and the two augmentation methods: VHMixup and VHBC+ on CIFAR10 dataset averaged over 5 runs.

<center>
        
| Model  | Error |
| ------------- | ------------- |
| ResNet-18  | 5.17 |
| VHMixup | 3.48  |
| VHBC+ | 3.62  |
        
</center>
