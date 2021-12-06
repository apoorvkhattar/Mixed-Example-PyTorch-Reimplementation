import torch

import numpy as np

def evaluate(model, test_dataloaders, device):
    model.eval()
    total_acc = 0
    for i, d in enumerate(test_dataloaders):
        x, y = d[0], d[1]
        x = x.float().to(device)
        y = y.long().to(device)

        with torch.no_grad():
            out_prob = model(x)

        pred = torch.argmax(out_prob, dim=1)
        prediction = pred.cpu().numpy()
        truth = y.cpu().numpy()
        acc = np.count_nonzero(prediction == truth)

        total_acc += acc
        
    return total_acc