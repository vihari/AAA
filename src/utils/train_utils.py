import sys 
import time
import tqdm

from torchvision import models

import numpy as np
import torch
from . import config

if config.xla:
    import torch_xla.debug.metrics as met
    import torch_xla.core.xla_model as xm

def evaluate(model, loader, dev=None, status=True):
    """
    Evaluate helper for a simple classification model
    """
    # this check is needed to handle cases where model is not torch.nn.Module such as when casting multioutputmodel to single output model in celeba dataset.
    if callable(getattr(model, "eval", None)):
        model.eval()
    
    corrs = []
    num = 0
    step = 0
    img_time, fetch_time, move_time = 0, 0, 0
    with torch.no_grad():
        if status:
            loader = tqdm.tqdm(loader)
        i = 0
        st = time.time()
        for torch_x, torch_y in loader:
            img_time += time.time() - st
            st = time.time()
            if dev is not None:
                torch_x = torch_x.to(dev)
                torch_y = torch_y.to(dev)
            move_time += time.time() - st
            st = time.time()
            logits = model(torch_x)
            preds = torch.argmax(logits, dim=1)
            corrs.append((preds == torch_y).detach().cpu().numpy())
            fetch_time += time.time() - st

            step += 1
#             if step % 10 == 0:
#                 print(met.metrics_report())
#             if step % 10 == 0:
#                 print ("Img time: %0.4f fetch time: %0.4f move time: %0.4f" % (img_time, fetch_time, move_time))
                    
            num += len(torch_x)
    corrs = np.concatenate(corrs, axis=0)
    return {'acc': corrs.sum()/num, 'correct': corrs}
        
def train(train_loader, val_loader, num_labels, save_name, nepochs=20, dev=None):
    """
    Train helper for a simple classification model
    """
    batch_size = 128
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
    if dev is not None:
        model = model.to(dev)
    
    best_val_acc, prev_val_acc = -1, -1
    strikes, tol = 0, 4
    initial_lr = 5e-2
    optimizer = torch.optim.SGD(model.parameters(), initial_lr)
    ce_loss = torch.nn.CrossEntropyLoss()
    
    for epoch in tqdm.tqdm(range(nepochs)):
        model.train()
        for torch_x, torch_y in tqdm.tqdm(train_loader):
            if dev is not None:
                torch_x = torch_x.to(dev)
                torch_y = torch_y.to(dev)
            
            logits = model(torch_x)
            loss = ce_loss(logits, torch_y)
            optimizer.zero_grad()
            loss.backward()
            if config.xla:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()

        print ("Evaluating...")
        train_acc = evaluate(model, train_loader, dev)['acc']
        val_acc = evaluate(model, val_loader, dev)['acc']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.cpu().state_dict(), save_name)
            # move it back to the device
            model = model.to(dev)
        sys.stderr.write ("Epoch: %d Train acc: %0.4f Val. Acc: %0.4f Best: %0.4f\n" % (epoch, train_acc, val_acc, best_val_acc))

        if val_acc < prev_val_acc:
            strikes += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr*(10**-strikes)
            print ("Changed LR to %f" % (initial_lr*(2**-strikes)))

        if strikes >= tol:
            break
        prev_val_acc = val_acc

    model.load_state_dict(torch.load(save_name))
    model = model.to(dev)
    sys.stderr.write ("Best val. accuracy: %0.4f\n" % (evaluate(model, val_loader, dev)['acc']))
    return model

def evaluate_arm_perf_fit(mu_hat, mu_true):
    """
    Evaluates the data fit of the per arm perf. predictions
    :mu_hat: array of predicted perf. for each bin in the same order as `dataset.arms`.
    """
    vals = [np.abs(mu_hat[_] - mu_true[_]) for _ in range(len(mu_hat))]

    ret = {'Max': np.max(vals), 'Mean': np.mean(vals)}
    return ret