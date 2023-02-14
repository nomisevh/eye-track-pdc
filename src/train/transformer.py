import numpy as np
import torch
from einops import rearrange
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm


def train_transformer(embedder, transformer, n_epochs, train_dl, test_dl, device, optimizer, lr_scheduler, loss_fn):
    pbar = tqdm(range(n_epochs), desc=f'Training Transformer')
    lt, lv, at, av = [], [], [], []
    for epoch in pbar:
        # Train
        accs_train, losses_train = [], []
        for tx, ty in train_dl:
            tx = tx.to(device)
            ty = ty.to(device)

            optimizer.zero_grad()

            # Get embeddings
            with torch.no_grad():
                tx_feat = rearrange(embedder(tx), '(b n) d -> b n d', b=tx.shape[0])

            # Get class predictions
            ty_hat = transformer(tx_feat)

            # Compute losses
            bce_loss = loss_fn(ty_hat, ty.float())

            # Update model
            bce_loss.backward()
            optimizer.step()
            losses_train.append(bce_loss.item())
            accs_train.append(accuracy_score(ty_hat.detach().cpu().argmax(axis=1), ty.detach().cpu().argmax(axis=1)))

        acc_train_mean = np.mean(accs_train)
        loss_train_mean = np.mean(losses_train)
        lr_scheduler.step()

        # Eval
        accs_val, losses_val = [], []
        with torch.no_grad():
            transformer.eval()
            for x, y in test_dl:
                x = x.to(device)
                y = y.to(device)
                y_hat = transformer(rearrange(embedder(x), '(b n) d -> b n d', b=x.shape[0]))
                losses_val.append(loss_fn(y_hat, y.float()).detach().item())
                accs_val.append(accuracy_score(y_hat.cpu().argmax(axis=1), y.cpu().argmax(axis=1)))
            transformer.train()
        acc_val_mean = np.mean(accs_val)
        loss_val_mean = np.mean(losses_val)

        pbar.set_description(f'Training Transformer [e|{epoch}][l|t:{loss_train_mean:.2f}, v:{loss_val_mean:.2f}]'
                             f'[acc|t:{acc_train_mean * 100:.2f}, v:{acc_val_mean * 100:.2f}]')

        at.append(acc_train_mean)
        av.append(acc_val_mean)
        lt.append(loss_train_mean)
        lv.append(loss_val_mean)
        # if es.early_stop(acc_val_mean):
        #     break
        # es.revert_chkpt()
    return lt, lv, at, av
