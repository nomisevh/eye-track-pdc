import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm


def train_inception_time(model, clf, n_epochs, train_dl, test_dl, device, optimizer, lr_scheduler, bce_loss_fn,
                         triplet_loss_fn):
    # es = EarlyStopper(patience=10, max_delta_acc=0.05, model=clf, ckpt_path=CHKPT_FPATH)
    pbar = tqdm(range(n_epochs), desc=f'Training InceptionTime')
    lt, lv, at, av = [], [], [], []
    for epoch in pbar:
        # Train
        accs_train, losses_train = [], []
        for ax, ay, px, py, nx, ny in train_dl:
            ax = ax.to(device)
            ay = ay.to(device)
            px = px.to(device)
            py = py.to(device)
            nx = nx.to(device)
            ny = ny.to(device)

            optimizer.zero_grad()

            # Get embeddings
            ax_feat = model(ax.swapaxes(1, 2))
            px_feat = model(px.swapaxes(1, 2))
            nx_feat = model(nx.swapaxes(1, 2))

            # Get class predictions
            ay_hat = clf(ax_feat)
            py_hat = clf(px_feat)
            ny_hat = clf(nx_feat)

            # Compute losses
            bce_loss = bce_loss_fn(ay_hat, ay.float()) + bce_loss_fn(py_hat, py.float()) + bce_loss_fn(ny_hat,
                                                                                                       ny.float())
            triplet_loss = triplet_loss_fn(ax_feat, px_feat, nx_feat) + 0.1 * triplet_loss_fn(ay_hat, py_hat, ny_hat)
            total_loss = bce_loss + triplet_loss

            # Update models
            total_loss.backward()
            optimizer.step()
            losses_train.append(total_loss.item())
            accs_train.append(accuracy_score(ay_hat.detach().cpu().argmax(axis=1), ay.detach().cpu().argmax(axis=1)))

        acc_train_mean = np.mean(accs_train)
        loss_train_mean = np.mean(losses_train)
        lr_scheduler.step()

        # Eval
        accs_val, losses_val = [], []
        with torch.no_grad():
            model.eval()
            clf.eval()
            for x, y in test_dl:
                x = x.to(device)
                y = y.to(device)
                y_hat = clf(model(x.swapaxes(1, 2)))
                losses_val.append(bce_loss_fn(y_hat, y.float()).detach().item())
                accs_val.append(accuracy_score(y_hat.cpu().argmax(axis=1), y.cpu().argmax(axis=1)))
            clf.train()
            model.train()
        acc_val_mean = np.mean(accs_val)
        loss_val_mean = np.mean(losses_val)

        pbar.set_description(f'[e|{epoch}][l|t:{loss_train_mean:.2f}, v:{loss_val_mean:.2f}]'
                             f'[acc|t:{acc_train_mean * 100:.2f}, v:{acc_val_mean * 100:.2f}]')

        at.append(acc_train_mean)
        av.append(acc_val_mean)
        lt.append(loss_train_mean)
        lv.append(loss_val_mean)
        # if es.early_stop(acc_val_mean):
        #     break
        # es.revert_chkpt()
    return lt, lv, at, av
