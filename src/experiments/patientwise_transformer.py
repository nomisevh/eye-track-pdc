import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.ki import KIDataset
from models.classifier import IndividualClassifier
from models.inception import InceptionTimeModel
from utils.data import PadCollate
from utils.misc import set_random_state
from utils.training import EarlyStopper
from utils.path import checkpoints_path as MODELS_PATH, data_path as DATA_PATH

torch.set_default_dtype(torch.float)
SEED = 42
DEVICE = 'cuda'
INCLUDE_PDON_TF = True
INCLUDE_PDON_SEGM = True
BINARY_CLF = True
N_EPOCHS_CLF = 200
# CLF_CHKPT = os.path.join(MODELS_PATH, f'individual_clf'
#                                       f'{"_pdon" if INCLUDE_PDON_SEGM else ""}.pth')
TEST_CLF = False
ANAL_CLF = True
set_random_state(SEED)

# Initialize Datasets
ds_segm_train = KIDataset(train=True, which='segments', config='ki_auto', ki_data_dirname='KI',
                          data_sources=['HC', 'PD_OFF', 'PD_ON'] if INCLUDE_PDON_SEGM else ['HC', 'PD_OFF'])
ds_segm_test = KIDataset(train=False, which='segments', config='ki_auto', ki_data_dirname='KI',
                         data_sources=['HC', 'PD_OFF', 'PD_ON'])
ds_segm_train = KIDataset(train=True, which='trials', config='ki_auto', ki_data_dirname='KI',
                          data_sources=['HC', 'PD_OFF', 'PD_ON'] if INCLUDE_PDON_TF else ['HC', 'PD_OFF'])
ds_segm_test = KIDataset(train=False, which='trials', config='ki_auto', ki_data_dirname='KI',
                         data_sources=['HC', 'PD_OFF', 'PD_ON'])

# Initialize InceptionTime (Embedder)
inception_chkpt = torch.load(os.path.join(MODELS_PATH, 'inception_4_0.pth'))
embedder = InceptionTimeModel(**inception_chkpt['params'])
embedder = embedder.to(DEVICE)

# Initialize Transformer (Classifier)
clf = IndividualClassifier(in_features=embedder.out_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256,
                           batch_first=True, dropout=0.1, n_classes=2 if BINARY_CLF else 3)
clf = clf.to(DEVICE)
clf_optim = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-3)
clf_sched = torch.optim.lr_scheduler.CosineAnnealingLR(clf_optim, T_max=N_EPOCHS_CLF)
crit = nn.BCEWithLogitsLoss()

# Create dataset for classification
ds_chkpt = os.path.join(DATA_PATH, f'ki_dataset_pp'
                                   f'{"_pdon" if INCLUDE_PDON_CLF else ""}'
                                   f'{"_binary" if BINARY_CLF else ""}.pth')
if not os.path.exists(ds_chkpt):
    pbar = tqdm(total=400, desc='Generating Embedded Dataset')
    for c in (['hc', 'pd_off', 'pd_on'] if INCLUDE_PDON_CLF else ['hc', 'pd_off']):
        for d in ['horiz', 'vert']:
            for s in ['pro', 'anti']:
                for i in range(90):
                    try:
                        ds = KIIndividualDataset(pp, ind_idx=i, which_cohort=c, which_direction=d, which_saccades=s,
                                                 binary=BINARY_CLF)
                    except FileNotFoundError:
                        continue
                    except FileNotUsableError as e:
                        # print(str(e), file=sys.stderr)
                        pbar.update()
                        continue

                    dl = DataLoader(ds, batch_size=16, shuffle=False)
                    x_ds = []
                    for x, _, y in dl:
                        x = x.to(DEVICE)
                        with torch.no_grad():
                            x_feat = embedder(x.swapaxes(1, 2))
                        x_ds.append(x_feat.detach().cpu())
                    torch.save(torch.concat(x_ds), ds.fpath.replace('.csv', '.pth'))
                    pbar.update()
dspp = KIIndividualDatasetPP(pdon=INCLUDE_PDON_CLF, binary=BINARY_CLF)

# Train
if not os.path.exists(CLF_CHKPT) or True:
    # Create Dataloader
    train_idx, test_idx = train_test_split(np.arange(len(dspp)), test_size=0.1, shuffle=True,
                                           stratify=np.array([y.argmax().item() for y in dspp.y]))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    dlpp_train = DataLoader(dspp, batch_size=16, pin_memory=DEVICE == 'cuda',
                            sampler=train_sampler, collate_fn=PadCollate(dim=0))
    dlpp_test = DataLoader(dspp, batch_size=16, pin_memory=DEVICE == 'cuda',
                           sampler=test_sampler, collate_fn=PadCollate(dim=0))

    # Training loop
    es = EarlyStopper(patience=10, max_delta_acc=0.5, model=clf, ckpt_path=CLF_CHKPT)
    pbar = tqdm(range(N_EPOCHS_CLF), desc=f'NaN')
    lt, lv, at, av = [], [], [], []
    for epoch in pbar:
        # Train
        accs_train, losses_train = [], []
        for x, y in dlpp_train:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # Get TF's output
            y_hat = clf(x)
            # Loss + update
            clf_optim.zero_grad()
            loss = crit(y_hat, y)
            loss.backward()
            clf_optim.step()
            losses_train.append(crit(y_hat, y).detach().item())
            accs_train.append(accuracy_score(y_hat.detach().cpu().argmax(axis=1), y.detach().cpu().argmax(axis=1)))
        acc_train_mean = np.mean(accs_train)
        loss_train_mean = np.mean(losses_train)

        clf_sched.step()

        # Eval
        accs_test, losses_test = [], []
        with torch.no_grad():
            clf.eval()
            for x, y in dlpp_test:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y_hat = clf(x)
                losses_test.append(crit(y_hat, y).detach().item())
                accs_test.append(accuracy_score(y_hat.cpu().argmax(axis=1), y.cpu().argmax(axis=1)))
            clf.train()
        acc_test_mean = np.mean(accs_test)
        loss_test_mean = np.mean(losses_test)

        pbar.set_description(f'[e|{epoch}][l|t:{loss_train_mean:.2f}, v:{loss_test_mean:.2f}]'
                             f'[acc|t:{acc_train_mean * 100:.2f}, v:{acc_test_mean * 100:.2f}]')

        at.append(acc_train_mean)
        av.append(acc_test_mean)
        lt.append(loss_train_mean)
        lv.append(loss_test_mean)
        if es.early_stop(acc_test_mean):
            break
    es.revert_chkpt()

    # Plots
    plt.plot(lt, label='loss train')
    plt.plot(lv, label='loss test')
    plt.legend()
    plt.show()
    plt.plot(at, label='acc train')
    plt.plot(av, label='acc test')
    plt.legend()
    plt.show()

# Test clf
anal_idx = None
if TEST_CLF:
    clf_eval = torch.load(CLF_CHKPT).eval()
    _, eval_idx = train_test_split(np.arange(len(dspp)), test_size=0.1, shuffle=True,
                                   stratify=np.array([y.argmax().item() for y in dspp.y]))
    eval_sampler = torch.utils.data.SubsetRandomSampler(eval_idx)
    dlpp_eval = DataLoader(dspp, batch_size=8, pin_memory=DEVICE == 'cuda',
                           sampler=eval_sampler, collate_fn=PadCollate(dim=0))
    accs_eval, f1s_eval, losses_eval = [], [], []
    with torch.no_grad():
        for x, y in dlpp_eval:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_hat = clf_eval(x)
            losses_eval.append(crit(y_hat, y).detach().item())
            accs_eval.append(accuracy_score(y_hat.cpu().clone().argmax(axis=1), y.cpu().clone().argmax(axis=1)))
            f1s_eval.append(f1_score(y_hat.cpu().clone().argmax(axis=1), y.cpu().clone().argmax(axis=1)))
    acc_eval_mean = np.mean(accs_eval)
    f1_eval_mean = np.mean(f1s_eval)
    loss_eval_mean = np.mean(losses_eval)

    # print(loss_eval_mean)
    print(acc_eval_mean)
    print(f1_eval_mean)
    print('')
    anal_idx = eval_idx

# Analyze Model
if ANAL_CLF:
    clf_anal = torch.load(CLF_CHKPT).eval()
    dspp = KIIndividualDatasetPP(pdon=INCLUDE_PDON_CLF, binary=BINARY_CLF, return_true_label=True)
    if anal_idx is None:
        _, anal_idx = train_test_split(np.arange(len(dspp)), test_size=0.1, shuffle=True,
                                       stratify=np.array([y.argmax().item() for y in dspp.y]))
    n_correct, n_total = 0, 0
    y_hats, ys = [], []
    with torch.no_grad():
        for i, (x, y, y_true) in enumerate(Subset(dspp, anal_idx)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_hat = nn.Sigmoid()(clf_anal(x.unsqueeze(0).to(DEVICE))).detach().cpu().squeeze(0).numpy()
            y_hats.append(np.argmax(y_hat).astype(int).item())
            ys.append(torch.argmax(y).int().item())
            correct = ys[-1] == y_hats[-1]
            if correct:
                n_correct += 1
            n_total += 1
            print(f'{i:02d}| true={y_true.item()} | pred={[f"{_:.2f}" for _ in y_hat]} | corr={str(correct).upper()}')
    print(f'ACC={100 * n_correct / n_total:.2f}%')
    cm = confusion_matrix(ys, y_hats, labels=[0, 1] if BINARY_CLF else [0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['HC', 'PD'] if BINARY_CLF else ['HC', 'PD_OFF', 'PD_ON'])
    disp.plot()
    plt.savefig(f'tf_cm{"_pdon" if INCLUDE_PDON_CLF else ""}.pdf')
    plt.show()
