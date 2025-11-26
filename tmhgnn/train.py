import torch
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os

from sklearn.metrics import average_precision_score, roc_auc_score

from conf import *
import net as networks
import util


def get_optimizer(name, model, lr, wd):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f'Optimizer {name} is not supported yet.')


def get_lr(optimizer):
    for pg in list(optimizer.param_groups)[::-1]:
        return pg['lr']


# Function to save the model
def saveModel():
    # args는 main에서 전역으로 설정됨
    path = args.output

    dir_name = os.path.dirname(path)
    if dir_name != "" and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    torch.save(model.state_dict(), path)


# Save Last Results into DataFrame
def saveResult(pred_epoch, true_epoch):
    pre_cat = np.concatenate((pred_epoch), axis=0).squeeze()
    true_cat = np.concatenate((true_epoch), axis=0).squeeze()
    result_df = pd.DataFrame({'pred': pre_cat.tolist(), 'label': true_cat.tolist()})
    path = args.exp_name + '.pkl'
    result_df.to_pickle(path)


def eval(model, loader, device):
    running_accuracy = 0
    running_val_loss = 0
    pred_epoch = []
    true_epoch = []

    with torch.no_grad():
        model.eval()
        for data in loader:
            data = data.to(device)

            # forward
            # x_n: [N, 4 + emb_dim] (word/note/taxonomy meta + 임베딩)
            out = model(data.x_n, data.edge_index_n, data.edge_mask, data.batch)

            logits = out.view(-1)            # (B,)
            y_true = data.y.view(-1).float() # (B,)

            val_loss = criterion(logits, y_true)

            probs = torch.sigmoid(logits)    # (B,)
            pred = (probs >= 0.5).long()     # 0 또는 1

            pred_np = probs.detach().cpu().numpy()
            true_np = y_true.detach().cpu().numpy()
            pred_epoch.append(pred_np)
            true_epoch.append(true_np)

            running_val_loss += val_loss.item()
            running_accuracy += int((pred == y_true.long()).sum())

    val_loss_value = running_val_loss / len(loader.dataset)
    accuracy = 100.0 * running_accuracy / len(loader.dataset)

    true_values = np.concatenate((true_epoch), axis=0).squeeze().tolist()
    predicted_values = np.concatenate((pred_epoch), axis=0).squeeze().tolist()
    auprc = average_precision_score(true_values, predicted_values)
    auroc = roc_auc_score(true_values, predicted_values)

    return val_loss_value, accuracy, auprc, auroc, pred_epoch, true_epoch


def train(num_epochs, model, train_loader, val_loader, test_loader, criterion, device):

    # best summary
    best_accuracy = 0.0
    best_loss = 1e9
    best_AUPRC = 0.0
    best_AUROC = 0.0

    print("Begin training...")
    step = 1
    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0
        running_train_correct = 0
        total = 0

        print('training epoch', epoch)
        model.train()
        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data.x_n, data.edge_index_n, data.edge_mask, data.batch)

            logits = out.view(-1)             # (B,)
            y_true = data.y.view(-1).float()  # (B,)
            loss = criterion(logits, y_true)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).long()

            running_train_correct += int((pred == y_true.long()).sum())
            total += y_true.size(0)

            step += 1

        train_loss_value = running_train_loss / len(train_loader.dataset)
        train_acc_value = 100.0 * running_train_correct / len(train_loader.dataset)

        val_loss, val_acc, val_auprc, val_auroc, _, _ = eval(model, val_loader, device)

        if val_acc > best_accuracy:
            best_accuracy = val_acc

        if val_loss < best_loss:
            best_loss = val_loss

        # AUPRC 기준으로 best model 저장
        if val_auprc > best_AUPRC:
            saveModel()
            best_AUPRC = val_auprc

            # best AUPRC 기준으로 test 평가
            _, test_acc, test_auprc, test_auroc, test_pred_epoch, test_true_epoch = eval(
                model, test_loader, device
            )
            print('Epoch:', epoch,
                  '========Test acc: %.4f,' % test_acc,
                  'Test auprc: %.4f,' % test_auprc,
                  'Test auroc: %.4f.' % test_auroc,
                  '========')

        if val_auroc > best_AUROC:
            best_AUROC = val_auroc

        print('Completed Epoch:', epoch,
              ', Training Loss : %.4f' % train_loss_value,
              ', Training Accuracy : %.4f %%' % train_acc_value,
              ', Validation Loss : %.4f' % val_loss,
              ', Validation Accuracy : %.4f %%' % val_acc)

    print("========End Training========")
    _, test_acc, test_auprc, test_auroc, test_pred_epoch, test_true_epoch = eval(
        model, test_loader, device
    )
    print('========Final Val acc: %.4f,' % val_acc,
          'Val auprc: %.4f,' % val_auprc,
          'Val auroc: %.4f.' % val_auroc,
          '========')
    print('========Final Test acc: %.4f,' % test_acc,
          'Test auprc: %.4f,' % test_auprc,
          'Test auroc: %.4f.' % test_auroc,
          '========')

    return test_pred_epoch, test_true_epoch


if __name__ == '__main__':
    # 1) conf.py 에서 sys.path, 기본 인자 설정
    user_config()
    # 2) 인자 파싱
    args = parse_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Multi Hyperedge 데이터 로더 설정
    if args.dload == 'multi_hyper':
        # train.py 기준 상위 폴더의 graph_construction 를 path에 추가
        here = os.path.dirname(os.path.abspath(__file__))
        graph_dir = os.path.join(here, '..', 'graph_construction')
        if graph_dir not in sys.path:
            sys.path.append(graph_dir)

        from LoadData import LoadPaitentData
        print('LoadData => LoadPaitentData imported')
    else:
        raise ValueError(f'Unknown dataloader: {args.dload}')

    # 시드 고정
    util.seed_everything(args.seed)

    # 데이터 로더 생성
    loader = LoadPaitentData(
        name=args.task,
        type='cutoff',
        tokenizer=args.tokenizer,
        data_type=args.dtype
    )
    train_loader, val_loader, test_loader, n_class = loader.get_train_test(
        batch_size=args.bsz,
        seed=args.seed
    )

    # 🔥 여기서 임베딩 차원을 "자동으로" 추론한다.
    #    clinicalbert, gatortron, word2vec 전부 지원 가능.
    example_data = train_loader.dataset[0]
    in_dim = example_data.x_n.size(1)  # x_n: [N, 4 + emb_dim]
    args.node_features = in_dim
    print(f"[INFO] Inferred node feature dim (in_dim) = {args.node_features}")

    # 모델 정의
    model_cls = getattr(networks, args.model)
    model = model_cls(
        num_features=args.node_features,
        hidden_channels=args.hidden_channels * args.heads_1
    )
    model.to(device)

    # Loss 설정
    if args.loss == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Unknown loss name: {args.loss}')

    # Optimizer 설정
    optimizer = get_optimizer(args.optimizer, model, args.init_lr, args.weight_decay)

    # 학습
    predicted_last, true_last = train(
        args.num_epochs, model,
        train_loader, val_loader, test_loader,
        criterion, device
    )

    # 마지막(best 기준이 아닌 최종 epoch 기준) 결과 저장
    saveResult(predicted_last, true_last)
