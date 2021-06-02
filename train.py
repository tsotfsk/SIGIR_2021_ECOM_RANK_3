import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import pickle
from tqdm import tqdm

from trainer import FullEvaluator, SeqLoader
from utils import args, get_model, init_env, test_remap


@torch.no_grad()
def test(args, test_dataloader, k=20):
    result = []
    uids = []
    model = get_model(args.model)(
        args, n_items=test_dataloader.n_items, device=device)
    model.to(device)
    model.load_state_dict(torch.load(
        f'./saved/{args.model}.pth')['state_dict'])
    with tqdm(test_dataloader, total=test_dataloader.batch_num, leave=False) as t:
        for _, batch_data in enumerate(t):
            logits = model.predict(batch_data)
            logits.scatter_(1, batch_data['seqs'], -np.inf)
            # logits[:, 0].fill_(-np.inf)
            result.append(torch.topk(logits[:, 1:], k=k, dim=-1)[1].cpu().numpy())
            uids.append(batch_data['user_ids'].cpu().numpy())
        iids = np.concatenate(result)
        uids = np.concatenate(uids)
    test_remap(uids, iids)


def valid(valid_dataloader, evaluator):
    model = get_model(args.model)(
        args, n_items=test_dataloader.n_items, device=device)
    model.to(device)
    model.load_state_dict(torch.load(
        f'./saved/{args.model}.pth')['state_dict'])
    model.eval()
    matrix = []
    result = []
    with torch.no_grad():
        with tqdm(valid_dataloader, total=valid_dataloader.batch_num, leave=False) as t:
            for _, batch_data in enumerate(t):
                logits = model.predict(batch_data)
                batch_matrix, idxs = evaluator.collect(
                    logits, batch_data)
                matrix.append(batch_matrix)
                result.append(idxs)
        mrr = evaluator.evaluate(matrix)
        logger.info(f'MRR@1 {mrr}')


def train(args, train_dataloader, valid_dataloader, model, optimizer, evaluator):
    best_result = None
    iters = 0
    for epoch in range(args.epochs):

        # Train
        total_loss = []
        model.train()
        train_dataloader.shuffle()
        with tqdm(train_dataloader, total=train_dataloader.batch_num, leave=False) as t:
            for idx, batch_data in enumerate(t):
                optimizer.zero_grad()
                loss = model.calculate_loss(batch_data)
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
                t.set_description(f'Epoch {epoch}')
                t.set_postfix(loss=loss.item())

        logger.info('[epoch={}]: loss {}'.format(epoch, np.mean(total_loss)))
        # logger.info('[epoch={}]: loss {}]'.format(epoch, np.mean(total_loss)))

        # Test
        # if args.stage == 'offline':
        model.eval()
        matrix = []
        result = []
        with torch.no_grad():
            with tqdm(valid_dataloader, total=valid_dataloader.batch_num, leave=False) as t:
                for _, batch_data in enumerate(t):
                    logits = model.predict(batch_data)
                    logits.scatter_(1, batch_data['seqs'], -np.inf)
                    batch_matrix, idxs = evaluator.collect(
                        logits, batch_data)
                    matrix.append(batch_matrix)
                    result.append(idxs)
            mrr = evaluator.evaluate(matrix)
            logger.info(f'[epoch={epoch}]: MRR@1 {mrr}')
            if best_result is None:
                best_result = mrr
                model.save_model()
            elif mrr > best_result:
                best_result = mrr
                iters = 0
                model.save_model()
            else:
                iters += 1
        if iters > args.early_stop:
            logger.info('Early Stop...')
            logger.info('Best Result:' + best_result)
            break


if __name__ == "__main__":
    logger = init_env(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_dataloader = SeqLoader(
        phase='valid', device=device, batch_size=2 * args.batch_size)

    evaluator = FullEvaluator(
        metrics=["MRR"], topk=10, pos_len=1)
    if args.evaluate:
        test_dataloader = SeqLoader(
            phase='test', device=device, batch_size=2 * args.batch_size)
        # valid(valid_dataloader, evaluator)
        test(args, test_dataloader)
    else:
        train_dataloader = SeqLoader(
            phase='train', device=device, batch_size=args.batch_size)
        model = get_model(args.model)(
            args, n_items=train_dataloader.n_items, device=device)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train(args, train_dataloader, valid_dataloader,
              model, optimizer, evaluator)
