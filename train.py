import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from trainer import FullEvaluator, SeqLoader
from utils import args, get_model, init_env


# @torch.no_grad()
# def online_test(test_dataloader, model):
#     result = []
#     uids = []
#     with tqdm(test_dataloader, total=test_dataloader.batch_num, leave=False) as t:
#         for _, batch_data in enumerate(t):
#             logits = model.predict(batch_data)
#             result.append(torch.argmax(logits, dim=-1).cpu().numpy())
#             uids.append(batch_data['user_ids'].cpu().numpy())
#         result = np.concatenate(result) + 1
#         uids = np.concatenate(uids)
#     test_result = pd.DataFrame({'user_id': uids, 'product_id': result})
#     test_result['product_id'] = test_result['product_id'].map(item_map)
#     test_result.to_csv('result.csv', index=False)


if __name__ == "__main__":
    logger = init_env(args)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    train_dataloader = SeqLoader(
        phase='train', device=device, batch_size=args.batch_size)

    # full
    test_dataloader = SeqLoader(
        phase='valid', device=device, batch_size=2 * args.batch_size)
    evaluator = FullEvaluator(
        metrics=["MRR"], topk=1, pos_len=1)

    # model = GRU4Rec(args, n_items=train_dataloader.n_items)
    model = get_model(args.model)(
        args, n_items=train_dataloader.n_items, device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            with tqdm(test_dataloader, total=test_dataloader.batch_num, leave=False) as t:
                for _, batch_data in enumerate(t):
                    logits = model.predict(batch_data)
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
        # else:
        #     online_test(test_dataloader, model)
