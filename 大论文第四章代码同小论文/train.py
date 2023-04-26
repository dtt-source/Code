import torch
from models.ner_model import BertSoftmaxForNer, LEBertSoftmaxForNer, LEBertCrfForNer, BertCrfForNer
from models.nested_lebert import NestLEBertForNer
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
import os
import numpy as np
from os.path import join
from loguru import logger
import time
from transformers import BertConfig
from torch.utils.data import Dataset, DataLoader
from processors.processor import LEBertProcessor, BertProcessor, NestLEBertProcessor
import json
from tqdm import tqdm
from transformers import BertTokenizerFast
from metrics.ner_metrics import SeqEntityScore, BiaffineScore
import transformers


def kl_div_table(table_scores_stu, table_scores_teach):


    table_scores_stu = torch.clamp(table_scores_stu, min=1e-7, max=1-1e-7)
    table_scores_stu_1 = 1 - table_scores_stu
    table_score_dist_stu = torch.cat(
        [table_scores_stu.unsqueeze(-1), table_scores_stu_1.unsqueeze(-1)], dim=-1
    )

    table_scores_teach = torch.clamp(table_scores_teach, min=1e-7, max=1-1e-7)
    table_scores_teach_1 = 1 - table_scores_teach
    table_score_dist_teach = torch.cat(
        [table_scores_teach.unsqueeze(-1), table_scores_teach_1.unsqueeze(-1)], dim=-1
    )

    """Kullback-Leibler divergence D(P || Q) for discrete distributions"""
    p = torch.log(table_score_dist_stu.view(-1, 2))
    q = torch.log(table_score_dist_teach.view(-1, 2))


    scores = torch.sum(p.exp() * (p - q), axis=-1)
    return scores.mean()


def set_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cnn', default=True)
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument("--output_path", type=str, default='output/', help='模型与预处理数据的存放位置')
    parser.add_argument("--pretrain_embed_path", type=str, default='word_embedding/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt', help='预训练词向量路径')

    parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'], help='损失函数类型')
    parser.add_argument('--add_layer', default=1, type=str, help='在bert的第几层后面融入词汇信息')
    parser.add_argument("--lr", type=float, default=2e-5, help='Bert的学习率')
    parser.add_argument("--crf_lr", default=1e-3, type=float, help="crf的学习率")
    parser.add_argument("--adapter_lr", default=1e-3, type=float, help="crf的学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size_train", type=int, default=8)
    parser.add_argument("--batch_size_eval", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=150, help="输入的最大长度")
    parser.add_argument("--max_word_num", type=int, default=3, help="每个汉字最多融合多少个词汇信息")
    parser.add_argument("--max_scan_num", type=int, default=10000, help="取预训练词向量的前max_scan_num个构造字典树")

    parser.add_argument("--data_path", type=str, default="datasets/renmin/", help='数据集存放路径')
    parser.add_argument("--dataset_name", type=str, default='renmin', help='数据集名称')
    parser.add_argument("--eval_step", type=int, default=20, help="训练多少步，查看验证集的指标")
    parser.add_argument("--contrastive", type=bool, default=True, help="训是否使用对比学习")

    parser.add_argument("--model_class", type=str, choices=['lebert-softmax', 'bert-softmax', 'bert-crf', 'lebert-crf','lebert-nest'],
                        default='lebert-nest', help='模型类别')
    parser.add_argument("--pretrain_model_path", type=str, default="bert-base-chinese")
    parser.add_argument("--overwrite", action='store_true', default=True, help="覆盖数据处理的结果")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=True)
    parser.add_argument("--load_word_embed", action='store_true', default=True, help='是否加载预训练的词向量')

    parser.add_argument('--markup', default='bios', type=str, choices=['bios', 'bio'], help='数据集的标注方式')
    parser.add_argument('--grad_acc_step', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪阈值')
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")

    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.')
    args = parser.parse_args()
    return args


def seed_everything(seed=42):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True


def get_optimizer(model, args, warmup_steps, t_total):

    no_bert = ["word_embedding_adapter", "word_embeddings", "classifier",  "crf"]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [

        {
            "params": [p for n, p in model.named_parameters()
                       if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, 'lr': args.lr
        },

        {
            "params": [p for n, p in model.named_parameters()
                       if (not any(nd in n for nd in no_bert) or n == 'bert.embeddings.word_embeddings.weight') and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, 'lr': args.lr
        },

        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": args.adapter_lr
        },

        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_bert) and n != 'bert.embeddings.word_embeddings.weight' and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr": args.adapter_lr
        }
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


def train(model, train_loader, dev_loader, test_loader, optimizer, scheduler, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    dev = 0
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx + 1
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)

            if args.model_class == 'bert-softmax':
                loss, logits = model(input_ids, attention_mask, token_type_ids, args.ignore_index, label_ids)
            elif args.model_class == 'bert-crf':
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'lebert-softmax':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, args.ignore_index, label_ids)
            elif args.model_class == 'lebert-crf':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
            elif args.model_class == 'lebert-nest':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                label_ids = data['nest_labels'].to(device)
                if args.contrastive:
                    loss1, logits1 = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
                    loss2, logits2 = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
                    kl_loss = kl_div_table(logits1, logits2) + kl_div_table(logits2, logits1)
                    loss = (loss1 + loss2)/2 + 10*kl_loss
                else:
                    loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)

            loss = loss.mean()


            loss = loss / args.grad_acc_step
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if step % args.grad_acc_step == 0:

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad()


            if step % args.eval_step == 0:
                logger.info('evaluate dev set')
                dev_result = evaluate(args, model, dev_loader)
                logger.info('evaluate test set')
                test_result = evaluate(args, model, test_loader)
                writer.add_scalar('dev loss', dev_result['loss'], step)
                writer.add_scalar('dev f1', dev_result['f1'], step)
                if args.model_class == 'lebert-nest':
                    writer.add_scalar('dev precision', dev_result['precision'], step)
                else:
                    writer.add_scalar('dev precision', dev_result['acc'], step)
                writer.add_scalar('dev recall', dev_result['recall'], step)

                writer.add_scalar('test loss', test_result['loss'], step)
                writer.add_scalar('test f1', test_result['f1'], step)
                if args.model_class == 'lebert-nest':
                    writer.add_scalar('test precision', test_result['precision'], step)
                else:
                    writer.add_scalar('test precision', test_result['acc'], step)
                writer.add_scalar('test recall', test_result['recall'], step)

                model.train()
                if best < test_result['f1']:
                    best = test_result['f1']
                    dev = dev_result['f1']
                    logger.info('higher f1 of testset is {}, dev is {} in step {} epoch {}'.format(best, dev, step, epoch+1))

                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_path)
    logger.info('best f1 of test is {}, dev is {}'.format(best, dev))


def evaluate(args, model, dataloader):
    """
    计算数据集上的指标
    :param args:
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()
    device = args.device
    if args.model_class == 'lebert-nest':
        metric = BiaffineScore(args.id2label)
    else:
        metric = SeqEntityScore(args.id2label, markup=args.markup)

    logger.info("***** Running evaluation *****")


    eval_loss = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label_ids = data['label_ids'].to(device)

            if args.model_class == 'bert-softmax':
                loss, logits = model(input_ids, attention_mask, token_type_ids, args.ignore_index, label_ids)
            elif args.model_class == 'bert-crf':
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'lebert-softmax':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, args.ignore_index,
                                     label_ids)
            elif args.model_class == 'lebert-crf':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
            elif args.model_class == 'lebert-nest':
                word_ids = data['word_ids'].to(device)
                word_mask = data['word_mask'].to(device)
                label_ids = data['nest_labels'].to(device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, word_ids, word_mask, label_ids)
            loss = loss.mean()
            eval_loss += loss

            input_lens = (torch.sum(input_ids != 0, dim=-1) - 2).tolist()
            if args.model_class in ['lebert-crf', 'bert-crf']:
                preds = model.crf.decode(logits, attention_mask).squeeze(0)
                preds = preds[:, 1:].tolist()
                label_ids = label_ids[:, 1:].tolist()
            elif args.model_class in ['lebert-nest']:
                preds = logits
            else:
                preds = torch.argmax(logits, dim=2)[:, 1:].tolist()
                label_ids = label_ids[:, 1:].tolist()

            if args.model_class in ['lebert-nest']:
                metric.update(pred_paths=preds, label_paths=label_ids)
            else:
                for i in range(len(label_ids)):
                    input_len = input_lens[i]
                    pred = preds[i][:input_len]
                    label = label_ids[i][:input_len]
                    metric.update(pred_paths=[pred], label_paths=[label])


    logger.info("\n")
    eval_loss = eval_loss / len(dataloader)
    if args.model_class not in ['lebert-nest']:
        eval_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        logger.info("***** Eval results *****")
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
        logger.info("***** Entity results *****")
        for key in sorted(entity_info.keys()):
            logger.info("******* %s results ********" % key)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            logger.info(info)
    else:
        eval_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        logger.info("***** Eval results *****")
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])

    return results


MODEL_CLASS = {
    'lebert-softmax': LEBertSoftmaxForNer,
    'lebert-crf': LEBertCrfForNer,
    'bert-softmax': BertSoftmaxForNer,
    'bert-crf': BertCrfForNer,
    'lebert-nest':NestLEBertForNer
}
PROCESSOR_CLASS = {
    'lebert-softmax': LEBertProcessor,
    'lebert-crf': LEBertProcessor,
    'bert-softmax': BertProcessor,
    'bert-crf': BertProcessor,
    'lebert-nest':NestLEBertProcessor
}


def main(args):

    tokenizer = BertTokenizerFast.from_pretrained(args.pretrain_model_path, do_lower_case=True)

    config = BertConfig.from_pretrained(args.pretrain_model_path)

    label2id_total = {'resume':{'O':0,'CONT':1,'EDU':2,'LOC':3,'NAME':4,'ORG':5,'PRO':6,'RACE':7,'TITLE':8},
                      'msra':{'O':0,'NR':1,'NS':2,'NT':3},
                      'ontonote4':{'O':0,'GPE':1,'LOC':2,'ORG':3,'PER':4},
                      'weibo':{'O':0,'GPE.NAM':1,'GPE.NOM':2,'LOC.NAM':3,'LOC.NOM':4,'ORG.NAM':5,'ORG.NOM':6,'PER.NAM':7,'PER.NOM':8},
                      'renmin':{'O':0,'PER':1,'T':2,'LOC':3,'ORG':4,},
                      'cmeee':{'O':0,'dep': 1, 'mic': 2, 'sym': 3, 'ite': 4, 'dis': 5, 'dru': 6, 'bod': 7, 'equ': 8, 'pro': 9},
                      'cluener':{'O':0,'address': 1, 'book': 2, 'company': 3, 'game': 4, 'government': 5, 'movie': 6, 'name': 7, 'organization': 8, 'position': 9, 'scene': 10}}

    if args.model_class in ['lebert-nest']:
        args.nest_label2id = label2id_total[args.dataset_name]
        args.id2label = {v:k for k,v in args.nest_label2id.items()}
        config.num_labels = len(args.nest_label2id)


    processor = PROCESSOR_CLASS[args.model_class](args, tokenizer)
    args.ignore_index = processor.label_vocab.convert_token_to_id('[PAD]')

    if args.model_class in ['lebert-softmax', 'lebert-crf']:
        args.id2label = processor.label_vocab.idx2token
        config.num_labels = processor.label_vocab.size


    config.loss_type = args.loss_type
    if args.model_class in ['lebert-softmax', 'lebert-crf', 'lebert-nest']:
        config.add_layer = args.add_layer
        config.word_vocab_size = processor.word_embedding.shape[0]
        config.word_embed_dim = processor.word_embedding.shape[1]
    config.use_cnn=args.use_cnn

    model = MODEL_CLASS[args.model_class].from_pretrained(args.pretrain_model_path, config=config).to(args.device)

    if args.model_class in ['lebert-softmax', 'lebert-crf'] and args.load_word_embed:
        logger.info('initialize word_embeddings with pretrained embedding')
        model.word_embeddings.weight.data.copy_(torch.from_numpy(processor.word_embedding))


    if args.do_train:

        train_dataset = processor.get_train_data()

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_dataset = processor.get_dev_data()

        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        test_dataset = processor.get_test_data()

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        t_total = len(train_dataloader) // args.grad_acc_step * args.epochs
        warmup_steps = int(t_total * args.warmup_proportion)




        optimizer, scheduler = get_optimizer(model, args, warmup_steps, t_total)
        train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args)


    if args.do_eval:

        dev_dataset = processor.get_dev_data()

        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)

        test_dataset = processor.get_test_data()

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        model = MODEL_CLASS[args.model_class].from_pretrained(args.output_path, config=config).to(args.device)
        model.eval()

        result = evaluate(args, model, dev_dataloader)
        if args.model_class not in ['lebert-nest']:
            prec = result['acc']
        else:
            prec = result['precision']
        logger.info('devset precision:{}, recall:{}, f1:{}, loss:{}'.format(prec, result['recall'], result['f1'], result['loss'].item()))

        result = evaluate(args, model, test_dataloader)
        if args.model_class not in ['lebert-nest']:
            prec = result['acc']
        else:
            prec = result['precision']
        logger.info(
            'testset precision:{}, recall:{}, f1:{}, loss:{}'.format(prec, result['recall'], result['f1'],
                                                                     result['loss'].item()))


if __name__ == '__main__':

    args = set_train_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

    pretrain_model = 'bert-base-chinese'
    args.output_path = join(args.output_path, args.dataset_name, args.model_class, pretrain_model, 'load_word_embed' if args.load_word_embed else 'not_load_word_embed')
    args.train_file = join(args.data_path, 'train.json')
    args.dev_file = join(args.data_path, 'dev.json')
    args.test_file = join(args.data_path, 'test.json')
    args.label_path = join(args.data_path, 'labels.txt')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        logger.info(args)
        writer = SummaryWriter(args.output_path)
    main(args)
