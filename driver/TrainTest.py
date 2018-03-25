import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Model import *
from data.Dataloader import *
from driver.Config import *
from driver.Parser import *
from transitionSystem.Action import *
from transitionSystem.State import *
from transitionSystem.Instance import *
import pickle

def get_gold_actions(data, vocab):
    all_actions = []
    for sentence in data:
        all_states = []
        start = State()
        all_states.append(start)
        start.ready(sentence, vocab)
        actions = []
        step = 0
        while not all_states[step].is_end():
            gold_action = all_states[step].get_gold_action(vocab)
            actions.append(gold_action)
            next_state = State()
            all_states[step].move(next_state, gold_action)
            all_states.append(next_state)
            step += 1
        all_actions.append(actions)
        result = all_states[step].get_result(vocab)
        arc_total, arc_correct, rel_total, rel_correct = evalDepTree(sentence, result)
        assert arc_total == arc_correct and rel_total == rel_correct
        assert len(actions) == (len(sentence) - 1) * 2
    return all_actions

def inst(data, actions):
    assert len(data) == len(actions)
    inst = []
    for idx in range(len(data)):
        inst.append((data[idx], actions[idx]))
    return inst

def train(train_data, dev_data, test_data, parser, vocab, config):
    encoder_optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.encoder.parameters()), config)
    decoder_optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.decoder.parameters()), config)

    global_step = 0
    best_UAS = 0
    batch_num = int(np.ceil(len(train_data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_action_correct,  overall_total_action = 0, 0
        for onebatch in data_iter(train_data, config.train_batch_size, True):
            words, extwords, tags, heads, rels, lengths, masks, sents, gold_actions, acs= \
                batch_data_variable_actions(onebatch, vocab)
            parser.encoder.train()
            parser.decoder.train()
            parser.training = True
            parser.encode(words, extwords, tags, masks)
            parser.decode(sents, gold_actions, vocab)
            loss = parser.compute_loss(acs)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            total_actions, correct_actions = parser.compute_accuracy()
            overall_total_action += total_actions
            overall_action_correct += correct_actions
            during_time = float(time.time() - start_time)

            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.2f" \
                %(global_step, iter, batch_iter,  during_time, overall_action_correct / overall_total_action, loss_value[0]))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.encoder.parameters()), \
                                        max_norm=config.clip)
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.decoder.parameters()), \
                                        max_norm=config.clip)
                encoder_optimizer.step()
                decoder_optimizer.step()
                parser.encoder.zero_grad()
                parser.decoder.zero_grad()
                global_step += 1

                arc_correct, rel_correct, arc_total, dev_uas, dev_las = \
                    evaluate(dev_data, parser, vocab, config.dev_file + '.' + str(global_step))
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))
                arc_correct, rel_correct, arc_total, test_uas, test_las = \
                    evaluate(test_data, parser, vocab, config.test_file + '.' + str(global_step))
                print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))

                if dev_uas > best_UAS:
                    print("Exceed best uas: history = %.2f, current = %.2f" % (best_UAS, dev_uas))
                    best_UAS = dev_uas
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(parser.model.state_dict(), config.save_model_path)


def evaluate(data, parser, vocab, outputFile):
    start = time.time()
    parser.training = False
    parser.encoder.eval()
    parser.decoder.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0
    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, tags, heads, rels, lengths, masks, sents, gold_actions, acs = \
            batch_data_variable_actions(onebatch, vocab)
        count = 0
        parser.encode(words, extwords, tags, masks)
        parser.decode(sents, gold_actions, vocab)
        for (idx, cur_states) in enumerate(parser.batch_states):
            tree = cur_states[-1].get_result(vocab)
            printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(sents[idx], tree)
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1
    output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las





class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()

if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = creatVocab(config.train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)


    train_data = read_corpus(config.train_file, vocab)
    dev_data = read_corpus(config.dev_file, vocab)
    test_data = read_corpus(config.test_file, vocab)

    train_actions = get_gold_actions(train_data, vocab)
    dev_actions = get_gold_actions(dev_data, vocab)
    test_actions = get_gold_actions(test_data, vocab)

    assert len(train_data) == len(train_actions) and \
           len(dev_data) == len(dev_actions) and \
           len(test_data) == len(test_actions)
    vocab.create_action_table(train_actions)

    train_insts = inst(train_data, train_actions)
    dev_insts = inst(dev_data, dev_actions)
    test_insts = inst(test_data, test_actions)

    encoder = ParserModel(vocab, config, vec)
    decoder = Decoder(vocab, config)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    parser = TransitionBasedParser(encoder, decoder, vocab.ROOT, config)
    train(train_insts, dev_insts, test_insts, parser, vocab, config)
