from transition.State import *
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class TransitionBasedParser(object):
    def __init__(self, encoder, decoder, root_id, config):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.root = root_id
        encoder_p = next(filter(lambda p: p.requires_grad, encoder.parameters()))
        self.use_cuda = encoder_p.is_cuda
        self.device = encoder_p.get_device() if self.use_cuda else None
        self.encoder_bucket = Variable(torch.zeros(1, self.config.lstm_hiddens * 2)).type(torch.FloatTensor)
        self.score_bucket = Variable(torch.zeros(1, self.decoder.vocab.ac_size)).type(torch.FloatTensor)
        if self.use_cuda:
            self.encoder_bucket =self.encoder_bucket.cuda(self.device)
            self.score_bucket =self.score_bucket.cuda(self.device)
        self.step = []
        self.gold_pred_actions = []
        self.training = True

    def encode(self, words, extwords, tags, masks):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device),
            tags = tags.cuda(self.device)
            masks = masks.cuda(self.device)
        self.encoder_outputs = self.encoder.forward(words, extwords, tags, masks)

    def compute_loss(self, true_acs):
        b, l1, l2 = self.decoder_outputs.size()
        true_acs = _model_var(
            self.encoder,
            pad_sequence(true_acs, length=l1, padding=-1, dtype=np.int64))
        arc_loss = F.cross_entropy(
            self.decoder_outputs.view(b * l1, l2), true_acs.view(b * l1),
            ignore_index=-1)
        return arc_loss

    def compute_accuracy(self):
        total_num = 0
        correct = 0
        for iter in self.gold_pred_actions:
            gold_len = len(iter[0])
            pred_len = len(iter[1])
            assert gold_len == pred_len
            total_num += gold_len
            for idx in range(0, gold_len):
                if iter[0][idx] == iter[1][idx]:
                    correct += 1
        return total_num, correct


    def decode(self, batch_data, bacth_gold_actions, vocab):
        decoder_scores = []
        self.step.clear()
        self.gold_pred_actions.clear()

        b = self.encoder_outputs.size()[0]
        start_states = []
        for idx in range(0, b):
            start_states.append(State())
            start_states[idx].ready(batch_data[idx], vocab)
            self.step.append(0)

        batch_states = []
        for idx in range(0, b):
            one_inst_states = []
            one_inst_states.append(start_states[idx])
            batch_states.append(one_inst_states)

        while not self.all_states_are_finished(batch_states):
            self.prepare_atom_feat(batch_states, vocab)
            if self.training:
                gold_actions = self.get_gold_actions(batch_states, bacth_gold_actions)
            hidden_states = self.batch_hidden_state(batch_states)
            all_candidates = self.get_candidates(batch_states, vocab)
            action_scores = self.decoder.forward(hidden_states, all_candidates)
            pred_ac_ids = self.get_predicted_ac_id(action_scores)
            pred_actions = self.get_predict_actions(pred_ac_ids, vocab)
            batch_action_scores = self.padding_action_scores(batch_states, action_scores)
            if self.training:
                self.move(batch_states, gold_actions, vocab)
                self.gold_pred_actions.append((gold_actions, pred_actions))
            else:
                self.move(batch_states, pred_actions, vocab)
            decoder_scores.append(batch_action_scores.unsqueeze(1))
        self.batch_states = batch_states
        self.decoder_outputs = torch.cat(decoder_scores, 1)

    def get_gold_actions(self, batch_states, batch_gold_actions):
        gold_actions = []
        for (idx, cur_states) in enumerate(batch_states):
            if not cur_states[-1].is_end():
                gold_ac = batch_gold_actions[idx][self.step[idx]]
                gold_actions.append(gold_ac)
        return gold_actions

    def get_predict_actions(self, pred_ac_ids, vocab):
        pred_actions = []
        for ac_id in pred_ac_ids:
            pred_ac = vocab.id2ac(ac_id)
            pred_actions.append(pred_ac)
        return pred_actions

    def all_states_are_finished(self, batch_states):
        is_finish = True
        for idx in range(0, len(batch_states)):
            if not batch_states[idx][-1].is_end():
                is_finish = False
                break
        return is_finish

    def prepare_atom_feat(self, batch_states, vocab):
        for idx in range(0, len(batch_states)):
            if not batch_states[idx][-1].is_end():
                batch_states[idx][-1].prepare_atom_feat(self.encoder_outputs, idx, self.encoder_bucket, vocab)

    def batch_hidden_state(self, batch_states):
        states = []
        for idx in range(0, len(batch_states)):
            if not batch_states[idx][-1].is_end():
                states.append(batch_states[idx][-1].hidden_state)
        states = torch.cat(states, 0)
        return states

    def padding_action_scores(self, batch_states, action_scores):
        offset = 0
        batch_action_scores = []
        for cur_states in batch_states:
            if not cur_states[-1].is_end():
                batch_action_scores.append(action_scores[offset].unsqueeze(0))
                offset += 1
            else:
                batch_action_scores.append(self.score_bucket)
        batch_action_scores = torch.cat(batch_action_scores, 0)
        return batch_action_scores

    def get_predicted_ac_id(self, action_scores):
        state_num = action_scores.size()[0]
        ac_ids = []
        for idx in range(0, state_num):
            ac_id = np.argmax(action_scores.data[idx])
            ac_ids.append(ac_id)
        return ac_ids

    def move(self, batch_states, pred_actions, vocab):
        count = 0
        for idx in range(0, len(batch_states)):
            if not batch_states[idx][-1].is_end():
                count += 1
        assert len(pred_actions) == count
        offset = 0
        for (idx, cur_states) in enumerate(batch_states):
            if not cur_states[-1].is_end():
                next_state = State()
                cur_states[-1].move(next_state, pred_actions[offset])
                cur_states.append(next_state)
                offset += 1
                self.step[idx] += 1


    def get_candidates(self, batch_states, vocab):
        all_candidates = []
        for cur_states in batch_states:
            if not cur_states[-1].is_end():
                candidates = cur_states[-1].get_candidate_actions(vocab)
                all_candidates.append(candidates)
        return all_candidates

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)
