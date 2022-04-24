import logging
import os

import numpy as np
import torch

from layers import (DecoderBlock, Embedding, EncoderBlock, LSTMCell,
                    NoisyLinear, PointerSoftmax, SelfAttention, masked_softmax)

logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module):
    def __init__(self, config, word_vocab, use_pretrained=True):
        super(BaseModel, self).__init__()
        self.config = config
        self.word_vocab = word_vocab
        self.word_vocab_size = len(word_vocab)
        self.read_config(use_pretrained=use_pretrained)
        self._def_layers()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self, use_pretrained=True):
        # model config
        model_config = self.config['general']['model']

        self.use_pretrained_embedding = use_pretrained and model_config[
            'use_pretrained_embedding']
        self.word_embedding_size = model_config['word_embedding_size']
        self.word_embedding_trainable = model_config[
            'word_embedding_trainable']
        self.pretrained_embedding_path = "crawl-300d-2M.vec.npy"
        if 'vocab_path' in model_config:
            self.pretrained_embedding_path = model_config[
                'vocab_path'].replace('.txt', '.npy')

        self.embedding_dropout = model_config['embedding_dropout']

        self.encoder_layers = model_config['encoder_layers']
        self.decoder_layers = model_config['decoder_layers']
        self.encoder_conv_num = model_config['encoder_conv_num']
        self.block_hidden_dim = model_config['block_hidden_dim']
        self.n_heads = model_config['n_heads']
        self.block_dropout = model_config['block_dropout']
        self.dropout = model_config['dropout']

        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']

        self.enable_recurrent_memory = self.config['rl']['model'][
            'enable_recurrent_memory']

    def _def_embedding_layer(self):
        if self.use_pretrained_embedding:
            self.word_embedding = Embedding(
                embedding_size=self.word_embedding_size,
                vocab_size=self.word_vocab_size,
                id2word=self.word_vocab,
                dropout_rate=self.embedding_dropout,
                load_pretrained=True,
                trainable=self.word_embedding_trainable,
                embedding_oov_init="random",
                pretrained_embedding_path=self.pretrained_embedding_path)
        else:
            self.word_embedding = Embedding(
                embedding_size=self.word_embedding_size,
                vocab_size=self.word_vocab_size,
                trainable=self.word_embedding_trainable,
                dropout_rate=self.embedding_dropout)

        self.word_embedding_prj = torch.nn.Linear(self.word_embedding_size,
                                                  self.block_hidden_dim,
                                                  bias=False)

    def _def_layers(self):
        self._def_embedding_layer()
        self.encoder = torch.nn.ModuleList([
            EncoderBlock(conv_num=self.encoder_conv_num,
                         ch_num=self.block_hidden_dim,
                         k=5,
                         block_hidden_dim=self.block_hidden_dim,
                         n_head=self.n_heads,
                         dropout=self.block_dropout)
            for _ in range(self.encoder_layers)
        ])
        self.self_attention_text = SelfAttention(self.block_hidden_dim,
                                                 self.n_heads, self.dropout)

        self.decoder = torch.nn.ModuleList([
            DecoderBlock(ch_num=self.block_hidden_dim,
                         k=5,
                         block_hidden_dim=self.block_hidden_dim,
                         n_head=self.n_heads,
                         dropout=self.block_dropout)
            for _ in range(self.decoder_layers)
        ])
        self.tgt_word_prj = torch.nn.Linear(self.block_hidden_dim,
                                            self.word_vocab_size,
                                            bias=False)
        self.pointer_softmax = PointerSoftmax(input_dim=self.block_hidden_dim,
                                              hidden_dim=self.block_hidden_dim)

        # recurrent memories
        self.recurrent_memory_single_input = LSTMCell(self.block_hidden_dim,
                                                      self.block_hidden_dim,
                                                      use_bias=True)

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear
        self.action_scorer_linear_1_bi_input = linear_function(
            self.block_hidden_dim * 2, self.block_hidden_dim)
        self.action_scorer_linear_2 = linear_function(self.block_hidden_dim, 1)

    def embed(self, input_words):
        word_embeddings, mask = self.word_embedding(
            input_words)  # batch x time x emb
        word_embeddings = self.word_embedding_prj(word_embeddings)
        word_embeddings = word_embeddings * mask.unsqueeze(
            -1)  # batch x time x hid
        return word_embeddings, mask

    def encode_text(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(
            mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder[i](
                encoding_sequence, squared_mask,
                i * (self.encoder_conv_num + 2) + 1,
                self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def encode_text_for_pretraining_tasks(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(
            mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder_for_pretraining_tasks[i](
                encoding_sequence, squared_mask,
                i * (self.encoder_conv_num + 2) + 1,
                self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def get_subsequent_mask(self, seq):
        ''' For masking out the subsequent info. '''
        _, length = seq.size()
        subsequent_mask = torch.triu(torch.ones((length, length)),
                                     diagonal=1).float()
        subsequent_mask = 1.0 - subsequent_mask
        if seq.is_cuda:
            subsequent_mask = subsequent_mask.cuda()
        subsequent_mask = subsequent_mask.unsqueeze(0)  # 1 x time x time
        return subsequent_mask

    def decode_for_obs_gen(self, input_target_word_ids, h_ag2,
                           prev_action_mask, h_ga2, node_mask):
        trg_embeddings, trg_mask = self.embed(
            input_target_word_ids)  # batch x target_len x emb

        trg_mask_square = torch.bmm(
            trg_mask.unsqueeze(-1),
            trg_mask.unsqueeze(1))  # batch x target_len x target_len
        trg_mask_square = trg_mask_square * self.get_subsequent_mask(
            input_target_word_ids)  # batch x target_len x target_len

        prev_action_mask_square = torch.bmm(trg_mask.unsqueeze(-1),
                                            prev_action_mask.unsqueeze(1))
        node_mask_square = torch.bmm(
            trg_mask.unsqueeze(-1),
            node_mask.unsqueeze(1))  # batch x target_len x num_nodes

        trg_decoder_output = trg_embeddings
        for i in range(self.decoder_layers):
            trg_decoder_output, _ = self.obs_gen_decoder[i](
                trg_decoder_output, trg_mask, trg_mask_square, h_ag2,
                prev_action_mask_square, h_ga2, node_mask_square, i * 3 + 1,
                self.decoder_layers)

        trg_decoder_output = self.obs_gen_tgt_word_prj(trg_decoder_output)
        trg_decoder_output = masked_softmax(trg_decoder_output,
                                            m=trg_mask.unsqueeze(-1),
                                            axis=-1)
        # eliminating pointer softmax
        return trg_decoder_output

    def decode(self, input_target_word_ids, h_og, obs_mask, h_go, node_mask,
               input_obs):
        trg_embeddings, trg_mask = self.embed(
            input_target_word_ids)  # batch x target_len x emb

        trg_mask_square = torch.bmm(
            trg_mask.unsqueeze(-1),
            trg_mask.unsqueeze(1))  # batch x target_len x target_len
        trg_mask_square = trg_mask_square * self.get_subsequent_mask(
            input_target_word_ids)  # batch x target_len x target_len

        obs_mask_square = torch.bmm(
            trg_mask.unsqueeze(-1),
            obs_mask.unsqueeze(1))  # batch x target_len x obs_len
        node_mask_square = None if node_mask is None else torch.bmm(
            trg_mask.unsqueeze(-1),
            node_mask.unsqueeze(1))  # batch x target_len x node_len

        trg_decoder_output = trg_embeddings
        for i in range(self.decoder_layers):
            trg_decoder_output, target_target_representations, target_source_representations, target_source_attention = self.decoder[
                i](trg_decoder_output, trg_mask, trg_mask_square, h_og,
                   obs_mask_square, h_go, node_mask_square, i * 3 + 1,
                   self.decoder_layers)  # batch x time x hid

        trg_decoder_output = self.tgt_word_prj(trg_decoder_output)
        trg_decoder_output = masked_softmax(trg_decoder_output,
                                            m=trg_mask.unsqueeze(-1),
                                            axis=-1)
        output = self.pointer_softmax(target_target_representations,
                                      target_source_representations,
                                      trg_decoder_output, trg_mask,
                                      target_source_attention, obs_mask,
                                      input_obs)

        return output

    def score_actions(self,
                      input_candidate_word_ids,
                      h_og=None,
                      obs_mask=None,
                      h_go=None,
                      node_mask=None,
                      previous_h=None,
                      previous_c=None):
        # input_candidate_word_ids: batch x num_candidate x candidate_len
        batch_size, num_candidate, candidate_len = input_candidate_word_ids.size(
            0), input_candidate_word_ids.size(
                1), input_candidate_word_ids.size(2)
        input_candidate_word_ids = input_candidate_word_ids.view(
            batch_size * num_candidate, candidate_len)

        cand_encoding_sequence, cand_mask = self.encode_text(
            input_candidate_word_ids)
        cand_encoding_sequence = cand_encoding_sequence.view(
            batch_size, num_candidate, candidate_len, -1)
        cand_mask = cand_mask.view(batch_size, num_candidate, candidate_len)

        _mask = torch.sum(cand_mask, -1)  # batch x num_candidate
        candidate_representations = torch.sum(
            cand_encoding_sequence, -2)  # batch x num_candidate x hid
        tmp = torch.eq(_mask, 0).float()
        if candidate_representations.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        candidate_representations = candidate_representations / _mask.unsqueeze(
            -1)  # batch x num_candidate x hid
        cand_mask = cand_mask.byte().any(-1).float()  # batch x num_candidate

        assert h_og is not None
        obs_mask_squared = torch.bmm(
            obs_mask.unsqueeze(-1),
            obs_mask.unsqueeze(1))  # batch x obs_len x obs_len
        obs_representations, _ = self.self_attention_text(
            h_og, obs_mask_squared, h_og, h_og)  # batch x obs_len x hid
        # masked mean
        _mask = torch.sum(obs_mask, -1)  # batch
        obs_representations = torch.sum(obs_representations, -2)  # batch x hid
        tmp = torch.eq(_mask, 0).float()
        if obs_representations.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        obs_representations = obs_representations / _mask.unsqueeze(
            -1)  # batch x hid

        if self.enable_recurrent_memory:
            # recurrent memory
            new_h, new_c = self.recurrent_memory_single_input(
                obs_representations, h_0=previous_h, c_0=previous_c)
            new_h_expanded = torch.stack([new_h] * num_candidate,
                                         1).view(batch_size, num_candidate,
                                                 new_h.size(-1))
        else:
            new_h, new_c = None, None
            new_h_expanded = torch.stack([obs_representations] * num_candidate,
                                         1).view(batch_size, num_candidate,
                                                 obs_representations.size(-1))
        output = self.action_scorer_linear_1_bi_input(
            torch.cat([candidate_representations, new_h_expanded],
                      -1))  # batch x num_candidate x hid

        output = torch.relu(output)
        output = output * cand_mask.unsqueeze(-1)
        output = self.action_scorer_linear_2(output).squeeze(
            -1)  # batch x num_candidate
        output = output * cand_mask

        return output, cand_mask, new_h, new_c

    def reset_noise(self):
        if self.noisy_net:
            self.action_scorer_linear_1_bi_input.reset_noise()
            self.action_scorer_linear_2.reset_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.action_scorer_linear_1_bi_input.zero_noise()
            self.action_scorer_linear_2.zero_noise()


class GameDiscModel(BaseModel):
    def __init__(self, config, word_vocab, num_games, use_pretrained=True):
        super().__init__(config, word_vocab, use_pretrained=use_pretrained)
        self.discriminator = torch.nn.Linear(self.block_hidden_dim, num_games)

    def discriminate_game(self, h_og=None, obs_mask=None):
        obs_mask_squared = torch.bmm(
            obs_mask.unsqueeze(-1),
            obs_mask.unsqueeze(1))  # batch x obs_len x obs_len
        obs_representations, _ = self.self_attention_text(
            h_og, obs_mask_squared, h_og, h_og)  # batch x obs_len x hid
        # masked mean
        _mask = torch.sum(obs_mask, -1)  # batch
        obs_representations = torch.sum(obs_representations, -2)  # batch x hid
        tmp = torch.eq(_mask, 0).float()
        if obs_representations.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        obs_representations = obs_representations / _mask.unsqueeze(-1)

        output = torch.relu(obs_representations)
        output = self.discriminator(output)
        return output
