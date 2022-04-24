import torch
import math
import numpy as np
import h5py
import copy
import torch.nn.functional as F


def compute_mask(x):
    mask = torch.ne(x, 0).float()
    if x.is_cuda:
        mask = mask.cuda()
    return mask


def to_one_hot(y_true, n_classes):
    # y_true: batch x time
    batch_size, length = y_true.size(0), y_true.size(1)
    y_onehot = torch.FloatTensor(batch_size, length, n_classes)  # batch x time x n_class
    if y_true.is_cuda:
        y_onehot = y_onehot.cuda()
    y_onehot.zero_()
    y_onehot = y_onehot.view(-1, y_onehot.size(-1))  # batch*time x n_class
    y_true = y_true.view(-1, 1)  # batch*time x 1
    y_onehot.scatter_(1, y_true, 1)  # batch*time x n_class
    return y_onehot.view(batch_size, length, -1)


def NegativeLogLoss(y_pred, y_true, mask=None, smoothing_eps=0.0):
    """
    Shape:
        - y_pred:    batch x time x vocab
        - y_true:    batch x time
        - mask:      batch x time
    """
    y_true_onehot = to_one_hot(y_true, y_pred.size(-1))  # batch x time x vocab
    if smoothing_eps > 0.0:
        y_true_onehot = y_true_onehot * (1.0 - smoothing_eps) + (1.0 - y_true_onehot) * smoothing_eps / (y_pred.size(-1) - 1)
    P = y_true_onehot * y_pred  # batch x time x vocab
    P = torch.sum(P, dim=-1)  # batch x time
    gt_zero = torch.gt(P, 0.0).float()  # batch x time
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch x time
    log_P = torch.log(P + epsilon) * gt_zero  # batch x time
    if mask is not None:
        log_P = log_P * mask
    output = -torch.sum(log_P, dim=1)  # batch
    return output

    
def NLL(y_pred, y_true_onehot, mask=None):
    """
    Shape:
        - y_pred:    batch x n_class
        - y_true:    batch x n_class
        - mask:      batch x n_class
    """
    y_true_onehot = y_true_onehot.float()
    P = y_true_onehot * y_pred  # batch x n_class
    gt_zero = torch.gt(P, 0.0).float()  # batch x n_class
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch x n_class
    log_P = torch.log(P + epsilon) * gt_zero  # batch x n_class
    if mask is not None:
        log_P = log_P * mask
    output = -torch.sum(log_P, dim=1)  # batch
    return torch.mean(output)


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def masked_mean(x, m=None, dim=1):
    """
        mean pooling when there're paddings
        input:  tensor: batch x time x h
                mask:   batch x time
        output: tensor: batch x h
    """
    if m is None:
        return torch.mean(x, dim=dim)
    x = x * m.unsqueeze(-1)
    mask_sum = torch.sum(m, dim=-1)  # batch
    tmp = torch.eq(mask_sum, 0).float()
    if x.is_cuda:
        tmp = tmp.cuda()
    mask_sum = mask_sum + tmp
    res = torch.sum(x, dim=dim)  # batch x h
    res = res / mask_sum.unsqueeze(-1)
    return res


def masked_ave_aggregator(x, mask):
    """
    ave aggregator of the node embedding
    input: tensor: batch x num_nodes x dim
    mask: tensor: batch x num_nodes

    """
    mask_sum = torch.sum(mask, -1).unsqueeze(-1) # batch x 1
    mask = mask.unsqueeze(-1) # batch x num_nodes x 1
    mask = mask.expand(-1, -1, x.shape[-1]) # batch x num_nodes x dim
    masked_x = x * mask # batch x num_nodes x dim
    sum_masked_x = torch.sum(masked_x, 1)
    ave_masked_x = sum_masked_x / mask_sum
    return ave_masked_x


class LayerNorm(torch.nn.Module):

    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(input_dim))
        self.beta = torch.nn.Parameter(torch.zeros(input_dim))
        self.eps = 1e-6

    def forward(self, x, mask=None):
        # x:        nbatch x hidden
        # mask:     nbatch
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True) + self.eps)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        if mask is not None:
            return output * mask.unsqueeze(-1)
        else:
            return output


class H5EmbeddingManager(object):
    def __init__(self, h5_path):
        f = h5py.File(h5_path, 'r')
        self.W = np.array(f['embedding'])
        print("embedding data type=%s, shape=%s" % (type(self.W), self.W.shape))
        self.id2word = f['words_flatten'][0].split('\n')
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))

    def __getitem__(self, item):
        item_type = type(item)
        if item_type is str:
            index = self.word2id[item]
            embs = self.W[index]
            return embs
        else:
            raise RuntimeError("don't support type: %s" % type(item))

    def word_embedding_initialize(self, words_list, dim_size=300, scale=0.1, oov_init='random'):
        shape = (len(words_list), dim_size)
        self.rng = np.random.RandomState(42)
        if 'zero' == oov_init:
            W2V = np.zeros(shape, dtype='float32')
        elif 'one' == oov_init:
            W2V = np.ones(shape, dtype='float32')
        else:
            W2V = self.rng.uniform(low=-scale, high=scale, size=shape).astype('float32')
        W2V[0, :] = 0
        in_vocab = np.ones(shape[0], dtype=np.bool)
        word_ids = []
        for i, word in enumerate(words_list):
            if word in self.word2id:
                word_ids.append(self.word2id[word])
            else:
                in_vocab[i] = False
        W2V[in_vocab] = self.W[np.array(word_ids, dtype='int32')][:, :dim_size]
        return W2V


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x ...
    outputs:embedding:  batch x ... x emb
            mask:       batch x ...
    '''

    def __init__(self, embedding_size, vocab_size, dropout_rate=0.0, trainable=True, id2word=None,
                 embedding_oov_init='random', load_pretrained=False, pretrained_embedding_path=None):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.id2word = id2word
        self.dropout_rate = dropout_rate
        self.load_pretrained = load_pretrained
        self.embedding_oov_init = embedding_oov_init
        self.pretrained_embedding_path = pretrained_embedding_path
        self.trainable = trainable
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        init_embedding_matrix = self.embedding_init()
        if self.embedding_layer.weight.is_cuda:
            init_embedding_matrix = init_embedding_matrix.cuda()
        self.embedding_layer.weight = torch.nn.Parameter(init_embedding_matrix)
        if not self.trainable:
            self.embedding_layer.weight.requires_grad = False

    def embedding_init(self):
        # Embeddings
        if self.load_pretrained is False:
            word_embedding_init = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
            word_embedding_init[0, :] = 0
        elif self.pretrained_embedding_path.endswith('.npy'):
            word_embedding_init = np.load(self.pretrained_embedding_path)
        elif self.pretrained_embedding_path.endswith('.h5'):
            embedding_initr = H5EmbeddingManager(self.pretrained_embedding_path)
            word_embedding_init = embedding_initr.word_embedding_initialize(self.id2word,
                                                                            dim_size=self.embedding_size,
                                                                            oov_init=self.embedding_oov_init)
            del embedding_initr
        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        return word_embedding_init

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float()
        if x.is_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x):
        embeddings = self.embedding_layer(x)  # batch x time x emb
        embeddings = F.dropout(embeddings, p=self.dropout_rate, training=self.training)
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask


class LSTMCell(torch.nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.pre_act_linear = torch.nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=False)
        if use_bias:
            self.bias_f = torch.nn.Parameter(torch.FloatTensor(hidden_size))
            self.bias_iog = torch.nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.pre_act_linear.weight.data)
        if self.use_bias:
            self.bias_f.data.fill_(1.0)
            self.bias_iog.data.fill_(0.0)

    def get_init_hidden(self, bsz, use_cuda):
        h_0 = torch.autograd.Variable(torch.FloatTensor(bsz, self.hidden_size).zero_())
        c_0 = torch.autograd.Variable(torch.FloatTensor(bsz, self.hidden_size).zero_())
        if use_cuda:
            h_0, c_0 = h_0.cuda(), c_0.cuda()
        return h_0, c_0

    def forward(self, input_, mask_=None, h_0=None, c_0=None):
        """
        Args:
            input_:     A (batch, input_size) tensor containing input features.
            mask_:      (batch)
            hx:         A tuple (h_0, c_0), which contains the initial hidden
                        and cell state, where the size of both states is
                        (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        if h_0 is None or c_0 is None:
            h_init, c_init = self.get_init_hidden(input_.size(0), use_cuda=input_.is_cuda)
            if h_0 is None:
                h_0 = h_init
            if c_0 is None:
                c_0 = c_init
        
        if mask_ is None:
            mask_ = torch.ones_like(torch.sum(input_, -1))
            if input_.is_cuda:
                mask_ = mask_.cuda()

        pre_act = self.pre_act_linear(torch.cat([input_, h_0], -1))  # batch x 4*hid
        if self.use_bias:
            pre_act = pre_act + torch.cat([self.bias_f, self.bias_iog]).unsqueeze(0)
        f, i, o, g = torch.split(pre_act, split_size_or_sections=self.hidden_size, dim=1)
        expand_mask_ = mask_.unsqueeze(1)  # batch x 1
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        c_1 = c_1 * expand_mask_ + c_0 * (1 - expand_mask_)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        h_1 = h_1 * expand_mask_ + h_0 * (1 - expand_mask_)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ActionScorerAttention(torch.nn.Module):

    def __init__(self, input_q_dim, output_dim, noisy_net=False):
        super(ActionScorerAttention, self).__init__()
        self.input_q_dim = input_q_dim
        self.output_dim = output_dim
        self.noisy_net = noisy_net

        if self.noisy_net:
            self.W_a = NoisyLinear(self.input_q_dim, self.output_dim)
        else:
            self.W_a = torch.nn.Linear(self.input_q_dim, self.output_dim)
        self.init_weights()

    def init_weights(self):
        if not self.noisy_net:
            torch.nn.init.xavier_uniform_(self.W_a.weight.data, gain=1)
            self.W_a.bias.data.fill_(0)

    def forward(self, H_q):
        # H_q: batch x inp_q
        Fk_prime = self.W_a(H_q)  # batch x out
        Fk_prime = torch.sigmoid(Fk_prime)  # batch x out
        return Fk_prime

    def reset_noise(self):
        if self.noisy_net:
            self.W_a.reset_noise()


class ActionScorerAttentionAdvantage(torch.nn.Module):

    def __init__(self, input_seq_dim, input_q_dim, output_dim, noisy_net=False):
        super(ActionScorerAttentionAdvantage, self).__init__()
        self.input_seq_dim = input_seq_dim
        self.input_q_dim = input_q_dim
        self.output_dim = output_dim
        self.noisy_net = noisy_net

        if self.noisy_net:
            self.V = NoisyLinear(self.input_seq_dim, self.output_dim)
            self.W_a = NoisyLinear(self.input_q_dim, self.output_dim)
        else:
            self.V = torch.nn.Linear(self.input_seq_dim, self.output_dim)
            self.W_a = torch.nn.Linear(self.input_q_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.init_weights()

    def init_weights(self):
        if not self.noisy_net:
            torch.nn.init.xavier_uniform_(self.V.weight.data, gain=1)
            torch.nn.init.xavier_uniform_(self.W_a.weight.data, gain=1)
            self.V.bias.data.fill_(0)
            self.W_a.bias.data.fill_(0)
        torch.nn.init.normal_(self.v.data, mean=0, std=0.05)

    def forward(self, H_r, mask_r, H_q):
        # H_r: batch x time x input_seq_dim
        # mask_r: batch x time
        # H_q: batch x inp_q
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V(H_r.view(-1, H_r.size(2)))  # batch*time x out
        Fk_prime = self.W_a(H_q)  # batch x out
        Fk = Fk.view(batch_size, time, -1)  # batch x time x out
        Fk = torch.sigmoid(Fk + Fk_prime.unsqueeze(1))  # batch x time x out
        Fk = Fk * mask_r.unsqueeze(-1)  # batch x time x out
        return Fk

    def reset_noise(self):
        if self.noisy_net:
            self.V.reset_noise()
            self.W_a.reset_noise()


class BoundaryDecoderAttention(torch.nn.Module):

    def __init__(self, input_dim, output_dim, noisy_net=False):
        super(BoundaryDecoderAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noisy_net = noisy_net

        if self.noisy_net:
            self.V = NoisyLinear(self.input_dim, self.output_dim)
            self.W_a = NoisyLinear(self.output_dim, self.output_dim)
        else:
            self.V = torch.nn.Linear(self.input_dim, self.output_dim)
            self.W_a = torch.nn.Linear(self.output_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.c = torch.nn.Parameter(torch.FloatTensor(1))
        self.init_weights()

    def init_weights(self):
        if not self.noisy_net:
            torch.nn.init.xavier_uniform_(self.V.weight.data)
            torch.nn.init.xavier_uniform_(self.W_a.weight.data)
            self.V.bias.data.fill_(0)
            self.W_a.bias.data.fill_(0)
        torch.nn.init.normal_(self.v.data, mean=0, std=0.05)
        self.c.data.fill_(1.0)

    def forward(self, H_r, mask_r, h_tm1):
        # H_r: batch x time x inp
        # mask_r: batch x time
        # h_tm1: batch x out
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V(H_r.view(-1, H_r.size(2)))  # batch*time x out
        Fk_prime = self.W_a(h_tm1)  # batch x out
        Fk = Fk.view(batch_size, time, -1)  # batch x time x out
        Fk = torch.tanh(Fk + Fk_prime.unsqueeze(1))  # batch x time x out

        beta = torch.matmul(Fk, self.v)  # batch x time
        beta = beta + self.c.unsqueeze(0)  # batch x time
        beta = masked_softmax(beta, mask_r, axis=-1)  # batch x time
        z = torch.bmm(beta.view(beta.size(0), 1, beta.size(1)), H_r)  # batch x 1 x inp
        z = z.view(z.size(0), -1)  # batch x inp
        return z, beta

    def reset_noise(self):
        if self.noisy_net:
            self.V.reset_noise()
            self.W_a.reset_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.V.zero_noise()
            self.W_a.zero_noise()


class BoundaryDecoder(torch.nn.Module):
    '''
    input:  encoded stories:    batch x time x input_dim
            story mask:         batch x time
            init states:        batch x hid
            encoded question:   batch x input_dim_q
    '''

    def __init__(self, input_dim, hidden_dim, noisy_net=False):
        super(BoundaryDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noisy_net = noisy_net
        self.attention_layer = BoundaryDecoderAttention(input_dim=input_dim,
                                                        output_dim=hidden_dim,
                                                        noisy_net=self.noisy_net)

        self.rnn = LSTMCell(self.input_dim, self.hidden_dim, use_bias=True)

    def forward(self, x, x_mask, h_0):

        state_stp = [(h_0, h_0)]
        beta_list = []
        mask = torch.autograd.Variable(torch.ones(x.size(0)))  # fake mask
        mask = mask.cuda() if x.is_cuda else mask
        for t in range(2):
            previous_h, previous_c = state_stp[t]
            curr_input, beta = self.attention_layer(x, x_mask, h_tm1=previous_h)
            new_h, new_c = self.rnn(curr_input, mask, previous_h, previous_c)
            state_stp.append((new_h, new_c))
            beta_list.append(beta)

        # beta list: list of batch x time
        res = torch.stack(beta_list, 2)  # batch x time x 2
        res = res * x_mask.unsqueeze(2)  # batch x time x 2
        return res

    def reset_noise(self):
        if self.noisy_net:
            self.attention_layer.reset_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.attention_layer.zero_noise()


class NoisyLinear(torch.nn.Module):
    # Factorised NoisyLinear layer with bias
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()
        self._zero_noise = False

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def zero_noise(self):
        self._zero_noise = True

    def forward(self, input):
        if self.training:
            if self._zero_noise is True:
                return F.linear(input, self.weight_mu, self.bias_mu)
            else:
                return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


################################# qanet

def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    length = x.size(1)
    channels = x.size(2)
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return x + (signal.cuda() if x.is_cuda else signal)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = torch.nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x = x.transpose(1,2)
        res = torch.relu(self.pointwise_conv(self.depthwise_conv(x)))
        res = res.transpose(1,2)
        return res


class SimpleSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(SimpleSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = torch.nn.Parameter(torch.Tensor(hidden_size, 1), requires_grad=True)
        torch.nn.init.xavier_uniform(self.att_weights.data)

    def forward(self, input, mask):
        batch_size = input.size(0)
        # apply attention layer
        weights = torch.bmm(input, self.att_weights.unsqueeze(0).repeat(batch_size, 1, 1)).squeeze(-1)
        attentions = masked_softmax(weights, mask)
        # apply attention weights
        weighted = torch.mul(input, attentions.unsqueeze(-1).expand_as(input))
        return weighted, attentions


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = masked_softmax(attn, mask, 2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class SelfAttention(torch.nn.Module):
    ''' From Multi-Head Attention module 
    https://github.com/jadore801120/attention-is-all-you-need-pytorch'''

    def __init__(self, block_hidden_dim, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.block_hidden_dim = block_hidden_dim
        self.w_qs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_ks = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_vs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        self.attention = ScaledDotProductAttention(temperature=np.power(block_hidden_dim, 0.5))
        self.fc = torch.nn.Linear(n_head * block_hidden_dim, block_hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(self.block_hidden_dim)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, mask, k, v):
        # q: batch x len_q x hid
        # k: batch x len_k x hid
        # v: batch x len_v x hid
        # mask: batch x len_q x len_k
        batch_size, len_q = q.size(0), q.size(1)
        len_k, len_v = k.size(1), v.size(1)
        assert mask.size(1) == len_q
        assert mask.size(2) == len_k
        residual = q

        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.block_hidden_dim)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.block_hidden_dim)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.block_hidden_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.block_hidden_dim) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.block_hidden_dim) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.block_hidden_dim) # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(self.n_head, batch_size, len_q, self.block_hidden_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class EncoderBlock(torch.nn.Module):
    def __init__(self, conv_num, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_C = torch.nn.ModuleList([torch.nn.LayerNorm(block_hidden_dim) for _ in range(conv_num)])
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)
        self.conv_num = conv_num

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 2) * blks
        # conv layers
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # self attention
        out, _ = self.self_att(out, mask, out, out)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class DecoderBlock(torch.nn.Module):
    def __init__(self, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.obs_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.node_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_0 = torch.nn.Linear(block_hidden_dim * 2, block_hidden_dim)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)

    def forward(self, x, mask, self_att_mask, obs_enc_representations, obs_mask, node_enc_representations, node_mask, l, blks):
        total_layers = blks * 3
        # conv layers
        out = PosEncoder(x)
        res = out
        # self attention
        out, _ = self.self_att(out, self_att_mask, out, out)
        out_self = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out_self, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # attention with encoder outputs
        out_obs, obs_attention = self.obs_att(out, obs_mask, obs_enc_representations, obs_enc_representations)
        if node_enc_representations is not None:
            out_node, _ = self.node_att(out, node_mask, node_enc_representations, node_enc_representations)

            out = torch.cat([out_obs, out_node], -1)
            out = self.FFN_0(out)
        out = torch.relu(out)
        out = out * mask.unsqueeze(-1)

        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # Fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        return out, out_self, out_obs, obs_attention

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class DecoderBlockForObsGen(torch.nn.Module):
    def __init__(self, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.obs_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.node_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_0 = torch.nn.Linear(block_hidden_dim * 2, block_hidden_dim)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)

    def forward(self, x, mask, self_att_mask, prev_action_enc_representations, prev_action_mask, node_enc_representations, node_mask, l, blks):
        total_layers = blks * 3
        # conv layers
        out = PosEncoder(x)
        res = out
        # self attention
        out, _ = self.self_att(out, self_att_mask, out, out)
        out_self = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out_self, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # attention with encoder outputs
        out_obs, obs_attention = self.obs_att(out, prev_action_mask, prev_action_enc_representations, prev_action_enc_representations)
        out_node, _ = self.node_att(out, node_mask, node_enc_representations, node_enc_representations)

        out = torch.cat([out_obs, out_node], -1)
        out = self.FFN_0(out)
        out = torch.relu(out)
        out = out * mask.unsqueeze(-1)

        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # Fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        return out, out_self #, out_obs, obs_attention

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(torch.nn.Module):
    def __init__(self, block_hidden_dim, dropout=0):
        super().__init__()
        self.dropout = dropout
        w4C = torch.empty(block_hidden_dim, 1)
        w4Q = torch.empty(block_hidden_dim, 1)
        w4mlu = torch.empty(1, 1, block_hidden_dim)
        torch.nn.init.xavier_uniform_(w4C)
        torch.nn.init.xavier_uniform_(w4Q)
        torch.nn.init.xavier_uniform_(w4mlu)
        self.w4C = torch.nn.Parameter(w4C)
        self.w4Q = torch.nn.Parameter(w4Q)
        self.w4mlu = torch.nn.Parameter(w4mlu)

        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.unsqueeze(-1)
        Qmask = Qmask.unsqueeze(1)
        S1 = masked_softmax(S, Qmask, axis=2)
        S2 = masked_softmax(S, Cmask, axis=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=self.dropout, training=self.training)
        Q = F.dropout(Q, p=self.dropout, training=self.training)
        max_q_len = Q.size(-2)
        max_context_len = C.size(-2)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, max_q_len])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, max_context_len, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class PointerSoftmax(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.pointer_softmax_context = torch.nn.Linear(input_dim, hidden_dim)
        self.pointer_softmax_target = torch.nn.Linear(input_dim, hidden_dim)
        self.pointer_softmax_squash = torch.nn.Linear(hidden_dim, 1)

    def forward(self, target_target_representations, target_source_representations, trg_decoder_output, target_mask, target_source_attention, source_mask, input_source):
        # target_target_representations:    batch x target_len x hid
        # target_source_representations:    batch x target_len x hid
        # trg_decoder_output:               batch x target len x vocab
        # target mask:                      batch x target len
        # target_source_attention:          batch x target len x source len
        # source mask:                      batch x source len
        # input source:                     batch x source len
        batch_size = target_source_attention.size(0)
        target_len = target_source_attention.size(1)
        source_len = target_source_attention.size(2)

        switch = self.pointer_softmax_context(target_source_representations)  # batch x trg_len x hid
        switch = switch  + self.pointer_softmax_target(target_target_representations)  # batch x trg_len x hid
        switch = torch.tanh(switch)
        switch = switch * target_mask.unsqueeze(-1)
        switch = self.pointer_softmax_squash(switch).squeeze(-1)  # batch x trg_len
        switch = torch.sigmoid(switch)
        switch = switch * target_mask  # batch x target len
        switch = switch.unsqueeze(-1)  # batch x target len x 1

        target_source_attention = target_source_attention * source_mask.unsqueeze(1)
        from_vocab = trg_decoder_output  # batch x target len x vocab
        from_source = torch.autograd.Variable(torch.zeros(batch_size * target_len, from_vocab.size(-1)))  # batch x target len x vocab
        if from_vocab.is_cuda:
            from_source = from_source.cuda()
        input_source = input_source.unsqueeze(1).expand(batch_size, target_len, source_len)
        input_source = input_source.contiguous().view(batch_size * target_len, -1)  # batch*target_len x source_len
        from_source = from_source.scatter_add_(1, input_source, target_source_attention.view(batch_size * target_len, -1))
        from_source = from_source.view(batch_size, target_len, -1)  # batch x target_len x vocab
        merged = switch * from_vocab + (1.0 - switch) * from_source  # batch x target_len x vocab
        merged = merged * target_mask.unsqueeze(-1)
        return merged
