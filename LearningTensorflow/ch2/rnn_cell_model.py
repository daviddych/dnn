class CharLoopModel(nn.Module):
    # This is an RNN!
    def __init__(self, vocab_size, in, h): super().__init__()
        self.h = h
        self.e = nn.Embedding(vocab_size, in)
        self.l_in = nn.Linear(in, h)
        self.l_hidden = nn.Linear(h, h)
        self.l_out = nn.Linear(h, vocab_size)

    def forward(self, *cs):
    '''
    cs (list(list(int))): input signals to propagate forward.
    Example: ((23,32,34), (12,24,23), (12, 45,23))
    '''
        bs = cs[0].size(0)
        hidden_state = Variable(torch.zeros(bs, self.h)).cuda()
        for c in cs:
            inp = F.relu(self.l_in(self.e(c)))
            hidden_state = F.tanh(self.l_hidden(hidden_state+inp))
        return F.log_softmax(self.l_out(hidden_state), dim=-1)