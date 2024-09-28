# Ensure that 'device' is defined
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """Implement the positional encoding function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize 'pe' with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model).to(device)
        # Create a tensor [[0], [1], [2], ..., [max_len-1]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # Compute the term inside the sine and cosine functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # Compute PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Compute PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension
        pe = pe.unsqueeze(0)
        # Register 'pe' as a buffer to save it with the model but exclude it from gradients
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Input embeddings, e.g., (batch_size, seq_len, embedding_dim)
        """
        # Add positional encoding to x
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
        
class TranslationModel(nn.Module):

    def __init__(self, d_model, src_vocab, tgt_vocab, dropout=0.1):
        super(TranslationModel, self).__init__()

        # Define embedding for source sentences
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        # Define embedding for target sentences
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        # Define positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_length)
        # Define the Transformer model
        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)
        # Define the final prediction layer; Softmax is applied outside the model
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        """
        Perform forward propagation; the output is from the decoder.
        Note: 'self.predictor' is not used here because training and inference behaviors differ, so it's applied outside the model.
        :param src: Batched source sentences, e.g., [[0, 12, 34, ..., 1, 2, 2, ...], ...]
        :param tgt: Batched target sentences, e.g., [[0, 74, 56, ..., 1, 2, 2, ...], ...]
        :return: Output from the Transformer or TransformerDecoder.
        """

        """
        Generate 'tgt_mask', a lower triangular matrix:
        [[0., -inf, -inf, -inf, -inf],
         [0.,  0., -inf, -inf, -inf],
         [0.,  0.,  0., -inf, -inf],
         [0.,  0.,  0.,  0., -inf],
         [0.,  0.,  0.,  0.,  0.]]
        'tgt.size()[-1]' is the length of the target sentence.
        """
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(-1)).to(device)
        # Mask the <pad> tokens in the source sentences
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        # Mask the <pad> tokens in the target sentences
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        # Encode 'src' and 'tgt'
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        # Add positional encoding to 'src' and 'tgt'
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Pass the data to the transformer
        out = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        """
        Return the transformer's output directly.
        Since training and inference behaviors differ,
        the linear prediction layer is applied outside the model.
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        Generate key_padding_mask.
        """
        return tokens == 2  # Assuming '2' is the index for <pad>

# Load or initialize the model
if model_checkpoint:
    model = torch.load(model_dir / model_checkpoint)
else:
    model = TranslationModel(256, en_vocab, zh_vocab)  # Replace 'zh_vocab' with 'cn_vocab' if consistent

model = model.to(device)
# Forward pass (assuming 'src' and 'tgt' are defined)
output = model(src, tgt)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

class TranslationLoss(nn.Module):

    def __init__(self):
        super(TranslationLoss, self).__init__()
        # Use KLDivLoss; the internal details are abstracted away
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = 2  # Index for <pad>

    def forward(self, x, target):
        """
        Forward pass of the loss function
        :param x: Output after passing the decoder's output through the predictor linear layer.
                  This is the state after the Linear layer and before Softmax.
        :param target: 'tgt_y', i.e., the labels, e.g., [[1, 34, 15, ...], ...]
        :return: Calculated loss
        """

        """
        Since KLDivLoss requires the input to be log-softmaxed, we use 'log_softmax',
        which is equivalent to 'log(softmax(x))'.
        """
        x = F.log_softmax(x, dim=-1)

        """
        Construct the label distribution by converting [[1, 34, 15, ...]] into:
        [[[0, 1, 0, ..., 0],
          [0, ..., 1, ..., 0],
          ...],
         ...]
        """
        # Create a zero tensor with the same shape as 'x'
        true_dist = torch.zeros_like(x).to(device)
        # Set the indices corresponding to the target to '1'
        true_dist.scatter_(1, target.data.unsqueeze(1), 1)
        # Find the <pad> tokens; for <pad> labels, set them to '0' to exclude them from loss computation
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        # Compute the loss
        return self.criterion(x, true_dist.detach())
