from torch import nn, Tensor
import math
import torch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.swapaxes(0, 1)  # [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        result = self.dropout(x)
        return result.swapaxes(0, 1)  # [batch_size, seq_len, embedding_dim]


class FluxAttention(nn.Module):
    def __init__(self, d_model=32, n_head=4, n_layer=2):
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layer
        super(FluxAttention, self).__init__()
        # nn.Module内部自定义方法函数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layer)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=0)
        self.data_conv = nn.Conv2d(1, d_model, kernel_size=3, stride=1, padding=1)
        self.sparse_conv = nn.Conv1d(1, d_model, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Sequential(
            nn.Conv1d(d_model, int(d_model/2), kernel_size=1, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(int(d_model/2), 1, kernel_size=1, stride=1),
            nn.AvgPool1d(kernel_size=2, stride=1),
        )
    # ********************************
    # model定义调用了前面的decoder和encoder
    def forward(self, data, sparse):
        # (Batch, Time, Channel) to (Batch, 1, Channel, Time)
        data = data.swapaxes(1, 2).unsqueeze(1)
        data = self.data_conv(data)  # (Batch, Feature, Channel, Time)
        num_batch = data.shape[0]
        # (Batch, Feature, Channel*Time)
        data = data.reshape((num_batch, self.d_model, -1))
        # (Batch, Channel*Time, Feature) = (batch, seq_len, embedding_dim)
        data = data.swapaxes(1, 2)
        data = self.pos_encoding(data)  # (batch, seq_len, embedding_dim)
        memory = self.encoder(data)  # (batch, seq_len, embedding_dim)
        sparse = sparse.unsqueeze(1)  # (Batch, 1, Time)
        sparse = self.sparse_conv(sparse)  # (Batch, Channel, Time)
        sparse = sparse.swapaxes(1, 2)  # (Batch, Time, Channel)
        sparse = self.pos_encoding(sparse)   # (Batch, Time, Channel)
        output = self.decoder(sparse, memory)  # (Batch, Time, Channel)
        output = output.swapaxes(1, 2)  # (Batch, Channel, Time)
        output = self.output_conv(output)  # (Batch, 1, 1)
        output = output.squeeze(-1)  # (Batch, 1) 这里的1是指该维度中生成的一个Size为1的预测值嘛？
        return output


