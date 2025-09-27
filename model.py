import torch
import torch.nn as nn
import torch.nn.functional as F
import random  # Add this import

# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, n_layers,
                          bidirectional=True, dropout=dropout, batch_first=False)
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        final_hidden = torch.tanh(self.fc_hidden(hidden_concat))

        cell_forward = cell[-2, :, :]
        cell_backward = cell[-1, :, :]
        cell_concat = torch.cat((cell_forward, cell_backward), dim=1)
        final_cell = torch.tanh(self.fc_cell(cell_concat))

        return outputs, final_hidden, final_cell


# -----------------------------
# Attention
# -----------------------------
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


# -----------------------------
# Decoder
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers=4, dropout=0.5, attention=None):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim * 2, dec_hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden[0], encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        embedded = embedded.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden, cell


# -----------------------------
# Seq2Seq
# -----------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Prepare hidden and cell states for decoder
        hidden = hidden.unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.decoder.rnn.num_layers, 1, 1)

        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


# -----------------------------
# Helper to create model
# -----------------------------
def create_model(input_dim, output_dim, device, enc_hid_dim=512, dec_hid_dim=512,
                 emb_dim=256, enc_layers=2, dec_layers=4, dropout=0.5):
    attn = Attention(enc_hid_dim, dec_hid_dim)
    encoder = Encoder(input_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                      n_layers=enc_layers, dropout=dropout)
    decoder = Decoder(output_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                      n_layers=dec_layers, dropout=dropout, attention=attn)
    model = Seq2Seq(encoder, decoder, device)

    # weight initialization
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

    return model
