import torch 
from torch import nn 



class Attention(nn.Module):
    def __init__(self, encode_size, hidden_size, bias = False,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.Uatt = nn.Linear(encode_size, hidden_size, bias= bias)
        self.Watt = nn.Linear(hidden_size, hidden_size)
        self.Vatt = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
    def forward(self, encode_feature, hidden_features):
        # encode_feature shape (batch, 49, 256)
        # hidden_features shape (batch, number_layer, hidden_size)
        
        U = self.Uatt(encode_feature) #(bacth, 49, hidden_size)
        W = self.Watt(hidden_features) #(batch, number_layer, hidden_size)
        
        score = self.Vatt(self.tanh(U + W))
        
        weights = torch.softmax(score, dim = 1)
        
        context_vetor = torch.bmm(weights, encode_feature)
        
        context_vetor = torch.sum(context_vetor, dim = 1)
         
        return context_vetor, weights #(bacth, hidden_size)
class Endcode(nn.Module):
    def __init__(self, in_channels, embedding, bias = False, drop_p = 0.5):
        super(Endcode, self).__init__()
        self.in_channels = in_channels
        self.embedding = embedding
        
        self.fc = nn.Linear(in_channels, embedding, bias= bias)
        self.drop = nn.Dropout(p = drop_p)
        self.relu = nn.ReLU()
        
    def forward(self, features):
        # features shape (bacth, 49, 256)
        x = self.fc(features)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.drop(x)
        
        # return x shape (batch, 49, 256)
        return x
    
class Decode(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, number_layer ,*args, **kwargs) -> None:
        super(Decode, self).__init__(*args, **kwargs)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, number_layer)
        self.attention = Attention(embedding_size,  hidden_size)
        
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
        self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout1d()
    
    def forward(self, x, features, hiddens):
        
        context_vector, attentions_weights = self.attention(features, hiddens)
        
        x = self.embedding(x) 
        
        x = torch.cat([torch.unsqueeze(context_vector, dim = 1), x], dim = 1) #(batch, 1, 512)
        
        out, hidden = self.gru(x, hiddens)
        
        x = self.fc(x)
        
        x = self.drop(x)
        
        x = self.bn(x)
        
        x = self.fc2(x)
        
        return x, hidden, attentions_weights
        
        
class CNNtoRNN(nn.Module):
    def __init__(self, in_channels, embedding_size, hidden_size, vocab_size, encode_size, number_layer, *args, **kwargs) -> None:
        super(CNNtoRNN, self).__init__(*args, **kwargs)
        self.encode = Endcode(in_channels, embedding_size)
        self.dencode = Decode(embedding_size, hidden_size, vocab_size, number_layer)
        