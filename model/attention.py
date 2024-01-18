from torch import nn 
import torch 
from vgg16_base import GRU, VGG16
class Encode(nn.Module):
    def __init__(self, in_chanels, embedding_dim, bias = False) -> None:
        super(Encode, self).__init__()
        self.in_chanels = in_chanels
        self.fc = nn.Linear(in_features = in_chanels, out_features= embedding_dim, bias= bias)
        self.dropout = nn.Dropout(p = 0.5)
        self.relu = nn.ReLU()
        self.embedding_dim = embedding_dim
    def forward(self, x):
        x = x.view(x.size(0), -1, self.in_chanels)
        batch, features, channels = x.shape
        print("in encode ", x.shape)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = x.view(batch, self.embedding_dim, features)
        return x
    
class Decode(nn.Module):
    def __init__(self, embedding_dim, units, vocab_size, dropout= 0.5) -> None:
        # embedding = 256, units = 512, vocab_size = len(tokenizer.word-index) + 1
        # features_shape = 512
        #attention_features_shape = 49
        
        super(Decode, self).__init__()
        # input_size, hidden_size, num_layer
        self.gru = GRU(units, units,  batch_first= True) #bacth, units
        self.units = units
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 256
        
        self.fc = nn.Linear(in_features = units, out_features= units) # 512
        self.dropout = nn.Dropout(p= dropout)
        
        self.bn = nn.BatchNorm1d(num_features=units, momentum=0.99, eps=0.001)
        
        self.fc2 = nn.Linear(in_features = units, out_features= vocab_size)
        
        #Attention mechanism 

        self.Uattn = nn.Linear(in_features= embedding_dim, out_features=units) # 256
        self.Wattn = nn.Linear(in_features= units, out_features= units) # 256
        self.Vattn = nn.Linear(in_features= units, out_features = 1)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x, features, hidden):
        hidden_with_time_axis = torch.unsqueeze(hidden, dim=1)
        # print("hidden_with_time_axis.shape ", hidden_with_time_axis.shape)
        
        # Attention score
        U = self.Uattn(features)
        # print("U ", U.shape)
        W = self.Wattn(hidden_with_time_axis)
        # print("W ", W.shape)
        
        score = self.Vattn(self.tanh(U + W))
        # print("score ", score.shape)

        # Attention weights
        attention_weights = torch.softmax(score, dim=1)
        # print("attention_weights ", attention_weights.shape)
        
        context_vector = attention_weights * features
        # print("context_vector ", context_vector.shape)
        
        context_vector = torch.sum(context_vector, dim=1)
        # print("context_vector ", context_vector.shape)
        
        x = self.embedding(x)
        # print("embedded_x ", x.shape)
        
        x = torch.cat([torch.unsqueeze(context_vector, dim=1), x], dim=-1)
        # print("concatenated_x ", x.shape)
        
        out, state = self.gru(x)
        # print("GRU_output ", out.shape)
        
        x = self.fc(out)
        # print("fully_connected_output ", x.shape)
        
        x = x.view((-1, x.shape[2]))
        # print("viewed_output ", x.shape)
        
        x = self.dropout(x)
        x = self.bn(x)
        # print("batch_normalized_output ", x.shape)
        
        x = self.fc2(x)
        # print("final_output ", x.shape)
        
        return x, state, attention_weights
    def reset_state(self, batch):
        ''' Reset the hidden states of all layers in the model to zero.'''
        return torch.zeros(size=(batch, self.units))