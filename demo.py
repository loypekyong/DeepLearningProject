import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import math

st.title("Alzheimer's Detection App")

# Upload image and show
data = st.file_uploader("Upload data", type=["csv", "xlsx"])

# Let the user select the model
model_names = {
    'Model 1: LSTM': 'lstm_model.pth',
    'Model 2: RNN': 'rnn_model.pth',
    'Model 3: Transformer Encoder': 'transformer_encoder_model.pth',
}

model_option = st.selectbox('Choose the model for prediction:', options=list(model_names.keys()))

def normalised(df, min_max=True):
    excluded_columns = ["month"]
    merge_df_normalized_gaussian = df.copy()
    merge_df_normalized_minmax = df.copy()

    epsilon = 1e-10

    for column in merge_df_normalized_gaussian.columns:
        if column not in excluded_columns:
            merge_df_normalized_gaussian[column] = (merge_df_normalized_gaussian[column] - merge_df_normalized_gaussian[column].mean()) / merge_df_normalized_gaussian[column].std()

    for column in merge_df_normalized_minmax.columns:
        if column not in excluded_columns:
            merge_df_normalized_minmax[column] = (merge_df_normalized_minmax[column] - merge_df_normalized_minmax[column].min() + epsilon) / (merge_df_normalized_minmax[column].max() - merge_df_normalized_minmax[column].min() + epsilon)
    return merge_df_normalized_minmax if min_max else merge_df_normalized_gaussian

class LSTMRainfallModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(LSTMRainfallModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer for regression

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take last timestep output
        return out.squeeze()
    
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1, num_layers=3):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define multi-layer RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to map hidden state to output (output_size=3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
    
        # Forward propagate through RNN
        out, _ = self.rnn(x)

        # Apply the fully connected layer only to the last time step
        out = self.fc(out[:, -1, :])

        return out.squeeze()
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class RainfallTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1, seq_length=12):
        super(RainfallTransformer, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_linear(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.output_linear(x[:, -1, :])
        return x.squeeze()


# Function to create sequences
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

# Define a function to load the model, this will use Streamlit's caching mechanism
@st.cache_resource
def load_model_wrapper(model_filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_filename.endswith('.pth'):
        model = torch.load(model_filename, map_location= device)
    return model

if data is not None:
    # Display the uploaded image
    if data.name.endswith('.xlsx'):
      df=pd.read_excel(data)
    else:
      df=pd.read_csv(data)

    df_normalized = normalised(df, min_max=True)
    df_normalized['month'] = pd.to_datetime(df_normalized['month'], format='%Y-%m').dt.to_period('M')
    df_normalized.set_index('month', inplace=True)

    X = df_normalized.values
    y = df_normalized['total_rainfall'].values  

    SEQ_LENGTH = 12

    X_seq, y_seq = create_sequences(X, y, 12)

    input_dim = X.shape[1]                                                                                                                   

    # Load the selected model (only once per session)         
    model_filename = model_names[model_option]                                          
    model_selected = load_model_wrapper(model_filename)

    if model_filename == 'lstm_model.pth':
        model = LSTMRainfallModel(input_dim)
    elif model_filename == 'rnn_model.pth':
        model = RNNModel(input_dim)
    elif model_filename == 'transformer_encoder_model.pth':
        model = RainfallTransformer(input_dim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(model_selected)
    model.to(device)
    model.eval()

    with torch.no_grad():
      # # Convert X_seq to tensor and move to device
      # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      # X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

      # # Get predictions
      # predictions = model(X_tensor).cpu().numpy()

      # months = df_normalized.index[-SEQ_LENGTH:]

      # predicted_rainfall = predictions[-SEQ_LENGTH:]  

      # st.subheader("Predicted Rainfall")
      # for i, value in enumerate(predicted_rainfall):
      #     st.write(f"{months[i]}: {value:.2f} mm")

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      # Use the last 12 months as input
      last_seq = df_normalized[-SEQ_LENGTH:].values  # shape: (12, num_features)
      X_input = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

      # Predict the next month
      prediction = model(X_input).cpu().numpy()

      # Optionally inverse-transform if you used MinMaxScaler for rainfall
      # prediction = scaler_y.inverse_transform([[prediction]])[0][0]

      st.subheader("Forecasted Rainfall")
      st.write(f"March 2025: {prediction:.2f} mm")
        



    
    

