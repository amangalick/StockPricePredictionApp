import streamlit as st
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []

    # create all possible sequences of length lookback
    for index in range(len(data_raw) - lookback + 1):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :, :]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize the hidden and cell state tensors
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def run_lstm_model(x_train, y_train, x_test, y_test):
    input_dim = x_train.shape[2]  # Number of features (Close price in this case)
    hidden_dim = 40
    num_layers = 2
    output_dim = 1

    # Create the LSTM model
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100
    lstm_hist = np.zeros(num_epochs)
    start_time = time.time()
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        lstm_hist[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time
    print("LSTM Training time: {}".format(training_time))
    lstm_result = {'training_time': training_time}

    start_time = time.time()
    model.eval()
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test)
    lstm_result['testing_time'] = time.time() - start_time
    lstm_result['mse'] = loss.item()
    return lstm_result

def run_gru_model(x_train, y_train, x_test, y_test):
    input_dim = 1
    hidden_dim = 40
    num_layers = 2
    output_dim = 1

    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100
    gru_hist = np.zeros(num_epochs)
    start_time = time.time()
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        gru_hist[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time
    print("GRU Training time: {}".format(training_time))
    gru_result = {'training_time': training_time}

    start_time = time.time()
    model.eval()
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test)
    gru_result['testing_time'] = time.time() - start_time
    gru_result['mse'] = loss.item()
    return gru_result

def main():
    st.title("Stock Price Prediction with LSTM and GRU")
    st.write("Upload a CSV file with data for prediction. Need a Close column")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Data preprocessing and scaling
        price = df[['Close']]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))

        lookback = 20
        x_train, y_train, x_test, y_test = split_data(price, lookback)

        # LSTM Model
        st.subheader("LSTM Model")
        lstm_result = run_lstm_model(x_train, y_train, x_test, y_test)
        st.write("LSTM Training Time:", lstm_result['training_time'])
        st.write("LSTM Testing Time:", lstm_result['testing_time'])
        st.write("LSTM MSE:", lstm_result['mse'])

        # GRU Model
        st.subheader("GRU Model")
        gru_result = run_gru_model(x_train, y_train, x_test, y_test)
        st.write("GRU Training Time:", gru_result['training_time'])
        st.write("GRU Testing Time:", gru_result['testing_time'])
        st.write("GRU MSE:", gru_result['mse'])

# Run the Streamlit app
if __name__ == "__main__":
    main()
