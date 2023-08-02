import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data from the CSV file
data = pd.read_csv('stocks5.csv')
# ... (rest of your data preprocessing)

# Create the Streamlit app structure
def main():
    st.title("Stock Price Prediction with ARIMA")
    st.write("Upload a CSV file with data for prediction. Need a Close column in your data!")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Split data into train and test
        train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]

        # Model training and prediction
        training_data = train_data['Close'].values
        test_data = test_data['Close'].values
        history = [x for x in training_data]
        model_predictions = []

        N_test_observations = len(test_data)
        for time_point in range(N_test_observations):
            model = ARIMA(history, order=(4, 1, 0))  # Use fixed ARIMA order (modify as needed)
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            true_test_value = test_data[time_point]
            history.append(true_test_value)

        # Calculate MSE and display result
        MSE_error = mean_squared_error(test_data, model_predictions)
        st.write("Testing Mean Squared Error is", MSE_error)

        # Convert the test_data and model_predictions to pandas Series
        dates = df[int(len(df)*0.7):].index
        test_series = pd.Series(test_data, index=dates)
        predictions_series = pd.Series(model_predictions, index=dates)

        # Plotting the actual and predicted prices over dates
        plt.plot(df.index, df['Close'], label='Actual')
        plt.plot(predictions_series.index, predictions_series, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Actual vs. Predicted Prices over Dates')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

# Run the Streamlit app
if __name__ == "__main__":
    main()
