import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Custom CSS for styling
st.markdown("""
    <style>
        .reportview-container {
            background-color: #2e2e2e; /* Dark background for the app */
        }
        .sidebar .sidebar-content {
            background-color: #4a4a4a; /* Darker background for the sidebar */
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #f0f0f0; /* Lighter color for the title */
        }
        .subheader {
            font-size: 1.5em;
            color: #d0d0d0; /* Lighter color for the subheader */
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .btn-container {
            display: flex;
            gap: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Stock Price Predictor Web Application</h1>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("User Inputs")
stock = st.sidebar.text_input("Enter the Stock ID", "GOOG")
start_date = st.sidebar.date_input("Start Date", datetime(datetime.now().year-20, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

if start_date > end_date:
    st.sidebar.error("Error: End date must be after start date.")
else:
    # Fetch stock data
    google_data = yf.download(stock, start_date, end_date)
    if google_data.empty:
        st.error("No data found for the selected stock and date range.")
    else:
        st.subheader("Stock Data")

        # Define the function for plotting graphs
        def plot_interactive_graph(title, x_data, y_data, line_name):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data.index, y=x_data.values, mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=y_data.index, y=y_data.values, mode='lines', name=line_name))
            fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
            return fig

        # Calculate splitting length
        splitting_len = int(len(google_data) * 0.7)
        
        # Prepare data for model prediction
        x_test = pd.DataFrame(google_data.Close[splitting_len:])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[['Close']])

        x_data = []
        y_data = []

        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Load the model
        model = load_model("Latest_stock_price_model.keras")

        # Predict using the model
        predictions = model.predict(x_data)
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        plotting_data = pd.DataFrame(
            {
                'original_test_data': inv_y_test.reshape(-1),
                'predictions': inv_pre.reshape(-1)
            },
            index=google_data.index[splitting_len+100:]
        )

        # Centered Layout for Buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button('Tabular Stock Data', key='tabular_data'):
                st.session_state['current_output'] = 'tabular_data'
        
        with col2:
            ma_option = st.radio(
                'Select Moving Average Duration',
                ['None', '250 days', '200 days', '100 days', '100 days & 250 days'],
                index=0,
                key='ma_option'
            )
            if st.button('Original Close Price vs Moving Average', key='ma_graph'):
                st.session_state['current_output'] = f'ma_graph_{ma_option}'
        
        with col3:
            if st.button('Original Values vs Predicted Values (Tabular Data)', key='tabular_vs_predicted'):
                st.session_state['current_output'] = 'tabular_vs_predicted'

            if st.button('Original Values vs Predicted Close Price (Graph)', key='graph_vs_predicted'):
                st.session_state['current_output'] = 'graph_vs_predicted'

        # Container for Outputs
        with st.container():
            if 'current_output' in st.session_state:
                output = st.session_state['current_output']

                if output == 'tabular_data':
                    st.write(google_data)

                elif 'ma_graph' in output:
                    if output == 'ma_graph_250 days':
                        google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
                        fig_ma = plot_interactive_graph('Original Close Price and MA for 250 days', google_data.Close, google_data['MA_for_250_days'], 'MA 250 Days')
                        st.plotly_chart(fig_ma)
                    elif output == 'ma_graph_200 days':
                        google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
                        fig_ma = plot_interactive_graph('Original Close Price and MA for 200 days', google_data.Close, google_data['MA_for_200_days'], 'MA 200 Days')
                        st.plotly_chart(fig_ma)
                    elif output == 'ma_graph_100 days':
                        google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
                        fig_ma = plot_interactive_graph('Original Close Price and MA for 100 days', google_data.Close, google_data['MA_for_100_days'], 'MA 100 Days')
                        st.plotly_chart(fig_ma)
                    elif output == 'ma_graph_100 days & 250 days':
                        google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
                        google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
                        fig_ma = plot_interactive_graph('Original Close Price and MA for 100 days and MA for 250 days', google_data.Close, google_data[['MA_for_100_days', 'MA_for_250_days']].mean(axis=1), 'MA Combined')
                        st.plotly_chart(fig_ma)

                elif output == 'tabular_vs_predicted':
                    st.write(plotting_data)

                elif output == 'graph_vs_predicted':
                    combined_data = pd.concat([google_data.Close[splitting_len+100:], plotting_data], axis=1)
                    combined_data.columns = ['Close Price', 'Original Test Data', 'Predicted Test Data']
                    combined_data = combined_data.dropna()

                    if combined_data.empty:
                        st.error("No data available to plot.")
                    else:
                        fig_final = go.Figure()
                        fig_final.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Close Price'], mode='lines', name='Close Price'))
                        fig_final.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Original Test Data'], mode='lines', name='Original Test Data'))
                        fig_final.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Predicted Test Data'], mode='lines', name='Predicted Test Data'))
                        fig_final.update_layout(
                            title='Original Close Price vs Predicted Close Price',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            title_font_size=24
                        )
                        st.plotly_chart(fig_final)
