import streamlit as st
import numpy as np
np.float_ = np.float64
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
START = "2021-01-01"
TODAY= date.today().strftime("%Y-%m-%d")
st.title("Material Properties Prediction (MPP) App")
tensile=("TENSILE STRENGTH SOIL-9wt","TENSILE MODULUS ATM-control","TENSILE MODULUS ATM-15wt","MODULUS OF ELASTICITY-3wt","MODULUS OF ELASTICITY-18wt","FLEXURAL MODULUS SOIL-control","FLEXURAL MODULUS SOIL-9wt","FLEXURAL MODULUS SOIL-15wt","FLEXURAL MODULUS SOIL-18wt","FLEXURAL MODULUS ATM-12wt","FLEXURAL MODULUS ATM-15wt","FLEXURAL MODULUS ATM-18wt","IMPACT STRENGTH")
selected_tensile=st.selectbox("Select Property for prediction",tensile)
n_years=st.slider("Years of predication:", 1, 20)
period=n_years*365



if selected_tensile == "TENSILE STRENGTH SOIL-9wt":
        
    data_load_state = st.text("Load data...")
    data=pd.read_csv('tssoil.csv')
    data_load_state.text("Loading data...done!")
    
    
   #st.subheader('Raw data')
    #st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['tssoil9wt'], name='tssoil9wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','tssoil9wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "tssoil9wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
      #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)

elif selected_tensile == "TENSILE MODULUS ATM-control":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('tmatm.csv')
    data_load_state.text("Loading data...done!")
    
    
    st.subheader('Raw data')
    st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['tmatmcontrol'], name='tmatmcontrol'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','tmatmcontrol']]
    df_train=df_train.rename(columns={"timeseries":"ds", "tmatmcontrol":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
      #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "TENSILE MODULUS ATM-15wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('tmatm.csv')
    data_load_state.text("Loading data...done!")
    
    
    #st.subheader('Raw data')
    #st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['tmatm15wt'], name='tmatm15wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','tmatm15wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "tmatm15wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
      #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "IMPACT SOIL-9wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('impact soil.csv')
    data_load_state.text("Loading data...done!")
    
    
    st.subheader('Raw data')
    st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['impactsoil9wt'], name='impactsoil9wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','impactsoil9wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "impactsoil9wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
      #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "MODULUS OF ELASTICITY-3wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('TENSILE_M_SOIL.csv')
    data_load_state.text("Loading data...done!")
    
    
    st.subheader('Raw data')
    st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['tmsoil3wt'], name='tmsoil3wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','tmsoil3wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "tmsoil3wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
      #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "MODULUS OF ELASTICITY-18wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('TENSILE_M_SOIL.csv')
    data_load_state.text("Loading data...done!")
    
    
    #st.subheader('Raw data')
    #st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['tmsoil18wt'], name='tmsoil18wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','tmsoil18wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "tmsoil18wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
      #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "FLEXURAL MODULUS SOIL-control":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('FLEXURAL_M_SOIL.csv')
    data_load_state.text("Loading data...done!")
    
    
    st.subheader('Raw data')
    st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['fmsoilcontrol'], name='fmsoilcontrol'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','fmsoilcontrol']]
    df_train=df_train.rename(columns={"timeseries":"ds", "fmsoilcontrol":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "FLEXURAL MODULUS SOIL-9wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('FLEXURAL_M_SOIL.csv')
    data_load_state.text("Loading data...done!")
    
    
    #st.subheader('Raw data')
    #st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['fmsoil9wt'], name='fmsoil9wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','fmsoil9wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "fmsoil9wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "FLEXURAL MODULUS SOIL-15wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('FLEXURAL_M_SOIL.csv')
    data_load_state.text("Loading data...done!")
    
    
    #st.subheader('Raw data')
    #st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['fmsoil15wt'], name='fmsoil15wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','fmsoil15wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "fmsoil15wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "FLEXURAL MODULUS SOIL-18wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('FLEXURAL_M_SOIL.csv')
    data_load_state.text("Loading data...done!")
    
    
    #st.subheader('Raw data')
    #st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['fmsoil18wt'], name='fmsoil18wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','fmsoil18wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "fmsoil18wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)


elif selected_tensile == "FLEXURAL MODULUS ATM-12wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('fmatm.csv')
    data_load_state.text("Loading data...done!")
    
    
    st.subheader('Raw data')
    st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['fmatm12wt'], name='fmatm12wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','fmatm12wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "fmatm12wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "FLEXURAL MODULUS ATM-15wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('fmatm.csv')
    data_load_state.text("Loading data...done!")
    
    
   #st.subheader('Raw data')
    #st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['fmatm15wt'], name='fmatm15wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','fmatm15wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "fmatm15wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
elif selected_tensile == "FLEXURAL MODULUS ATM-18wt":
    data_load_state = st.text("Load data...")
    data=pd.read_csv('fmatm.csv')
    data_load_state.text("Loading data...done!")
    
    
    #st.subheader('Raw data')
    #st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['fmatm18wt'], name='fmatm18wt'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','fmatm18wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "fmatm18wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
    #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)
else:
    data_load_state = st.text("Load data...")
    data=pd.read_csv('IMPACT_SOIL.csv')
    data_load_state.text("Loading data...done!")
    
    
    st.subheader('Raw data')
    st.write(data.tail())
    
    def plot_raw_data():
    	fig=go.Figure()
    	fig.add_trace(go.Scatter(x=data['timeseries'], y=data['impactsoil9wt'], name='impact strength'))
    	fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    plot_raw_data()
    
    
    df_train=data[['timeseries','impactsoil9wt']]
    df_train=df_train.rename(columns={"timeseries":"ds", "impactsoil9wt":"y"})
    
    m=Prophet()
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=period)
    
    forecast = m.predict(future)
    
    
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write('forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
      #st.write('forecast components')
    #fig2=m.plot_components(forecast)
   # st.write(fig2)


