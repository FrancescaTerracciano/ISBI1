import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse as rmse_function
from statsmodels.tools.eval_measures import aic
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, SpatialDropout3D
from keras.models import Sequential
from keras.optimizers import Adam