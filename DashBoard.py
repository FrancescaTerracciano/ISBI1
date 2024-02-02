
## LIBRARIES IMPORT ##
from libraries import *



## PAGE CONFIGURATION ##
st.set_page_config(
    page_title='Parasites DashBoard',
    page_icon=':🦗:',
    layout='wide',
    initial_sidebar_state='expanded'  # Espandi la barra laterale inizialmente, se desiderato
    )
st.title('Environmental Dashboard: Male Parasites Under the Lens 🍃🪳')



## DATA SOURCE ## 
url = 'https://raw.githubusercontent.com/Dad-cip/Information-System-e-Business-Intelligent/main/Dataset_ISBI.csv'
url_target = 'https://raw.githubusercontent.com/Dad-cip/Information-System-e-Business-Intelligent/main/Dataset_ISBI_Target.csv'

date_column = 'time'
date_column_target = 'Date'

@st.cache_data
def load_data(url, date_col):
    """
    Load data from an url in a pandas dataframe, setting the date column indicated.

    Args:
    url (str): The url from which download the data.
    date_col (str): The name of the date column in the dataframe.

    Returns:
    pd.dataframe: Dataframe containing the loaded data.
    """
    df = pd.read_csv(url, sep=';', parse_dates=[date_col])
    df.fillna(method='ffill', inplace=True)
    df['temperature_mean'] = df['temperature_mean'].str.replace(',', '.').astype(float)
    df['selected'] = [False] * len(df)      
    return df

df = load_data(url, date_column)
df_target = load_data(url_target, date_column_target)

# Aggiustiamo la colonna relativa alla data del dataframe senza target
df[date_column] = df[date_column].dt.date     # Cancelliamo le ore dalla colonna "time"

# Aggiustiamo la colonna relativa alla data del dataframe con target
mappa_mesi = {
    'gen': 'Jan-2022',
    'feb': 'Feb-2022',
    'mar': 'Mar-2022',
    'apr': 'Apr-2022',
    'mag': 'May-2022',
    'giu': 'Jun-2022',
    'lug': 'Jul-2022',
    'ago': 'Aug-2022',
    'set': 'Sep-2022',
    'ott': 'Oct-2022',
    'nov': 'Nov-2022',
    'dic': 'Dec-2022'
}
for ita, eng in mappa_mesi.items():
    for i in range(len(df_target[date_column_target])):
        new_value = df_target.iloc[i, df_target.columns.get_loc(date_column_target)].replace(ita, eng)
        df_target.iloc[i, df_target.columns.get_loc(date_column_target)] = new_value
df_target[date_column_target] = [datetime.strptime(data_string, '%d-%b-%Y') for data_string in df_target[date_column_target]]
df_target[date_column_target] = df_target[date_column_target].dt.date



## STATE VARIABLES INIZIALIZATION ##
if 'selected_df' not in st.session_state:
    st.session_state['selected_df'] = df.copy()
if 'selected_df_target' not in st.session_state:
    st.session_state['selected_df_target'] = df_target.copy()
if 'temp_filtered_df' not in st.session_state:
    st.session_state['temp_filtered_df'] = df.copy()
if 'temp_filtered_df_target' not in st.session_state:
    st.session_state['temp_filtered_df_target'] = df_target.copy()



## SIDEBAR: TEMPORAL FILTERS ##      
st.sidebar.title('Temporal Filter')  
st.sidebar.subheader('Dataset without Target')

# Divisione in due colonne della sidebar
side_left_column, side_right_column = st.sidebar.columns(2)

# Definizione di variabili di stato per il filtro temporale
start_date = side_left_column.date_input("Start Date", df[date_column].min(), df[date_column].min(), df[date_column].max())
end_date = side_right_column.date_input("End Date", df[date_column].max(), df[date_column].min(), df[date_column].max())

# Filtraggio dei dati in base alle date selezionate
temp_filtered_df = st.session_state.selected_df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
st.session_state.temp_filtered_df = temp_filtered_df


st.sidebar.subheader('Dataset with target')

# Divisione in due colonne della sidebar
side_left_column, side_right_column = st.sidebar.columns(2)

# Definizione di variabili di stato per il filtro temporale
start_date_target = side_left_column.date_input("Start Date", df_target[date_column_target].min(), df_target[date_column_target].min(), df_target[date_column_target].max())
end_date_target = side_right_column.date_input("End Date", df_target[date_column_target].max(), df_target[date_column_target].min(), df_target[date_column_target].max())

# Filtraggio dei dati in base alle date selezionate
temp_filtered_df_target = st.session_state.selected_df_target[(df_target[date_column_target] >= start_date_target) & (df_target[date_column_target] <= end_date_target)]
st.session_state.temp_filtered_df_target = temp_filtered_df_target



## VISUALIZE DATASET PLOTS ##
left_column, right_column = st.columns(2)
left_check = left_column.checkbox("Dataset without target")
right_check = right_column.checkbox("Dataset with target")

# Visualizziamo il grafico del dataset senza target quando viene selezionata la checkbox
if left_check:
    left_column.subheader("Temperature and Humidity Trend")
    left_chart = go.Figure()
    left_chart.add_trace(go.Scatter(x=temp_filtered_df[date_column], y=temp_filtered_df['relativehumidity_mean'],
                    mode='lines',
                    name='humidity'))
    left_chart.add_trace(go.Scatter(x=temp_filtered_df[date_column], y=temp_filtered_df['temperature_mean'],
                    mode='lines',
                    name='temperature'))
    for row_index, row in st.session_state.selected_df.iterrows():
        if row['selected'] and (row_index in st.session_state.temp_filtered_df.index):
            selected_sample = df.iloc[row_index]
            left_chart.add_trace(go.Scatter(x=[selected_sample[date_column]], y=[selected_sample['temperature_mean']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
            left_chart.add_trace(go.Scatter(x=[selected_sample[date_column]], y=[selected_sample['relativehumidity_mean']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
    left_column.plotly_chart(left_chart, use_container_width=True)

# Visualizziamo il grafico del dataset con il target quando viene selezionata la checkbox
if right_check:
    right_column.subheader("Parasites, Temperature and Humidity Trend")
    no_zero_df = temp_filtered_df_target[temp_filtered_df_target['no. of Adult males'] != 0]
    right_chart = go.Figure()
    right_chart.add_trace(go.Scatter(x=temp_filtered_df_target[date_column_target], y=temp_filtered_df_target['relativehumidity_mean'],
                    mode='lines',
                    name='humidity'))
    right_chart.add_trace(go.Scatter(x=temp_filtered_df_target[date_column_target], y=temp_filtered_df_target['temperature_mean'],
                    mode='lines',
                    name='temperature'))
    right_chart.add_trace(go.Scatter(x=no_zero_df[date_column_target], y=no_zero_df['no. of Adult males'],
                    mode='markers',
                    name='no. parasites',
                    marker=dict(
                        size=no_zero_df['no. of Adult males'],  
                        sizemode='area',  
                        sizeref=0.1,  
                    ),
                    ))
    for row_index, row in st.session_state.selected_df_target.iterrows():
        if row['selected'] and (row_index in st.session_state.temp_filtered_df_target.index):
            selected_sample_right = df_target.iloc[row_index]
            right_chart.add_trace(go.Scatter(x=[selected_sample_right[date_column_target]], y=[selected_sample_right['temperature_mean']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
            right_chart.add_trace(go.Scatter(x=[selected_sample_right[date_column_target]], y=[selected_sample_right['relativehumidity_mean']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
            if selected_sample_right['no. of Adult males'] > 0:
                right_chart.add_trace(go.Scatter(x=[selected_sample_right[date_column_target]], y=[selected_sample_right['no. of Adult males']],
                            mode='markers',
                            showlegend=False,
                            marker=dict(color='yellow', size=7)))
    right_column.plotly_chart(right_chart, use_container_width=True)

# Definizione della callback di aggiornamento della variabile di stato
def update(df,key):
    """
    Update the state variable to preserve persistence of the changes made on the visualized table.

    Args:
    df (str): The dataframe name that has to be modified.
    key (str): The key to access the modified table in the session state.

    """
    for elem in st.session_state[key]['edited_rows']:
        st.session_state[df]['selected'][elem] = st.session_state[key]['edited_rows'][elem]['selected']

# Visualizzazione della tabella relativa al dataset senza target
left_column.subheader("Dataframe: Dataset without target")
left_column.data_editor(df, key="left_editor", 
                        column_order=('selected', 'time', 'temperature_mean', 'relativehumidity_mean'),
                        disabled=['time', 'temperature_mean', 'relativehumidity_mean'], 
                        hide_index=True, on_change=update, args=('selected_df','left_editor'))

# Visualizzazione della tabella relativa al dataset con il target
right_column.subheader("Dataframe: Dataset with target")
right_column.data_editor(df_target, key="right_editor", 
                         column_order=('selected', 'Date', 'no. of Adult males', 'temperature_mean', 'relativehumidity_mean'),
                         disabled=['Date', 'no. of Adult males', 'temperature_mean', 'relativehumidity_mean'], 
                         hide_index=True, on_change=update, args=('selected_df_target','right_editor'))



## DATA CORRELATION ## 
left_column.markdown("<h1>Correlazione <span style='font-size: 28px;'>without target</span></h1>", unsafe_allow_html=True)

df_reduced = df.drop(columns=[date_column,'selected'])

left_left_column, left_right_column = left_column.columns(2)
selected_x = left_left_column.selectbox("Seleziona colonna per l'asse x", df_reduced.columns)
selected_y = left_right_column.selectbox("Seleziona colonna per l'asse y", df_reduced.columns)

corr = df[selected_x].corr(df[selected_y])
corr_plot = plt.figure(figsize=(8, 6))
sns.regplot(x=selected_x, y=selected_y, data=df, scatter_kws={'s': 100})
plt.xlabel(selected_x)
plt.ylabel(selected_y)
plt.text(df[selected_x].min(), df[selected_y].max(), f'Correlation: {corr:.2f}', ha='left', va='bottom')
plt.grid(True)
left_column.pyplot(corr_plot)


right_column.markdown("<h1>Correlazione <span style='font-size: 28px;'>with target</span></h1>", unsafe_allow_html=True)

df_target_reduced = df_target.drop(columns=[date_column_target, 'selected'])

right_left_column, right_right_column = right_column.columns(2)
selected_x_target = right_left_column.selectbox("Seleziona colonna per l'asse x", df_target_reduced.columns)
selected_y_target = right_right_column.selectbox("Seleziona colonna per l'asse y", df_target_reduced.columns)

corr_target = df_target[selected_x_target].corr(df_target[selected_y_target])
corr_target_plot = plt.figure(figsize=(8, 6))
sns.regplot(x=selected_x_target, y=selected_y_target, data=df_target, scatter_kws={'s': 100})
plt.xlabel(selected_x_target)
plt.ylabel(selected_y_target)
plt.text(df_target[selected_x_target].min(), df_target[selected_y_target].max(), f'Correlation: {corr_target:.2f}', ha='left', va='bottom')
plt.grid(True)
right_column.pyplot(corr_target_plot)



## DATASET TARGET AUTOCORRELATION ##
st.markdown("<h1 style='text-align: center;'>INSPECTION DATASET TARGET</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Autocorrelation</h2>", unsafe_allow_html=True)

# Definiamo il nome della colonna target
target_column = 'no. of Adult males'

left_column, center_column, right_column = st.columns(3)
lags = center_column.slider(label="lags", min_value=1, max_value=52, step=1, value=20)

left_column, right_column = st.columns(2)

# Grafico ACF
@st.cache_data
def acf_plot(df, target_column, lags, title, zero=False):
    """
    Compute the plot of the ACF of the selected column.

    Args:
    df (pd.DataFrame): The input dataframe.
    target_column (str): The name of the target column in the dataframe.
    lags (int): The number of lags to show in the plot.
    title (str): The title to show above the graph.
    zero (boolean, optional): Flag indicating whether to include the 0-lag autocorrelation. Default is False.

    Returns:
    Figure: The created Figure.
    """
    return sgt.plot_acf(df[target_column], lags=lags, zero=zero, title=title)
fig_acf = acf_plot(df_target, target_column, lags, title="Autocorrelation of no. of Adult males")
left_column.pyplot(fig_acf)

# Grafico PACF
@st.cache_data
def pacf_plot(df, target_column, lags, title, zero=False, method='ols'):
    """
    Compute the plot of the PACF of the selected column.

    Args:
    df (pd.DataFrame): The input dataframe.
    target_column (str): The name of the target column in the dataframe.
    lags (int): The number of lags to show in the plot.
    title (str): The title to show above the graph.
    zero (boolean, optional): Flag indicating whether to include the 0-lag autocorrelation. Default is False.
    method (str, optional): Specifies which method for the calculations to use. Default is 'ols'.

    Returns:
    Figure: The created Figure.
    """
    return sgt.plot_pacf(df[target_column], lags=lags, zero=zero, method=method, title=title)
fig_pacf = pacf_plot(df_target, target_column, lags, title="PACF of no. of Adult males")
right_column.pyplot(fig_pacf)



## DATASET TARGET DIFFERENTIATION ##
df_target[date_column_target] = pd.to_datetime(df_target[date_column_target], format='%d-%b')
df_target[date_column_target] = df_target[date_column_target].apply(lambda x: x.replace(year=2022))
df_target.set_index(date_column_target, inplace=True)
df_target = df_target.drop(columns='selected')

st.markdown("<h2 style='text-align: center;'>Differentiation</h2>", unsafe_allow_html=True)

left_column, right_column = st.columns(2)
left_column.text('Select the order of differentiation:')
left_column.text('\n\n')
diff_order = right_column.number_input("", min_value=1, max_value=10, value=1, step=1, label_visibility='collapsed')
df_diff = df_target.diff(diff_order).dropna()
    
fig, ax = plt.subplots(figsize=(12, 6)) 
ax.plot(df_target.index, df_target[target_column]) 
ax.set_title('Before Differentiation') 
ax.set_xlabel(date_column_target) 
ax.set_ylabel(target_column) 
ax.legend() 
left_column.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 6)) 
ax.plot(df_diff.index, df_diff[target_column]) 
ax.set_title('After Differentiation') 
ax.set_xlabel(date_column_target) 
ax.set_ylabel(target_column) 
ax.legend() 
right_column.pyplot(fig)



## SIDEBAR: MODEL SELECTION ##
st.sidebar.header('Model List')

with st.sidebar:
    var_check = st.checkbox("VAR")
    arimax_check = st.checkbox("ARIMAX")
    rt_check = st.checkbox("Regression Tree")
    nn_check = st.checkbox("Neural Network")



## MODEL FITTING ##
# Splittiamo il dataset differenziato (90-10)
train_size = int(len(df_diff)*0.9)
df_diff_train = df_diff.iloc[:train_size]
df_diff_test = df_diff.iloc[train_size:]

# Definiamo la funzione che crea il grafico delle differenze tra i valori di test reali e predetti
@st.cache_data
def plot_differencies(_x1,y1,_x2,y2):
    """
    Generate a line plot comparing the differences between two columns from two Pandas DataFrames.

    Args:
    _x1 (pd.Series): The x-values for the actual dataset.
    y1 (pd.Series): The y-values for the actual dataset.
    _x2 (pd.Series): The x-values for the predicted dataset.
    y2 (pd.Series): The y-values for the predicted dataset.

    Returns:
    Figure: The created Figure.
    """
    fig = plt.figure(figsize=(10, 4))
    plt.plot(_x1, y1, label='Actual', color='blue')
    plt.plot(_x2, y2, label='Predicted', color='red')
    plt.xlabel('Data')
    plt.ylabel('No. of Adult males')
    plt.title('Confronto tra Valori Effettivi e Previsti')
    plt.legend()
    return fig


# Addestriamo il modello VAR
if var_check:
    st.markdown("<h2>VAR Model</h2>", unsafe_allow_html=True)

    left_column, right_column = st.columns(2)
    left_column.text("Enter the number of lags you want to fit:")
    lags_var = left_column.number_input("", min_value=0, max_value=52, value=6, step=1, label_visibility='collapsed')
    
    @st.cache_data
    def var_model_fit(df_train, lags): 
        """
        Fit a VAR model on the provided training dataset.

        Args:
        df_train (pd.DataFrame): The training dataset.
        lags (int): The number of lags to consider in the VAR model.

        Returns:
        statsmodels.tsa.vector_ar.var_model.VARResults: The training results.
        
        """
        model = VAR(df_train)
        results = model.fit(lags)
        return results
    results_var = var_model_fit(df_diff_train, lags_var)

    # Effettuiamo le predizioni sul test
    lag_order = results_var.k_ar
    forecast_var = results_var.forecast(df_diff_train.values[-lag_order:], steps=len(df_diff_test))
    target_forecast_var = forecast_var[:, 0]    # estraiamo solo la colonna relativa al target
    comparison_var = pd.DataFrame({'Predicted': target_forecast_var, 'Actual': df_diff_test['no. of Adult males']})
    
    rmse_var = np.sqrt(mean_squared_error(comparison_var['Actual'], comparison_var['Predicted']))
    mae_var = mean_absolute_error(comparison_var['Actual'], comparison_var['Predicted'])
    left_column.markdown(f'**RMSE**: {rmse_var}')
    left_column.markdown(f'**MAE**: {mae_var}')

    fig = plot_differencies(comparison_var.index, comparison_var['Actual'], comparison_var.index, comparison_var['Predicted'])
    right_column.pyplot(fig)
    
    
# Addestriamo il modello ARIMAX
if arimax_check:
    st.markdown("<h2>ARIMAX Model</h2>", unsafe_allow_html=True)
    
    left_column, right_column = st.columns(2)
    left_column.text("Enter the order:")
    
    ll, cl, rl = left_column.columns(3)
    AR_ord = ll.number_input("AR", min_value=0, max_value=10, value=1, step=1, key='lll')
    I_ord = cl.number_input("I", min_value=0, max_value=10, value=0, step=1, key='cll')
    MA_ord = rl.number_input("MA", min_value=0, max_value=10, value=1, step=1, key='rll')
    selected_order = (AR_ord,I_ord,MA_ord)
        
    # Definiamo le variabili esogene per il training
    exog_vars = df_diff_train[['relativehumidity_mean','temperature_mean']]

    @st.cache_data
    def arimax_model_fit(df_train, target_column, exog_vars, order): 
        """
        Fit an ARIMAX model on the provided training dataset.

        Args:
        df_train (pd.DataFrame): The training dataset.
        target_column (str): The name of the target column in the dataset.
        exog_vars (list): The list of columns to consider as exogenous variables.
        order (tuple): The desired order for the ARIMAX model.

        Returns:
        statsmodels.tsa.arima.model.ARIMAResults: The training results.
        
        """
        model = ARIMA(df_train[target_column], exog = exog_vars, order=order)
        results = model.fit()
        return results
    results_arimax = arimax_model_fit(df_diff_train, target_column, exog_vars, selected_order)    
    
    # Effettuiamo le predizioni sul test
    start_date = df_diff_test.index[0]
    end_date = df_diff_test.index[-1]
    forecast_arimax = results_arimax.predict(start = start_date, end = end_date, exog=df_diff_test[['relativehumidity_mean','temperature_mean']])
    comparison_arimax = pd.DataFrame({'Predicted': forecast_arimax, 'Actual': df_diff_test['no. of Adult males']})

    rmse_arimax = np.sqrt(mean_squared_error(comparison_arimax['Actual'], comparison_arimax['Predicted']))
    mae_arimax = mean_absolute_error(comparison_arimax['Actual'], comparison_arimax['Predicted'])
    left_column.markdown(f'**RMSE**: {rmse_arimax}')
    left_column.markdown(f'**MAE**: {mae_arimax}')
    
    fig = plot_differencies(comparison_arimax.index, comparison_arimax['Actual'], comparison_arimax.index, comparison_arimax['Predicted'])
    right_column.pyplot(fig)
    
    
# Determine the number of optimal lags for each column
@st.cache_data
def find_optimal_lags(dataframe, _columnnames, max_lags=20):
    """
    Function to find the optimal number of lags for each column to satisfy the ADF test.

    Args:
    dataframe (pd.DataFrame): The dataframe containing the time series data.
    columnnames (list): List of column names to evaluate.
    max_lags (int): Maximum number of lags to test for stationarity.

    Returns:
    dict: Dictionary of column names and their respective optimal number of lags.
    """

    optimal_lags = {}
    for column in _columnnames:
        for lag in range(1, max_lags + 1):
            # Apply differencing based on the current lag
            differenced_series = dataframe[column].diff(lag).dropna()

            # Perform ADF test
            p_value = adfuller(differenced_series)[1]

            # Check if the series is stationary
            if p_value <= 0.05:
                optimal_lags[column] = lag
                break
        else:
            # If none of the lags up to max_lags make the series stationary,
            # use the max_lags value
            optimal_lags[column] = max_lags

    return optimal_lags
column_names = df_target.columns
optimal_lags = find_optimal_lags(df_target, column_names)

# Create a new dataframe with the lagged version of columns
@st.cache_data
def create_combined_dataset(dataframe, optimal_lags, target_column):
    """
    Create a combined dataset with original features, their lagged versions, and the original target variable.

    Args:
    dataframe (pd.DataFrame): The original dataframe.
    optimal_lags (dict): Dictionary of column names and their respective optimal number of lags.
    target_column (str): Name of the target column.

    Returns:
    pd.DataFrame: Dataframe with original features, lagged features, and the original target variable.
    """
    combined_df = pd.DataFrame(index=dataframe.index)

    # Include original features
    for column in dataframe.columns:
        if column != target_column:  # Exclude target column from lagging
            combined_df[column] = dataframe[column]

            # Create lagged features for each column based on optimal lags
            for lag in range(1, optimal_lags.get(column, 1) + 1):
                combined_df[f'{column}_lag_{lag}'] = dataframe[column].shift(lag)

    # Add the original target variable
    if target_column != '':
        combined_df[target_column] = dataframe[target_column]

    # Remove rows with NaN values created by lagging
    combined_df.dropna(inplace=True)
    return combined_df
combined_df = create_combined_dataset(df_target, optimal_lags, target_column)

# Addestriamo il modello Regression Tree      
if rt_check:
    st.markdown("<h2>Regression Tree Model</h2>", unsafe_allow_html=True)
    left_column, right_column = st.columns(2)
    left_column.text("Enter max_depth, min_samples_split, min_samples_leaf:")
    
    ll, cl, rl = left_column.columns(3)
    max_depth = ll.number_input("max depth", min_value=1, max_value=10, value=7, step=1, key='l')
    min_samples_split = cl.number_input("split", min_value=2, max_value=10, value=5, step=1, key='c')
    min_samples_leaf = rl.number_input("leaf", min_value=1, max_value=10, value=2, step=1, key='r')
    
    # Splittiamo il dataset in train e test (90-10)
    train_size = int(len(combined_df)*0.9)
    df_train = combined_df.iloc[:train_size]
    df_test = combined_df.iloc[train_size:]
    X_train = df_train.drop(target_column, axis=1)
    y_train = df_train[target_column]
    X_test_tree = df_test.drop(target_column, axis=1)
    y_test = df_test[target_column]
    
    @st.cache_data
    def rt_model_fit(X_train, y_train, max_depth, min_samples_leaf, min_samples_split): 
        """
        Fit a Regression Tree model on the provided training dataset.

        Args:
        X_train (pd.DataFrame): The input features of the training dataset.
        y_train (pd.Series): The target values of the training dataset.
        max_depth (int): The maximum depth of the tree.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        min_samples_split (int): The minimum number of samples required to split an internal node.

        Returns:
        sklearn.tree.DecisionTreeRegressor: The trained model.
        """
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        return model
    model_rt = rt_model_fit(X_train, y_train, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    
    # Effettuiamo le predizioni sul test
    y_pred_rt = model_rt.predict(X_test_tree)

    # Valutiamo le prestazioni del modello
    mse_rt = mean_squared_error(y_test, y_pred_rt)
    left_column.markdown(f'**Mean Squared Error**: {mse_rt}')
    
    fig = plot_differencies(df_test.index, y_test, df_test.index, y_pred_rt)
    right_column.pyplot(fig)
    
    
# Effettuiamo il pre-processing dei dati per l'addestramento della rete neurale    
@st.cache_data    
def preprocess_data(dataframe, target_column, lr):
    """
    Preprocess the data for MLP training. Splits data, normalizes features, and creates an MLP model.

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    target_column (str): The name of the target column in the dataframe.

    Returns:
    tuple: (X_train, X_test, y_train, y_test, model) where `model` is an instance of a Keras Sequential model.
    """
    # Splitting the dataset into features and target
    X = dataframe.drop(target_column, axis=1)
    y = dataframe[target_column]

    # Manual split
    train_size = int(len(dataframe)*0.8)
    val_size = int(len(dataframe)*0.1)
    X_train = X.iloc[:train_size+1]
    y_train = y.iloc[:train_size+1]
    X_val = X.iloc[train_size+1:train_size+1+val_size]
    y_val = y.iloc[train_size+1:train_size+1+val_size]
    X_test_nn = X.iloc[train_size+1+val_size:]
    y_test = y.iloc[train_size+1+val_size:]

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_nn)

    # Define MLP model
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1, activation='relu'))  # Output layer

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, model

if nn_check:
    st.markdown("<h2>Neural Network Model</h2>", unsafe_allow_html=True)
    
    left_column, right_column = st.columns(2)
    left_column.text("Enter number of epochs, batch size, learning rate:")
    
    ll, cl, rl = left_column.columns(3)
    epochs = ll.number_input("epochs", min_value=1, max_value=100, value=10, step=1, key='ll')
    batch_size = cl.number_input("batch size", min_value=1, max_value=85, value=1, step=1, key='cl')
    learning_rate = rl.number_input("learning rate", min_value=1e-6, max_value=1.0, value=1e-3, step=1e-6, format="%f", key='rl')
    
    x_train, x_val, x_test_nn, y_train, y_val, y_test, model_nn = preprocess_data(combined_df, target_column, learning_rate)
    
    # Fit the model
    @st.cache_data
    def nn_model_fit(_model, x_train, y_train, epochs, batch_size, validation_data, verbose=0):
        """
        Fit a neural network model on the provided training dataset.

        Args:
        _model (Model): The neural network model to be trained.
        x_train (pd.DataFrame): The input features of the training dataset.
        y_train (pd.Series): The target values of the training dataset.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size used for training.
        validation_data (tuple): Tuple (x_val, y_val) for validation data.
        verbose (int, optional): Verbosity mode. Default is 0.

        Returns:
        tensorflow.python.keras.callbacks.History: The training history of the model.
        """
        history = _model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=verbose)
        return history
    history_nn = nn_model_fit(model_nn, x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=0)
    
    # Effettuiamo le predizioni sul test
    y_pred_nn = model_nn.predict(x_test_nn)

    # Valuta le prestazioni del modello
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    left_column.markdown(f'**Mean Squared Error**: {mse_nn}')
    
    fig = plot_differencies(y_test.index, y_test, y_test.index, y_pred_nn)
    right_column.pyplot(fig)
    
    fig = plt.figure(figsize=(10, 4))
    plt.plot(history_nn.history['loss'], label='Train Loss')
    plt.plot(history_nn.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    left_column.pyplot(fig)

    
## FORECASTING ##
st.title('Forecasting')
df_forecasting = df.copy()
df_forecasting[date_column] = pd.to_datetime(df_forecasting[date_column])
df_forecasting = df_forecasting.set_index(date_column)

# Differeneziamo il dataset per renderlo stazionario
df_diff_forecasting = df_forecasting.diff(diff_order).dropna()

# Plottiamo l'andamento della serie reale e delle predizioni
fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(df_target.index, df_target[target_column], label='Actual', marker='o')

end_prediction = st.date_input("End Prediction Date", df_forecasting.index.max(), df_diff_test.index.max()+timedelta(1), df_forecasting.index.max())
end_prediction = pd.to_datetime(end_prediction)
df_diff_forecasting = df_diff_forecasting[(df_diff_forecasting.index > df_diff_test.index.max()) & (df_diff_forecasting.index <= end_prediction)]

df_diff_forecasting = pd.concat([df_diff_test, df_diff_forecasting], axis=0)

# Plottiamo l'andamento della serie reale e delle predizioni
fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(df_target.index, df_target[target_column], label='Actual', marker='o')


if var_check:
    # Effettuiamo le predizioni sul test
    lag_order = results_var.k_ar
    forecast_var = results_var.forecast(df_diff_train.values[-lag_order:], steps=len(df_diff_forecasting))
    target_forecast_var = forecast_var[:, 0]    # estraiamo solo la colonna relativa al target
    target_forecast_var = target_forecast_var.cumsum()
    target_forecast_var[target_forecast_var<0] = 0 
    target_forecast_var = target_forecast_var[len(df_diff_test):]
    df_diff_forecasting_plot = df_diff_forecasting.copy()
    df_diff_forecasting_plot = df_diff_forecasting_plot[df_diff_forecasting_plot.index > df_diff_test.index.max()] 
    ax.plot(df_diff_forecasting_plot.index, target_forecast_var, label='VAR', linestyle='dashed', marker='o')


if arimax_check:
    # Effettuiamo le predizioni sul test
    start_date = df_diff_forecasting.index[0]
    end_date = end_prediction
    forecast_arimax = results_arimax.predict(start = start_date, end = end_date, exog=df_diff_forecasting[['relativehumidity_mean','temperature_mean']])
    forecast_arimax = forecast_arimax.cumsum()
    forecast_arimax[forecast_arimax<0] = 0 
    forecast_arimax = forecast_arimax[len(df_diff_test):]    
    df_diff_forecasting_plot = df_diff_forecasting.copy()
    df_diff_forecasting_plot = df_diff_forecasting_plot[df_diff_forecasting_plot.index > df_diff_test.index.max()]    
    ax.plot(df_diff_forecasting_plot.index, forecast_arimax, label='ARIMAX', linestyle='dashed', marker='o')
    

if rt_check:    
    df_combinate_forecasting = df_forecasting[(df_forecasting.index >= X_test_tree.index.max()) & (df_forecasting.index <= end_prediction)]

    df_combinate_forecasting = create_combined_dataset(df_combinate_forecasting.drop(columns='selected'), optimal_lags, '')

    y_pred_rt = model_rt.predict(df_combinate_forecasting)
    y_pred_rt[y_pred_rt < 0] = 0 

    ax.plot(df_combinate_forecasting.index, y_pred_rt, label='Regression Tree', linestyle='dashed', marker='o')


if nn_check:
    df_combinate_forecasting = df_forecasting[(df_forecasting.index >= df_target.index.max()) & (df_forecasting.index <= end_prediction)]

    df_combinate_forecasting = create_combined_dataset(df_combinate_forecasting.drop(columns='selected'), optimal_lags, '')
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_forecasting_scaled = scaler.fit_transform(df_combinate_forecasting)

    y_pred_nn = model_nn.predict(df_forecasting_scaled)

    ax.plot(df_combinate_forecasting.index, y_pred_nn, label='Neural Network', linestyle='dashed', marker='o')


ax.set_title('Actual & Predicted')
ax.set_xlabel('Date')
ax.set_ylabel(target_column)
ax.legend()
st.pyplot(fig)









