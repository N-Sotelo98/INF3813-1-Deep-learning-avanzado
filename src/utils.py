import pandas as pd

def calculate_kpis(
    df, datetime_col, n_hours, delta_hours, date_col='fecha_eval'
):
    """
    Generate an augmented dataset with a sliding window approach.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        datetime_col (str): Column name for the datetime values.
        n_hours (int): Number of hours to select for ranking (e.g., top 6 and bottom 6).
        delta_hours (int): Size of the window in hours (e.g., 32 hours).
        date_col (str): Name of the column for unique dates (default is 'fecha_eval').
        
    Returns:
        tuple: A tuple containing:
            - Augmented dataset (pd.DataFrame).
            - Metadata DataFrame (pd.DataFrame).
    """
    # Ensure the datetime column is in datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[date_col] = df[datetime_col].dt.date
    
    unique_dates = df[date_col].unique()
    windows = []
    windows_metadata_list = []  # List to store metadata

    for date in unique_dates:
        # Define the 32-hour window starting from 00:00 of the current day
        start_time = pd.to_datetime(date)
        end_time = start_time + pd.Timedelta(hours=delta_hours)

        # Filter data for the window
        window_data = df[
            (df[date_col] == date)
            & (df[datetime_col] >= start_time)
            & (df[datetime_col] < end_time)
        ].copy()

        # Add necessary columns
        window_data[date_col] = date
        window_data['correlative'] = range(1, len(window_data) + 1)
        window_data = window_data.assign(
            coincidence=0, charge_real=0, charge_pred=0, discharge_real=0, discharge_pred=0
        )

        # Get top and bottom hours for 'real' and 'pred'
        top_hours_real = window_data.nlargest(n_hours, 'real')['correlative'].values
        bottom_hours_real = window_data.nsmallest(n_hours, 'real')['correlative'].values
        top_hours_avg = window_data.nlargest(n_hours, 'pred')['correlative'].values
        bottom_hours_avg = window_data.nsmallest(n_hours, 'pred')['correlative'].values

        # Update columns for discharge and charge flags
        window_data.loc[window_data['correlative'].isin(top_hours_real), 'discharge_real'] = 1
        window_data.loc[window_data['correlative'].isin(top_hours_avg), 'discharge_pred'] = 1
        window_data.loc[window_data['correlative'].isin(bottom_hours_real), 'charge_real'] = 1
        window_data.loc[window_data['correlative'].isin(bottom_hours_avg), 'charge_pred'] = 1

        # Calculate coincidences
        window_data['coincidence'] = (
            (window_data['charge_real'] == window_data['charge_pred'])
            & (window_data['discharge_real'] == window_data['discharge_pred'])
        ).astype(int)

        # Calculate revenue
        window_data['revenue_real'] = (
            window_data['discharge_real'] * window_data['real']
            - window_data['charge_real'] * window_data['real']
        )
        window_data['revenue_pred'] = (
            window_data['discharge_pred'] * window_data['real']
            - window_data['charge_pred'] * window_data['real']
        )

        # Calculate KPIs
        factor_coincidence = window_data['coincidence'].mean()
        factor_value = (
            window_data['revenue_pred'].sum() / window_data['revenue_real'].sum()
            if window_data['revenue_real'].sum() != 0
            else 0
        )

        # Calculate MAE, RMSE, and Bias
        window_data['mae'] = abs(window_data['real'] - window_data['pred'])
        window_data['mse'] = ((window_data['real'] - window_data['pred']) ** 2)
        window_data['bias'] = (window_data['pred'] - window_data['real'])
        
        mae = window_data['mae'].mean()
        rmse = window_data['mse'].mean() **0.5
        bias = window_data['bias'].mean()

        # Append metadata
        windows_metadata_list.append([date, factor_coincidence, factor_value, mae, rmse, bias])
        windows.append(window_data)

    # Combine all windows into one DataFrame
    kpis_df = pd.concat(windows, ignore_index=True)

    # Create metadata DataFrame
    summary_df = pd.DataFrame(
        windows_metadata_list,
        columns=['date', 'factor_coincidence', 'factor_value', 'mae', 'rmse', 'bias']
    )

    return kpis_df, summary_df