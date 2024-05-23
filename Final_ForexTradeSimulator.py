import requests
import time
import os
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from pycaret.regression import *
import pandas as pd
from pycaret.regression import setup, compare_models, pull, save_model
from pycaret.regression import load_model, predict_model
from sklearn.metrics import mean_absolute_error



################################################################       PART 1      ################################################################


# Function to fetch the current conversion rate for a currency pair using an API.
def get_conversion_rate(pair, api_key):
    url = f"https://api.polygon.io/v1/conversion/{pair[0]}/{pair[1]}?amount=1&precision=4&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['converted'], data['last']['timestamp']                 # Return the result of the conversion and the timestamp of the last update
    else:
        print(f"Error fetching data for {pair}: {response.status_code}")
        return None, None

# Setup MongoDB connections for storing currency data in two different databases.
def setup_mongodb(currency_pairs):
    client = MongoClient()
    db_aux = client['FinalExam_AuxDB']                              # Auxillary DB to store 6 minute rates for each currency pair
    db_final = client['FinalExam_StatsDB']                          # Statistics DB to store statistical metrics for each currency pair
    for pair in currency_pairs:
        collection_name_aux = f"{pair[0]}_{pair[1]}"                # Collections for each currency pair in auxillary database
        collection_name_final = f"final_{pair[0]}_{pair[1]}"        # Collections for each currency pair in statistics database
        db_aux[collection_name_aux]
        db_final[collection_name_final] 
    return client, db_aux, db_final

# Insert data into MongoDB for a specific currency pair with the fetched rate and timestamp.
def insert_data_mongodb(collections, pair, rate, fx_timestamp):
    collection_name = f"{pair[0]}_{pair[1]}"
    fx_timestamp_dt = datetime.fromtimestamp(fx_timestamp / 1000)           # Convert the fx_timestamp to a datetime object
    entry_timestamp_dt = datetime.now()                                     # Create a column for the entry timestamp to show when the data was inserted
    collections[collection_name].insert_one({"fx_rate": rate, "fx_timestamp": fx_timestamp_dt, "entry_timestamp": entry_timestamp_dt})

# Clear all data from a specific MongoDB collection after every 6 minute window.
def clear_database_data_mongodb(collections, pair):
    collection_name = f"{pair[0]}_{pair[1]}"
    collections[collection_name].delete_many({})

# Update statistical metrics for a currency pair based on newly fetched rates.
def update_statistics(stats, rate):
    stats['sum'] += rate
    stats['count'] += 1
    stats['max'] = max(stats['max'], rate)
    stats['min'] = min(stats['min'], rate)
    stats['mean'] = stats['sum'] / stats['count']
    stats['vol'] = (stats['max'] - stats['min']) / stats['mean'] if stats['mean'] else 0
    return stats

# Calculate the upper and lower bands based on the mean rate and volatility.
def calculate_keltner_channels(mean_rate, vol):
    upper_bands = [(mean_rate + n * 0.025 * vol) for n in range(1, 101)]
    lower_bands = [(mean_rate - n * 0.025 * vol) for n in range(1, 101)]
    return upper_bands, lower_bands

# Track the jumps for a currency pair using bands from previous window.
def track_price_jumps_mongodb(db, pair, bands):
    collection_name = f"{pair[0]}_{pair[1]}"
    collection = db[collection_name]
    rates_cursor = collection.find({}, {'fx_rate': 1, '_id': 0})
    # Get the rates from table corresponding to the currency pair in auxillary database
    rates = [doc['fx_rate'] for doc in rates_cursor]
    jumps = 0
    upper_bands, lower_bands = bands['upper'], bands['lower']   
    # Increment jump counter by 1 if the rate is outside the bands
    for rate in rates:
        for ub, lb in zip(upper_bands, lower_bands):
            if rate > ub or rate < lb:
                jumps += 1
                break
    return jumps

# Update the statistics database currency tables with calculated statistical metrics for a currency pair.
def update_final_mongodb(db_final, pair, max_rate, min_rate, mean_rate, vol, fd, first_fx_timestamp):
    collection_name = f"final_{pair[0]}_{pair[1]}"
    db_final[collection_name].insert_one({
        "max_rate": max_rate, 
        "min_rate": min_rate, 
        "mean_rate": mean_rate, 
        "vol": vol, 
        "fd": fd, 
        "data_timestamp": datetime.fromtimestamp(first_fx_timestamp / 1000), 
        "entry_timestamp": datetime.now()
    })

# Calculate the correlations between the 10 records of currency pair and 10 records of each EURUSD, USDJPY.
def calculate_correlations_mongodb(db_final, pair):
    correlations = {}
    # Fetch rates for the current pair
    current_pair_rates = fetch_rates(db_final, pair)
    if len(current_pair_rates) == 10:
        # Calculate correlation with EURUSD and USDJPY only
        target_pairs = [('EUR', 'USD'), ('USD', 'JPY')]
        for target_pair in target_pairs:
            other_pair_rates = fetch_rates(db_final, target_pair)
            if len(other_pair_rates) == 10:
                # Calculate the correlation between the current pair and the target pair
                correlation = np.corrcoef(current_pair_rates, other_pair_rates)[0, 1]
                correlations[f"correlation_with_{target_pair[0]}_{target_pair[1]}"] = correlation
    return correlations

# Fetch the latest 10 rates for a currency pair from MongoDB.
def fetch_rates(db, pair):
    collection_name = f"final_{pair[0]}_{pair[1]}"
    cursor = db[collection_name].find({}, {'mean_rate': 1, '_id': 0}).sort('entry_timestamp', -1).limit(10)
    return [doc['mean_rate'] for doc in cursor]

# Calculate the Mean Absolute Error (MAE) for the currency pairs using the saved model
def calculate_mae_for_currency_pairs(db_final, model_name):
    mae_results = {}
    model = load_model(model_name)                              # Load the saved model
    for pair in [('GBP','USD'), ('USD','JPY')]:                 # Iterate through non base currency pairs
        try:
            table_name_final = f"final_{pair[0]}_{pair[1]}"
            # Fetch relevant features for the last 20 records for the currency pair 
            cursor = db_final[table_name_final].find({}, {'max_rate': 1, 'min_rate': 1, 'mean_rate': 1, 'vol': 1, 'fd': 1, 'correlation_with_USD_JPY': 1, 'correlation_with_EUR_USD': 1, '_id': 0}).sort('entry_timestamp', -1).limit(20)
            df = pd.DataFrame(list(cursor))

            if not df.empty:
                features = df.drop('mean_rate', axis=1)                             # Drop the target variable                         
                target = df['mean_rate']                                            # Set the target variable
                predictions = predict_model(model, data=features)                   # Make predictions using the best model
                df['predicted_mean_rate'] = predictions['prediction_label']         # Add the predicted values to the DataFrame                 
                mae = mean_absolute_error(target, df['predicted_mean_rate'])        # Calculate MAE between the actual and predicted values
                print("MAE for", pair, ":", mae)
                mae_results[pair] = mae
            else:
                mae_results[pair] = None                                            # No data available for calculation   
        except Exception as e:
            print(e)
    return mae_results

# PyCaret to run regression analysis and identify the best model.
def run_pycaret_experiment(db_final):
    print("Running PyCaret experiment...")
    try:
        # Base currency pairs
        currency_pairs = [
            ('EUR', 'USD'), ('EUR', 'CHF'), ('EUR', 'CAD'),
            ('GBP', 'EUR'), ('GBP', 'CHF'), ('GBP', 'CAD'),
            ('USD', 'CHF'), ('USD', 'CAD')
        ]
        # Initialize an empty DataFrame for synthetic records
        synthetic_data = pd.DataFrame()
        # Iterate over the last 20 timestamps
        for i in range(20):
            records = []
            # Collect records for each currency pair for the i-th last record
            for pair in currency_pairs:
                table_name_final = f"final_{pair[0]}_{pair[1]}"
                cursor = db_final[table_name_final].find({}, {
                    'max_rate': 1, 'min_rate': 1, 'mean_rate': 1,
                    'vol': 1, 'fd': 1, 'correlation_with_USD_JPY': 1,
                    'correlation_with_EUR_USD': 1, '_id': 0
                }).sort('entry_timestamp', -1).skip(i).limit(1)
                df = pd.DataFrame(list(cursor))
                if not df.empty:
                    records.append(df)
            # Create a synthetic record by averaging across all currency pairs
            if records:
                synthetic_record = pd.concat(records).mean(axis=0).to_frame().T
                synthetic_data = pd.concat([synthetic_data, synthetic_record], ignore_index=True)

        setup(synthetic_data, target='mean_rate')                   # Set PyCaret regression
        best_model = compare_models(sort='MAE', n_select=1)         # Compare models and select the best one based on MAE
        model_name = "best_model"
        save_model(best_model, model_name)                          # Save the best model

    except Exception as e:
        print(e)

# Transform the MAE results to a dictionary with currency pairs as keys.
def transform_mae_results(mae_results):
    transformed_results = {}
    for pair, value in mae_results.items():
        currency_pair = pair[0]+pair[1]
        transformed_results[currency_pair] = value
    return transformed_results



# Initialize
api_key = '<ADD API KEY HERE>'
currency_pairs = [('EUR', 'USD'), ('USD', 'JPY'), ('GBP','CHF'), ('USD', 'CAD'), ('EUR', 'CHF'), ('EUR', 'CAD'), ('GBP','EUR'), ('GBP','USD'), ('GBP','CAD'), ('USD','CHF')]

# Setup databases
mongodb_client, mongodb_aux, mongodb_final = setup_mongodb(currency_pairs)


# Initialize statistics
# {('EUR', 'USD'): {'sum': 0, 'count': 0, 'max': float('-inf'), 'min': float('inf'), 'mean':0, 'vol':0, 'first_timestamp': None}, ...}
statistics = {pair: {'sum': 0, 'count': 0, 'max': float('-inf'), 'min': float('inf'), 'mean':0, 'vol':0, 'first_timestamp': None} for pair in currency_pairs}

# {('EUR', 'USD'): {'upper': [], 'lower': []}, ...}
bands = {pair: {'upper': [], 'lower': []} for pair in currency_pairs}        # Nested dictionary to store upper and lower bands for each currency pair

iteration = 0                                                                # Variable to keep track of the current 6 minute window
i = 4                                                                        # Variable to keep track of the hour
correlations_enabled = False                                                 # Variable to enable correlations after 2 hours                    




# Main loop simulating 5 hour window
for cycles in range(1, 51):
    # Simulate a 6-minute window for that cycle
    for m in range(360):
        # Iterate through each currency pair
        for pair in currency_pairs:
            rate, fx_timestamp = get_conversion_rate(pair, api_key)                     # Fetch the conversion rate
            insert_data_mongodb(mongodb_aux, pair, rate, fx_timestamp)                  # Insert the data into auxiliary database tables
            statistics[pair] = update_statistics(statistics[pair], rate)                # Update the statistics for the currency pair
            if statistics[pair]['first_timestamp'] is None and fx_timestamp is not None:
                statistics[pair]['first_timestamp'] = fx_timestamp                      # Set the first_timestamp for currency pair using the first fetched timestamp of the 6 minute window
    print("Cycle ", cycles, ": 6 minute simulation completed.")

    iteration += 1
    # IF block executed only after the first 6 minute window
    if iteration == 1:
        for pair in currency_pairs:
            # Clear the auxiliary database table corresponding to the currency pair
            clear_database_data_mongodb(mongodb_aux, pair)
            # Calculated upper and lower bands that will be used in the next window
            bands[pair]['upper'], bands[pair]['lower'] = calculate_keltner_channels(statistics[pair]['mean'], statistics[pair]['vol'])
        # Reset statistics for each currency pair
        statistics = {pair: {'sum': 0, 'count': 0, 'max': float('-inf'), 'min': float('inf'), 'mean':0, 'vol':0, 'first_timestamp': None} for pair in currency_pairs}
    
    # ELSE block executed after every second 6 minute window 
    else:
        for pair in currency_pairs:
            # Call the 'track_price_jumps_mongodb' function to track the price jumps
            jumps = track_price_jumps_mongodb(mongodb_aux, pair, bands[pair])
            # Calculated fractal dimensions using jumps, min and max values
            fd = jumps / (statistics[pair]['max'] - statistics[pair]['min']) if (statistics[pair]['max'] - statistics[pair]['min']) != 0 else 0
            # Update the statistics database with the calculated metrics
            update_final_mongodb(mongodb_final, pair, statistics[pair]['max'], statistics[pair]['min'], statistics[pair]['mean'], statistics[pair]['vol'], fd, statistics[pair]['first_timestamp'])
            # Clear the auxiliary database table for the next window of 6 minutes
            clear_database_data_mongodb(mongodb_aux, pair)
            # Calculated upper and lower bands that will be used in the next window
            bands[pair]['upper'], bands[pair]['lower'] = calculate_keltner_channels(statistics[pair]['mean'], statistics[pair]['vol'])
            # After 2 hours, start computing correlations
            if correlations_enabled:
                correlations = calculate_correlations_mongodb(mongodb_final, pair)
                # Retrieve the ID of the last database entry for this currency pair, sorted by timestamp in descending order (i.e., the most recent entry).
                last_entry_id = mongodb_final[f"final_{pair[0]}_{pair[1]}"].find_one(sort=[('entry_timestamp', -1)])['_id']
                # Update the most recent entry for this currency pair with the newly calculated correlations.
                mongodb_final[f"final_{pair[0]}_{pair[1]}"].update_one({'_id': last_entry_id}, {'$set': correlations})
        # Reset statistics for each currency pair
        statistics = {pair: {'sum': 0, 'count': 0, 'max': float('-inf'), 'min': float('inf'), 'mean':0, 'vol':0, 'first_timestamp': None} for pair in currency_pairs}

    # Enable correlations after 2 hours
    if not correlations_enabled and cycles >= 20:
        correlations_enabled = True
        print("Correlations will be computed from now.")
    
    # After 4th and 5th hour, classify non-base currency pairs as forecastable or non-forecastable
    if cycles == 40 or cycles == 50:
        print("Starting PyCaret experiment for hour", i)
        run_pycaret_experiment(mongodb_final)
        model_name = 'best_model'
        mae_results = calculate_mae_for_currency_pairs(mongodb_final, model_name)
        transformed_results = transform_mae_results(mae_results)                         # {('GBP', 'USD'): 0.123, ...} => {'GBPUSD': 0.123, ...}
        sorted_results = sorted(transformed_results.items(), key=lambda x: x[1])         # Sort the 'transformed_results' dictionary by values in ascending order.
        sorted_dict = {pair: value for pair, value in sorted_results}                    # Convert the sorted results back to a dictionary
        print(sorted_dict)    

        results_df = pd.DataFrame(columns=['Hour', 'Pair', 'Classification'])

        # Append results to DataFrame
        for pair, mae in sorted_dict.items():
            # Use a MAE threshold to classify currency pairs
            if mae > 0.5:
                classification = 'NON FORECASTABLE'
            else:
                classification = 'FORECASTABLE'
            new_row = pd.DataFrame({
                'Hour': [i],
                'Pair': [pair],
                'MAE': [sorted_dict[pair]],
                'Classification': [classification]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        file_exists = os.path.isfile('classification_results.csv')                                          # Check if the file exists to determine if headers should be written
        results_df.to_csv('classification_results.csv', mode='a', header=not file_exists, index=False)      # Save the DataFrame to CSV, append if file exists, write header only if file does not exist
        i+=1
    
    # Sleep for 2 minutes after every 1.5 hours
    if cycles == 15 or cycles == 30 or cycles == 45:
        time.sleep(120)


# Close the MongoDB client
mongodb_client.close()




###################################################################################################################################################









################################################################       PART 2      ################################################################


from sklearn.linear_model import LinearRegression
import numpy as np

api_key = '<ADD API KEY HERE>'

def extract_data_and_compute_slope(db, collection_name):
    # Retrieve the last 20 records from MongoDB collection, sorted by 'data_timestamp'.
    cursor = db[collection_name].find({}, {'data_timestamp': 1, 'mean_rate': 1, '_id': 0}).sort('data_timestamp', -1).limit(20)
    data = list(cursor)
    # Convert timestamps from the data to numeric format suitable for linear regression.
    timestamps = np.array([d['data_timestamp'].timestamp() for d in data]).reshape(-1, 1)
    rates = np.array([d['mean_rate'] for d in data])
    model = LinearRegression()
    model.fit(timestamps, rates)        # Fit a linear regression model to the data to find the trend (slope).
    slope = model.coef_[0]              # Extract the slope of the fitted line
    return slope

# Set up MongoDB connection and retrieve the slope of the exchange rate trends for two currency pairs.
mongodb_client = MongoClient()
db_final = mongodb_client['FinalExam_StatsDB']
slope_gbpusd = extract_data_and_compute_slope(db_final, 'final_GBP_USD')
slope_usdjpy = extract_data_and_compute_slope(db_final, 'final_USD_JPY')

print("\n")
# Determine which currency pair to long and which to short based on the slopes.
if slope_gbpusd > slope_usdjpy:
    long, short = ('GBP', 'USD'), ('USD', 'JPY')
    print("Long pair:", long, "with slope = ", slope_gbpusd)
    print("Short pair:", short, "with slope = ", slope_usdjpy)
else:
    long, short = ('USD', 'JPY'), ('GBP', 'USD')
    print("Long pair:", long, "with slope = ", slope_usdjpy)
    print("Short pair:", short, "with slope = ", slope_gbpusd)

hour = 5

# Fetch the conversion rates for the two currency pairs at 5th hour.
rate_gbpusd, fx_timestamp = get_conversion_rate(('GBP', 'USD'), api_key)
rate_usdjpy, fx_timestamp = get_conversion_rate(('USD','JPY'), api_key)

# Store the rates in a dictionary for further analysis.
if long == ('GBP', 'USD'):
    rates = {long: [(hour,rate_gbpusd)], short: [(hour,rate_usdjpy)]}           # rates = {('GBP', 'USD'): [(5, 1.4798523),...], ('USD', 'JPY'): [(5, 157.128467),...]}
else:
    rates = {long: [(hour,rate_usdjpy)], short: [(hour,rate_gbpusd)]}
print("\nRates store: ", rates)

# Find how many units of the long pair to buy for every unit of the short pair sold.
units = rate_usdjpy/rate_gbpusd
print("\nBuy ", units, "units of", long, "and sell", "1 unit of", short, "at the end of ", hour, "hour.")

# Repeat same investments at the end of the 6th and 7th hours with updated rates
while True:
    print("Sleeping for 1 hour...")
    time.sleep(3600)
    hour += 1
    if hour == 8:
        break
    # Fetch the conversion rates for the two currency pairs at the end of the hour.
    rate_gbpusd, fx_timestamp = get_conversion_rate(('GBP', 'USD'), api_key)
    rate_usdjpy, fx_timestamp = get_conversion_rate(('USD','JPY'), api_key)
    print("Buy ", units, "units of", long, "and sell", "1 units of", short, "at the end of ", hour, "hour.")
    if long == ('GBP', 'USD'):
        rates[long].append((hour,rate_gbpusd))
        rates[short].append((hour,rate_usdjpy))
    else:
        rates[long].append((hour,rate_usdjpy))
        rates[short].append((hour,rate_gbpusd))

print("\nClosing L/S position at the end of 8th hour...")

# Print final rates
rate_gbpusd_current, fx_timestamp = get_conversion_rate(('GBP', 'USD'), api_key)
rate_usdjpy_current, fx_timestamp = get_conversion_rate(('USD', 'JPY'), api_key)
print(f"GBPUSD at 8th hour: {rate_gbpusd_current}")
print(f"USDJPY at 8th hour: {rate_usdjpy_current}")

# Calculate financial changes
# {('GBP', 'USD'): [(5, 1.4798523), (6, 1.4798523), (7, 1.4798523)], ('USD', 'JPY'): [(5, 157.128467), (6, 157.128467), (7, 157.128467)]}
total_long = sum(units * rate for _, rate in rates[long])           # 112 * 1.4798523 + 112 * 1.4798523 + 112 * 1.4798523
total_short = sum(rate for _, rate in rates[short])                 # 1 * 157.128467 + 1 * 157.128467 + 1 * 157.128467
change_long = (rate_gbpusd_current * units * 3) - total_long        # (1.4823 * 112 * 3) - total_long
change_short = (rate_usdjpy_current * 3) - total_short              # (157.128467 * 3) - total_short

print("Change in long pair:", change_long)
print("Change in short pair:", change_short)
print("Net change:", change_long + change_short)



###################################################################################################################################################