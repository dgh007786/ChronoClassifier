import requests
import pandas as pd
import time
import datetime
from datetime import timedelta
import threading
import logging
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, DateTime
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
from pycaret.regression import *
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = 'beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq' 
CURRENCY_PAIRS = ['USD/CHF', 'EUR/USD', 'USD/CAD']

hour_count = 0

engine = create_engine('sqlite:///fx_data_multiple.db', connect_args={'timeout': 40})
metadata = MetaData()
fx_rates_sql = Table('fx_rates_new_hw2', metadata,
                    Column('id', Integer, primary_key=True),
                    Column('currency_pair', String),
                    Column('data_timestamp', String),
                    Column('db_timestamp', DateTime),
                    Column('max_value', Float),
                    Column('min_value', Float),
                    Column('mean_value', Float),
                    Column('vol', Float),
                    Column('fd', Float))
metadata.create_all(engine)
Session = sessionmaker(bind=engine)

final_engine = create_engine('sqlite:///final_fx_data_multiple.db')
final_metadata = MetaData()
final_fx_rates_sql = Table('final_fx_rates_hw2', final_metadata,
                           Column('id', Integer, primary_key=True),
                           Column('currency_pair', String),
                           Column('data_timestamp', String),
                           Column('db_timestamp', DateTime),
                           Column('max_value', Float),
                           Column('min_value', Float),
                           Column('mean_value', Float),
                           Column('vol', Float),
                           Column('fd', Float),
                           Column('hour', Float))
final_metadata.create_all(final_engine)
FinalSession = sessionmaker(bind=final_engine)

client = MongoClient('mongodb://localhost:27017/')
mongo_db = client['fx_data_multiple']
fx_rates_mongo = mongo_db['fx_rates_a']

final_mongo_client = MongoClient('mongodb://localhost:27017/')
final_mongo_db = final_mongo_client['fx_data_multiple']
final_fx_rates_mongo = final_mongo_db['fx_rates_f']

def fetch_fx_data(pair):
    url = f'https://api.polygon.io/v1/conversion/{pair}'
    params = {
        'amount': 1,
        'precision': 4,
        'apiKey': API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # print("fetching ",pair," rate: ",data['converted'])
        return data
    except requests.RequestException as e:
        logging.error(f"Error fetching data for {pair}: {str(e)}")
        return None
def clear_sql_table(session):
    try:
        session.execute(fx_rates_sql.delete())  # Delete all rows in the table
        session.commit()
        # print("cleared auxiliary SQL table")
    except Exception as e:
        logging.error(f"Error clearing SQL table: {e}")

def clear_mongo_collection():
    # print("cleared auxiliary mongo table")
    fx_rates_mongo.delete_many({})  # Delete all documents in the collection

def store_in_final_sql_db(session, max_value, min_value, mean_value, vol, fd, timestamp, currency_pair):
    global hour_count
    try:
        # Insert the data vector into the final SQLite database
        ins_query = final_fx_rates_sql.insert().values(
            currency_pair=currency_pair,
            data_timestamp=timestamp,
            db_timestamp=datetime.datetime.now(),
            max_value=max_value,
            min_value=min_value,
            mean_value=mean_value,
            vol=vol,
            fd=fd,
            hour=hour_count
        )
        session.execute(ins_query)
        session.commit()  # Commit the transaction
        # print("Stored data in final sql db")
    except Exception as e:
        logging.error(f"Error storing in final SQLite DB: {e}")

def store_in_final_mongo_db(max_value, min_value, mean_value, vol, fd, timestamp, currency_pair):
    global hour_count
    try:
        data_to_insert = {
            'currency_pair': currency_pair,
            'data_timestamp': timestamp,
            'db_timestamp':datetime.datetime.now(),
            'max_value': max_value,
            'min_value': min_value,
            'mean_value':mean_value,
            'vol':vol,
            'fd':fd,
            'hour': hour_count 
        }
        final_fx_rates_mongo.insert_one(data_to_insert)
        # print("Stored data in final mongo db")
    except Exception as e:
        logging.error(f"Write failed: {e}, retrying...")

def store_in_auxiliary_db_sql(session, max_value, min_value, mean_value, vol, fd, timestamp, currency_pair):
    try:
        ins_query = fx_rates_sql.insert().values(
            currency_pair=currency_pair,
            data_timestamp=timestamp,
            db_timestamp=datetime.datetime.now(),
            max_value=max_value,
            min_value=min_value,
            mean_value=mean_value,
            vol=vol,
            fd=fd
        )
        session.execute(ins_query)
        session.commit()
        # print("Stored data in auxiliary sql db")
    except Exception as e:
        logging.error(f"Write failed: {e}, retrying...")

def store_in_auxiliary_db_mongo(max_value, min_value, mean_value, vol, fd, timestamp, currency_pair):
    try:
        data_to_insert = {
            'currency_pair': currency_pair,
            'data_timestamp': timestamp,
            'db_timestamp':datetime.datetime.now(),
            'max_value': max_value,
            'min_value': min_value,
            'mean_value':mean_value,
            'vol':vol,
            'fd':fd
        }
        fx_rates_mongo.insert_one(data_to_insert)
        # print("Stored data in auxiliary mongo db")
    except Exception as e:
        logging.error(f"Write failed: {e}, retrying...")

def return_df_from_final_sql_db():
    try:
        df = pd.read_sql_table('final_fx_rates_hw2', final_engine)
        return df
    except Exception as e:
        logging.error(f"Error reading from final SQLite DB: {e}")

def return_df_from_final_mongo_db():
    try:
        df = pd.DataFrame(list(final_fx_rates_mongo.find()))
        return df
    except Exception as e:
        logging.error(f"Error reading from final MongoDB: {e}")

hourly_means = {pair: [] for pair in CURRENCY_PAIRS}

def correlation(data1, data2):
    print("correlation data1",data1, "data2",data2)
    return pd.Series(data1).corr(pd.Series(data2))

results_dict = {}

def classify_correlations(df_train, df_test,pair):
    global hour_count
    logging.info(f"Classifying for {pair}...")
    print("df_train",df_train)
    print("df_test",df_test)
    logging.info(f"Setting up model for {pair}...")
    clf1 = setup(data = df_train, target = 'mean_value', test_data=df_test , session_id = 123)
    best_model = compare_models()
    print("best_model",best_model)
    logging.info(f"Predicting for {pair}...")
    predictions = predict_model(best_model, data=df_test)
    logging.info(f"Evaluating for {pair}...")
    mae = mean_absolute_error(predictions['mean_value'], predictions['prediction_label'])
    if pair in results_dict:
        results_dict[pair].append({'best_model': best_model.__class__.__name__, 'MAE': mae})
    else:
        results_dict[pair] = [{'best_model': best_model.__class__.__name__, 'MAE': mae}]
    logging.info(f"Best model for {pair}: {best_model.__class__.__name__}, MAE: {mae}")


def compute_correlations_and_classify():
    global hour_count
    logging.info("Computing correlations and classifying...")
    pairs = list(hourly_means.keys())
    correlations = {}
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            corr = correlation(hourly_means[pairs[i]], hourly_means[pairs[j]])
            correlations[(pairs[i], pairs[j])] = corr
    data = pd.DataFrame({
        'currency_pair': pairs,
        'corr1': [1 if i == 0 else None for i in range(3)],
        'corr2': [1 if i == 1 else None for i in range(3)],
        'corr3': [1 if i == 2 else None for i in range(3)]
    })

    # Filling in the correlation values based on the dictionary
    for (pair1, pair2), value in correlations.items():
        # Find indices for pair1 and pair2
        idx1 = pairs.index(pair1)
        idx2 = pairs.index(pair2)
        
        # Update the DataFrame
        data.iloc[idx1, idx2 + 1] = value  # +1 because 'pair' is the first column
        data.iloc[idx2, idx1 + 1] = value  # Symmetric value

    df = return_df_from_final_sql_db()
    df = df.drop(columns=['id', 'db_timestamp','data_timestamp'])
    df = df.merge(data, on='currency_pair', how='left')
    #separate each currency pair into different dataframes
    if hour_count >= 1:
        for pair in CURRENCY_PAIRS:
            df_pair = df[df['currency_pair'] == pair]
            df_train = df_pair[df_pair['hour'] < hour_count]
            df_test = df_pair[df_pair['hour'] == hour_count]
            classify_correlations(df_train,df_test, pair)
    hour_count += 1

minOfBucket = 6
numOfHours = 5
barrier = threading.Barrier(len(CURRENCY_PAIRS), action=compute_correlations_and_classify) 

def calculate_fd(N, max_value, min_value):
    return N / (max_value - min_value) if (max_value - min_value) != 0 else 1

def process_currency_pair(currency_pair):
    session = Session()
    final_session = FinalSession()
    j = 0
    try:
        for i in range((numOfHours*3600)//(minOfBucket*60)): 
            logging.info(f"Starting loop {i} for {currency_pair}")
            start_time = datetime.datetime.now()
            end_time = start_time + timedelta(seconds=minOfBucket*60)

            max_value = 0
            min_value = float('inf')
            sum_values = 0
            total = 0
            prev_rate = 0
            prev_vol = 1
            N = 0
            hour = 0
            while datetime.datetime.now() < end_time:
                data = fetch_fx_data(currency_pair)
                if data and 'converted' in data:
                    current_rate = data['converted']
                    timestamp = datetime.datetime.now()

                    min_value = min(min_value, current_rate)
                    max_value = max(max_value, current_rate)

                    sum_values += current_rate

                    total += 1

                    mean_value = sum_values / total if total else 0
                    hourly_means[currency_pair].append(mean_value)
                    vol = (max_value - min_value) / mean_value if mean_value else float('inf')
                    if vol == 0:
                        vol = 1

                    N += (current_rate - prev_rate) / (0.025 * prev_vol) if i != 0 else 0
                    fd = calculate_fd(N, max_value, min_value)

                    prev_rate = current_rate
                    prev_vol = vol

                    store_in_auxiliary_db_sql(session, max_value, min_value, mean_value, vol, fd, timestamp, currency_pair)
                    store_in_auxiliary_db_mongo(max_value, min_value, mean_value, vol, fd, timestamp, currency_pair)
                time.sleep(1)

            store_in_final_sql_db(final_session, max_value, min_value, mean_value, vol, fd, start_time, currency_pair)
            store_in_final_mongo_db(max_value, min_value, mean_value, vol, fd, start_time, currency_pair)

            clear_sql_table(session)
            clear_mongo_collection()
            j += 1
            if j % 10 == 0:
                logging.info(f"Done calculating for {currency_pair} and waiting for others")
                barrier.wait()
                hour += 1

    except Exception as e:
        logging.error(f"Error in processing for {currency_pair}: {e}")
    finally:
        session.close()
        final_session.close()


threads = [threading.Thread(target=process_currency_pair, args=(pair,)) for pair in CURRENCY_PAIRS]
for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print("results_dict", results_dict)
logging.info("All threads have finished processing.")

results_dict
