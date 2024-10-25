from dsapplicationregistration.dsar_core import api_endpoint, function
from escrowapi.escrow_api import EscrowAPI

import time
import duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

@api_endpoint
def upload_data_in_csv(de_in_bytes):
    return EscrowAPI.CSVDEStore.write(de_in_bytes)

@api_endpoint
def propose_contract(dest_agents, des, f, *args, **kwargs):
    return EscrowAPI.propose_contract(dest_agents, des, f, *args, **kwargs)

@api_endpoint
def approve_contract(contract_id):
    return EscrowAPI.approve_contract(contract_id)

@api_endpoint
def upload_data_in_csv(content):
    return EscrowAPI.CSVDEStore.write(content)

# What should the income query look like? We will do a query replacement
# select * from facebook
# select facebook.firstname from facebook

# update_query will do replace("facebook") with read_csv_auto("path_to_facebook") as facebook
# (and similarly for YouTube)
def update_query(query):
    formatted = query.format(de1_filepath = EscrowAPI.CSVDEStore.read(1), de2_filepath = EscrowAPI.CSVDEStore.read(2))
    return formatted

@api_endpoint
@function
def run_query(query):
    """
    Run a user given query.
    """
    updated_query = update_query(query)
    print(updated_query)
    conn = duckdb.connect()
    res_df = conn.execute(updated_query).fetchdf()
    conn.close()
    return res_df


@api_endpoint
@function
def train_model_over_joined_data_v1(model_name, label_name, query=None):
    # First check if the joined df has been preserved already
    joined_de_id = EscrowAPI.load("joined_de_id")
    res_df = None
    if joined_de_id:
        print(joined_de_id)
        res_df = EscrowAPI.ObjectDEStore.read(joined_de_id)
    elif query:
        print("Need to preserve intermediate DEs!")
        res_df = run_query(query)
        joined_de_id = EscrowAPI.ObjectDEStore.write(res_df)
        EscrowAPI.store("joined_de_id", joined_de_id["de_id"])
    get_combined_data_time = time.time()
    if res_df is not None:
        X = res_df.drop(label_name, axis=1)
        y = res_df[label_name]
        if model_name == "logistic_regression":
            # print("Training LOGIT model!!!")
            clf = LogisticRegression().fit(X, y)
            return clf.coef_, get_combined_data_time
        elif model_name == "MLP":
            clf = MLPClassifier()
            clf.fit(X, y)
            return clf.coefs_, get_combined_data_time


@api_endpoint
@function
def train_model_over_joined_data_v2(model_name, label_name, query):
    res_df = run_query(query)
    # print("Result Dataframe is", res_df)
    get_combined_data_time = time.time()
    if res_df is not None:
        X = res_df.drop(label_name, axis=1)
        y = res_df[label_name]
        if model_name == "logistic_regression":
            clf = LogisticRegression().fit(X, y)
            return clf.coef_, get_combined_data_time
        elif model_name == "MLP":
            clf = MLPClassifier()
            clf.fit(X, y)
            return clf.coefs_, get_combined_data_time
