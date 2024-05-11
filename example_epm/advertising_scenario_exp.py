from dsapplicationregistration.dsar_core import api_endpoint, function
from escrowapi.escrow_api import EscrowAPI

import duckdb
from sklearn.linear_model import LogisticRegression

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
    from_index = query.lower().find("from")
    query_first_half = query[:from_index]
    query_second_half = query[from_index:]
    facebook_accessed = query.lower().find("facebook")
    youtube_accessed = query.lower().find("youtube")
    if facebook_accessed != -1:
        facebook_path = EscrowAPI.CSVDEStore.read(1)
        query_second_half = query_second_half.replace("facebook", f"read_csv_auto('{facebook_path}') as facebook", 1)
    if youtube_accessed != -1:
        youtube_path = EscrowAPI.CSVDEStore.read(2)
        query_second_half = query_second_half.replace("youtube", f"read_csv_auto('{youtube_path}') as youtube", 1)
    return query_first_half + query_second_half


def run_query(query):
    """
    Run a user given query.
    """
    updated_query = update_query(query)
    conn = duckdb.connect()
    res_df = conn.execute(updated_query).fetchdf()
    conn.close()
    return res_df


@api_endpoint
@function
def train_model_over_joined_data(label_name, query=None):
    # First check if the joined df has been preserved already
    # print(EscrowAPI)
    # print(EscrowAPI.test_get_comp())
    joined_de_id = EscrowAPI.load("joined_de_id")
    res_df = None
    if joined_de_id:
        print(joined_de_id)
        res_df = EscrowAPI.ObjectDEStore.read(joined_de_id)
    elif query:
        res_df = run_query(query)
        print("Need to preserve intermediate DEs!")
        joined_de_id = EscrowAPI.ObjectDEStore.write(res_df)
        EscrowAPI.store("joined_de_id", joined_de_id["de_id"])
    if res_df is not None:
        X = res_df.drop(label_name, axis=1)
        y = res_df[label_name]
        clf = LogisticRegression().fit(X, y)
        return clf