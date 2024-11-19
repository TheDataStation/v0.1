from dsapplicationregistration.dsar_core import api_endpoint, function
from escrowapi.escrow_api import EscrowAPI

import time
import duckdb

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
    query_start_time = time.perf_counter()
    res_df = conn.execute(updated_query).fetchdf()
    query_end_time = time.perf_counter()
    conn.close()
    print("total query time: ", query_end_time - query_start_time)
    return res_df

