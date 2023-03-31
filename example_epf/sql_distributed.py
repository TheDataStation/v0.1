from dsapplicationregistration.dsar_core import api_endpoint, function
from escrowapi.escrow_api import EscrowAPI
import duckdb

@api_endpoint
def register_data(username,
                  data_name,
                  data_type,
                  access_param,
                  optimistic):
    print("This is a customized register data!")
    return EscrowAPI.register_data(username, data_name, data_type, access_param, optimistic)

@api_endpoint
def upload_data(username,
                data_id,
                data_in_bytes):
    return EscrowAPI.upload_data(username, data_id, data_in_bytes)

@api_endpoint
def suggest_share(username, agents, functions, data_elements):
    return EscrowAPI.suggest_share(username, agents, functions, data_elements)

@api_endpoint
def ack_data_in_share(username, data_id, share_id):
    return EscrowAPI.ack_data_in_share(username, data_id, share_id)

def get_data(de):
    if de.type == "file":
        return f"'{de.access_param}'"

def assemble_table(conn, table_name):
    accessible_de = EscrowAPI.get_all_accessible_des()
    first_partition_flag = True
    for de in accessible_de:
        if de.name == f"{table_name}.csv":
            table_path = get_data(de)
            if first_partition_flag:
                query = f"CREATE TABLE {table_name} AS SELECT * FROM {table_path}"
                conn.execute(query)
                first_partition_flag = False
            else:
                query = f"INSERT INTO {table_name} SELECT * FROM {table_path}"
                conn.execute(query)

@api_endpoint
@function
def select_star(table_name):
    """run select * from a table"""
    # Note: creating conn here, because we need to the same in-memory database
    conn = duckdb.connect()
    assemble_table(conn, table_name)
    query = f"SELECT * FROM {table_name}"
    res = conn.execute(query).fetchall()
    return res

@api_endpoint
@function
def tpch_1():
    conn = duckdb.connect()
    assemble_table(conn, "lineitem")
    query = f"SELECT l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, " \
            f"sum(l_extendedprice * (1-l_discount)) as sum_disc_price, " \
            f"sum(l_extendedprice * (1-l_discount) * (1+l_tax)) as sum_charge, " \
            f"avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, " \
            f"avg(l_discount) as avg_disc, count(*) as count_order " \
            f"FROM lineitem " \
            f"WHERE l_shipdate <= date '1998-12-01' - interval '84' day " \
            f"GROUP BY l_returnflag, l_linestatus " \
            f"ORDER BY l_returnflag, l_linestatus"
    res = conn.execute(query).fetchall()
    return res

@api_endpoint
@function
def tpch_2():
    conn = duckdb.connect()
    assemble_table(conn, "part")
    assemble_table(conn, "supplier")
    assemble_table(conn, "partsupp")
    assemble_table(conn, "nation")
    assemble_table(conn, "region")
    query = f"SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment " \
            f"FROM part, supplier, partsupp, nation, region " \
            f"WHERE p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = 16 " \
            f"and p_type like '%STEEL' and s_nationkey = n_nationkey and n_regionkey = r_regionkey " \
            f"and r_name = 'AFRICA' and ps_supplycost = (" \
            f"SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation, region " \
            f"WHERE p_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey " \
            f"and n_regionkey = r_regionkey and r_name = 'AFRICA') " \
            f"order by s_acctbal desc, n_name, s_name, p_partkey limit 100"
    res = conn.execute(query).fetchall()
    return res
