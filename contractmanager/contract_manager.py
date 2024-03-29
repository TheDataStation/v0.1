import json

from dbservice import database_api


def propose_contract(user_id,
                     contract_id,
                     dest_agents,
                     data_elements,
                     function,
                     write_ahead_log,
                     key_manager,
                     *args,
                     **kwargs,
                     ):
    param_json = {"args": args, "kwargs": kwargs}
    param_str = json.dumps(param_json)

    # Check if this is a valid function
    function_res = database_api.get_all_functions()
    if function_res["status"] == 1:
        return function_res
    if function not in function_res["data"]:
        return {"status": 1, "message": "Contract function not valid"}

    # Add to the Contract table
    if write_ahead_log:
        wal_entry = f"database_api.create_contract({contract_id}, {function}, {param_str})"
        write_ahead_log.log(user_id, wal_entry, key_manager, )

    db_res = database_api.create_contract(contract_id, function, param_str)
    if db_res["status"] == 1:
        return db_res

    # Add to ContractDest table and ContractDE table
    for a_id in dest_agents:
        if write_ahead_log:
            wal_entry = f"database_api.create_contract_dest({contract_id}, {a_id})"
            write_ahead_log.log(user_id, wal_entry, key_manager, )
        db_res = database_api.create_contract_dest(contract_id, a_id)
        if db_res["status"] == 1:
            return db_res

    for de_id in data_elements:
        if write_ahead_log:
            wal_entry = f"database_api.create_contract_de({contract_id}, {de_id})"
            write_ahead_log.log(user_id, wal_entry, key_manager, )
        db_res = database_api.create_contract_de(contract_id, de_id)
        if db_res["status"] == 1:
            return db_res

    # Add to ContractStatus table. First get the src agents, default to DE owners.
    src_agent_de_dict = {}

    simple_de_ides = get_simple_des_from_het_des(data_elements)

    for de_id in simple_de_ides:
        owner_id_res = database_api.get_de_owner_id(de_id)
        if owner_id_res["status"] == 1:
            return owner_id_res
        src_a_id = owner_id_res["data"]
        # If the key exists, append the value to its list
        if src_a_id in src_agent_de_dict:
            src_agent_de_dict[src_a_id].add(de_id)
        else:
            src_agent_de_dict.setdefault(src_a_id, set()).add(de_id)
    for src_a_id in src_agent_de_dict:
        if write_ahead_log:
            wal_entry = f"database_api.create_contract_status({contract_id}, {src_a_id}, 0)"
            write_ahead_log.log(user_id, wal_entry, key_manager, )
        db_res = database_api.create_contract_status(contract_id, src_a_id, 0)
        if db_res["status"] == 1:
            return db_res

    # # Now: Apply CMPs to do auto-approval and auto-rejection
    # # For the source agents, we fetch all of their relevant CMPs, by function
    # # Intuition: (0, 0) -> I approve of all my DEs in this contract for any dest agents, for this f()
    # # Intuition: (dest_a_id, a_id) -> I approve of this DE, for this dest agent, for this f()
    #
    # # Step 1: Get what DEs in this contract are each src_agent are responsible for
    # # print(src_agent_de_dict)
    #
    # # Step 2: Fetch relevant CMPs that each of these src agents have specified
    # # We go over each src agent one by one
    # for src_a_id in src_agent_de_dict:
    #     dest_request_dict = {}
    #     for dest_a_id in dest_agents:
    #         dest_request_dict[dest_a_id] = set(data_elements)
    #     de_accessible_to_all = set()
    #     cur_src_approval_dict = {}
    #     auto_approval = False
    #     for dest_a_id in dest_agents:
    #         cur_src_approval_dict[dest_a_id] = set()
    #     cur_policies = database_api.get_cmp_for_src_and_f(src_a_id, function)
    #     cur_policies = list(map(lambda ele: [ele.dest_a_id, ele.de_id], cur_policies))
    #     for policy in cur_policies:
    #         if policy == [0, 0]:
    #             auto_approval = True
    #             break
    #         elif policy[0] == 0:
    #             de_accessible_to_all.add(policy[1])
    #         elif policy[1] == 0:
    #             cur_src_approval_dict[policy[0]] = set(data_elements)
    #         else:
    #             cur_src_approval_dict[policy[0]].add(policy[1])
    #     for dest_a_id in dest_agents:
    #         cur_src_approval_dict[dest_a_id].update(de_accessible_to_all)
    #
    #     for dest_a_id in dest_agents:
    #         dest_request_dict[dest_a_id] = dest_request_dict[dest_a_id].intersection(src_agent_de_dict[src_a_id])
    #     # print("Relevant request for current source agent is", dest_request_dict)
    #     # print("Auto approved request for curret source agent is ", cur_src_approval_dict)
    #     # Now we check (by each dest in dest_request_dict): if requested DE is a subset for the same key value in
    #     # src's approval. If all passes, set auto-approval to true.
    #     if not auto_approval:
    #         all_dest_passed = True
    #         for key in dest_request_dict:
    #             if not dest_request_dict[key].issubset(cur_src_approval_dict[key]):
    #                 all_dest_passed = False
    #                 break
    #         auto_approval = all_dest_passed
    #     # Finally: if auto_approval is True, we call approve contract
    #     if auto_approval:
    #         approve_contract(src_a_id, contract_id, write_ahead_log, key_manager)
    #
    # # Also, if caller is a src agent, they auto-approve
    # if user_id in src_agent_de_dict:
    #     approve_contract(user_id, contract_id, write_ahead_log, key_manager)
    #
    # # Return the status of the contract at the end
    # contract_approved = check_contract_ready(contract_id)
    # return {"status": 0, "message": "success", "contract_id": contract_id, "contract_approved": contract_approved}
    return {"status": 0, "message": "success", "contract_id": contract_id}


def show_contract(user_id, contract_id):
    # Check if caller is contract's src or dest agent
    src_agents = database_api.get_src_for_contract(contract_id)
    dest_agents = database_api.get_dest_for_contract(contract_id)
    src_agents_list = list(map(lambda ele: ele[0], src_agents))
    dest_agents_list = list(map(lambda ele: ele[0], dest_agents))
    if int(user_id) not in src_agents_list and int(user_id) not in dest_agents_list:
        print("Caller not a src or dest agent of the contract.")
        return None

    return get_contract_object(contract_id)


def show_all_contracts_as_dest(user_id):
    contract_ids_resp = database_api.get_all_contracts_for_dest(user_id)
    contract_objects = []
    if contract_ids_resp["status"] == 0:
        for c_id in contract_ids_resp["data"]:
            cur_contract = get_contract_object(c_id[0])
            contract_objects.append(cur_contract)
    return contract_objects


def show_all_contracts_as_src(user_id):
    contract_ids_resp = database_api.get_all_contracts_for_src(user_id)
    contract_objects = []
    if contract_ids_resp["status"] == 0:
        for c_id in contract_ids_resp["data"]:
            cur_contract = get_contract_object(c_id[0])
            contract_objects.append(cur_contract)
    return contract_objects


def approve_contract(user_id,
                     contract_id,
                     write_ahead_log,
                     key_manager,
                     ):
    src_agents = database_api.get_src_for_contract(contract_id)
    src_agents_list = list(map(lambda ele: ele[0], src_agents))
    if int(user_id) not in src_agents_list:
        return {"status": 1, "message": "Caller not a source agent."}

    # If in no_trust mode, we need to record this in wal
    if write_ahead_log is not None:
        wal_entry = f"database_api.approve_contract({user_id}, {contract_id})"
        write_ahead_log.log(user_id, wal_entry, key_manager, )

    approval_res = database_api.approve_contract(user_id, contract_id)
    if approval_res["status"] == 1:
        return approval_res
    # Whenever approve contract is called, we see if the associated contract is approved completely.
    # If True, add a list of policies to Policy table.
    contract_status = check_contract_ready(contract_id)
    if contract_status:
        dest_a_ids = get_dest_ids_for_contract(contract_id)
        de_ids = get_de_ids_for_contract(contract_id)
        function, function_param = get_contract_function_and_param(contract_id)
        de_ids.sort()
        de_ids_str = " ".join(de_ids)
        for a_id in dest_a_ids:
            if write_ahead_log:
                wal_entry = f"database_api.create_policy({a_id}, {de_ids_str}, {function}, {function_param})"
                write_ahead_log.log(user_id, wal_entry, key_manager, )
            db_res = database_api.create_policy(a_id, de_ids_str, function, function_param)
            if db_res["status"] == 1:
                return db_res
    return approval_res

def reject_contract(user_id,
                    contract_id,
                    write_ahead_log,
                    key_manager,
                    ):
    src_agents = database_api.get_src_for_contract(contract_id)
    src_agents_list = list(map(lambda ele: ele[0], src_agents))
    if int(user_id) not in src_agents_list:
        return {"status": 1, "message": "Caller not a source agent."}

    # If in no_trust mode, we need to record this in wal
    if write_ahead_log is not None:
        wal_entry = f"database_api.reject_contract({user_id}, {contract_id})"
        write_ahead_log.log(user_id, wal_entry, key_manager, )

    return database_api.reject_contract(user_id, contract_id)


def upload_cmp(user_id,
               dest_a_id,
               de_id,
               function,
               write_ahead_log,
               key_manager, ):
    # Check if this is a valid function
    function_res = database_api.get_all_functions()
    if function_res["status"] == 1:
        return function_res
    if function not in function_res["data"]:
        return {"status": 1, "message": "Function in CMP not valid"}

    # Check if caller is the owner of de_id
    if de_id:
        owner_id_res = database_api.get_de_owner_id(de_id)
        if owner_id_res["status"] == 1:
            return owner_id_res
        if int(user_id) != owner_id_res["data"]:
            return {"status": 1, "message": "Upload CMP failure: Caller not owner of de"}

    if write_ahead_log:
        wal_entry = f"database_api.create_cmp({user_id}, {dest_a_id}, {de_id}, {function})"
        write_ahead_log.log(user_id, wal_entry, key_manager, )

    return database_api.create_cmp(user_id, dest_a_id, de_id, function)


def check_contract_ready(contract_id):
    db_res = database_api.get_status_for_contract(contract_id)
    status_list = list(map(lambda ele: ele[0], db_res))
    if 0 in status_list:
        return False
    return True


def get_de_ids_for_contract(contract_id):
    des_in_contract = database_api.get_de_for_contract(contract_id)
    des_list = list(map(lambda ele: ele[0], des_in_contract))
    return des_list


def get_dest_ids_for_contract(contract_id):
    dest_agents = database_api.get_dest_for_contract(contract_id)
    dest_agents_list = list(map(lambda ele: ele[0], dest_agents))
    return dest_agents_list


def get_contract_function_and_param(contract_id):
    contract_db_res = database_api.get_contract(contract_id)

    function = contract_db_res["data"].function
    function_param = contract_db_res["data"].function_param
    return function, function_param


def get_contract_object(contract_id):
    # Get the destination agents, the data elements, the template and its args, and the ready status
    des_in_contract = database_api.get_de_for_contract(contract_id)
    des_list = list(map(lambda ele: ele[0], des_in_contract))
    dest_agents = database_api.get_dest_for_contract(contract_id)
    dest_agents_list = list(map(lambda ele: ele[0], dest_agents))
    contract_db_res = database_api.get_contract(contract_id)
    function_param = json.loads(contract_db_res["data"].function_param)
    ready_status = check_contract_ready(contract_id)

    contract_obj = {
        "a_dest": dest_agents_list,
        "des": des_list,
        "function": contract_db_res["data"].function,
        "args": function_param["args"],
        "kwargs": function_param["kwargs"],
        "ready_status": ready_status,
    }
    return contract_obj


def get_simple_des_from_het_des(het_de_ids):
    """
    Helper.
    Given a heterogeneous set of DEs (simple or intermediate), return a simple set
    """
    # het_des = database_api.get_des_by_ids(het_de_ids)["data"]
    # all_des = database_api.get_all_des()["data"]
    # # print(het_des)
    # simple_de_id_set = set()
    # het_de_id_set = set()
    # de_contract_dict = {}
    # for de in het_des:
    #     het_de_id_set.add(de.id)
    # for de in all_des:
    #     de_contract_dict[de.id] = de.contract_id
    # while het_de_id_set:
    #     cur_de_id = het_de_id_set.pop()
    #     if de_contract_dict[cur_de_id] == 0:
    #         simple_de_id_set.add(cur_de_id)
    #     else:
    #         # print(de_contract_dict)
    #         # convert the intermediate DE into DEs from its contract
    #         des_in_contract = database_api.get_de_for_contract(de_contract_dict[cur_de_id])
    #         des_to_add = set(map(lambda ele: ele[0], des_in_contract))
    #         het_de_id_set.update(des_to_add)
    # # print(het_de_id_set)
    # # print(simple_de_id_set)
    # return simple_de_id_set
    return het_de_ids

def check_release_status(dest_a_id, de_accessed, f, param_str):
    print(dest_a_id)
    print(de_accessed)
    print(f)
    print(param_str)
    exit()
