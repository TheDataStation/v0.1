from dbservice import database_api


def verify_de_owner(de_id, user_id):
    # get data element owner id
    de_owner_id_resp = database_api.get_de_owner_id(de_id)
    if de_owner_id_resp["status"] == 1:
        return de_owner_id_resp
    # print("Dataset owner id is: " + str(dataset_owner_id))

    # print("User id is", user_id)
    # print("Owner id is", dataset_owner_id)
    if int(user_id) != int(de_owner_id_resp["data"]):
        print(user_id == de_owner_id_resp["data"])
        return {"status": 1, "message": "Current user is not owner of dataset"}

    return {"status": 0, "message": "verifying DE owner success"}
