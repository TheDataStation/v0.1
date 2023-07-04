from dbservice import database_api
from common.pydantic_models.dataset import Dataset
from common.pydantic_models.user import User
from common.pydantic_models.response import Response


def verify_dataset_owner(dataset_id, user_id):
    # get data element owner id
    dataset_owner_id = database_api.get_de_owner(dataset_id)
    if dataset_owner_id is None:
        return Response(status=1, message="Error retrieving data owner.")
    # print("Dataset owner id is: " + str(dataset_owner_id))

    if user_id != dataset_owner_id:
        return Response(status=1, message="Current user is not owner of dataset")

    return Response(status=0, message="verification success")
