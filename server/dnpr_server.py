import os
import argparse

from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import uvicorn

from main import initialize_system
from dsapplicationregistration.dsar_core import get_registered_api_endpoint
from common.general_utils import clean_test_env

app = FastAPI()


# Create a new user.
@app.post("/create_agent")
def create_agent(username, password):
    return ds.create_agent(username, password)


# Login Related
# Specifying oauth2_scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.post("/token/", include_in_schema=False)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    response = ds.login_agent(form_data.username,
                              form_data.password)
    if response["status"] == 1:
        # raise an exception to indicate login has failed
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
        access_token = response["data"]
        return {"access_token": access_token, "token_type": "bearer"}


@app.post("/upload_data_in_csv")
async def upload_data_in_csv(token: str = Depends(oauth2_scheme), file: UploadFile = File(...)):
    de_in_bytes = file.file.read()
    return ds.call_api(token, "upload_data_in_csv", de_in_bytes)


@app.post("/approve_all_causal_queries")
async def approval_all_causal_queries(token: str = Depends(oauth2_scheme)):
    return ds.call_api(token, "approve_all_causal_queries")


@app.post("/run_causal_query")
async def run_causal_query(user_de_id: int,
                           additional_vars: list[str],
                           dag_spec: list[tuple[str, str]],
                           treatment: str,
                           outcome: str,
                           token: str = Depends(oauth2_scheme)):
    return ds.call_api(token, "run_causal_query", user_de_id, additional_vars, dag_spec, treatment, outcome)


if __name__ == "__main__":
    """
    Initializes an api to communicate by HTTP with a client.
    """
    parser = argparse.ArgumentParser(
        prog='FastAPI Client for DS',
        description='A Client API for Data Station',
        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--ds_config', default='data_station_config.yaml', type=str)
    parser.add_argument('-p', '--port', default=8080, type=int)
    parser.add_argument('-hs', '--host', default="localhost", type=str)

    clean_test_env()

    args = parser.parse_args()
    # print(args)
    port = args.port
    host = args.host

    # Initializing an instance of DS, according to the config file
    global ds
    ds = initialize_system(args.ds_config)

    log_path = ds.data_station_log.log_path
    if os.path.exists(log_path):
        os.remove(log_path)

    uvicorn.run(app, host='0.0.0.0', port=8000)
