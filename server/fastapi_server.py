import os
import shutil
import argparse

from fastapi import FastAPI, File, UploadFile
from typing import Optional
import uvicorn

from main import initialize_system
from dsapplicationregistration.dsar_core import get_registered_api_endpoint, get_registered_functions

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# register agent
@app.post("/register_agent")
def register_agent(username, password):
    return ds.create_user(username, password)


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

    # Clean up
    if os.path.exists("data_station.db"):
        os.remove("data_station.db")
    folders = ['SM_storage']
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    args = parser.parse_args()
    print(args)
    port = args.port
    host = args.host

    # Initializing an instance of DS, according to the config file
    global ds
    ds = initialize_system(args.ds_config)

    log_path = ds.data_station_log.log_path
    if os.path.exists(log_path):
        os.remove(log_path)

    # Get the list of api_endpoints defined in the EPM (not including @functions)
    # Create FastAPI endpoints for them
    api_endpoints = get_registered_api_endpoint()
    functions = get_registered_functions()
    for api in api_endpoints:
        print(api.__name__)
        if api.__name__ == "upload_de":
            upload_de_def = api

            def upload_file(user_id: int, data_id: int, file: UploadFile = File(...)):
                contents = file.file.read()
                return upload_de_def(user_id, data_id, contents)
            app.add_api_route(f"/{api.__name__}", upload_file, methods=["POST"])
        elif api.__name__ == "propose_contract":
            propose_contract_def = api

            def propose_contract(user_id: int,
                                 dest_agents: list[int],
                                 data_elments: list[int],
                                 f,
                                 args: Optional[list] = None):
                if not args:
                    return propose_contract_def(user_id, dest_agents, data_elments, f)
                else:
                    return propose_contract_def(user_id, dest_agents, data_elments, f, *args)
            app.add_api_route(f"/{api.__name__}", propose_contract, methods=["POST"])
        else:
            app.add_api_route(f"/{api.__name__}", api, methods=["POST"])

    uvicorn.run(app, host='0.0.0.0', port=8000)
