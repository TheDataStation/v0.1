from os.path import isfile
from os import path, remove, makedirs, removedirs
from models.response import *

class StorageManager:

    def __init__(self, storage_path):
        self.storage_path = storage_path

    def get_dir_path(self, data_id):
        dir_path = path.join(self.storage_path, str(data_id))
        return dir_path

    def store(self, data_name, data_id, data):
        file_name = path.basename(data_name)
        dir_path = self.get_dir_path(data_id)
        # Speficy path for file
        dst_file_path = path.join(dir_path, file_name)

        try:
            # if dir already exists, data with the same id has already been stored
            makedirs(dir_path, exist_ok=False)
            # Store the file
            f = open(dst_file_path, 'wb')
            f.write(data)
            f.close()
        except OSError as error:
            return Response(status=1,
                            message="data id=" + str(data_id) + "already exists")

        return Response(status=0, message="success")

    def remove(self, data_name, data_id):
        file_name = path.basename(data_name)
        dir_path = self.get_dir_path(data_id)
        dst_file_path = path.join(dir_path, file_name)
        try:
            remove(dst_file_path)
            removedirs(dir_path)
        except OSError as error:
            return Response(status=1,
                            message="Error removing data from storage")
        return Response(status=0, message="success")