class EscrowAPI:
    __comp = None

    @classmethod
    def set_comp(cls, api_implementation):
        print("setting escrow api composition to: ", api_implementation)
        cls.__comp = api_implementation

    @classmethod
    def get_all_accessible_des(cls):
        return cls.__comp.get_all_accessible_des()

    @classmethod
    def get_de_by_id(cls, de_id):
        return cls.__comp.get_de_by_id(de_id)

    @classmethod
    def register_data(cls,
                      username,
                      data_name,
                      data_type,
                      access_param,
                      optimistic,
                      ):
        return cls.__comp.register_data(username, data_name, data_type, access_param, optimistic)

    @classmethod
    def upload_data(cls,
                    username,
                    data_id,
                    data_in_bytes):
        return cls.__comp.upload_file(username, data_id, data_in_bytes)

    @classmethod
    def upload_policy(cls, username, user_id, api, data_id):
        return cls.__comp.upload_policy(username, user_id, api, data_id)

    @classmethod
    def suggest_share(cls, username, agents, functions, data_elements):
        return cls.__comp.suggest_share(username, agents, functions, data_elements)

    @classmethod
    def ack_data_in_share(cls, username, share_id, data_id):
        return cls.__comp.ack_data_in_share(username, share_id, data_id)

    @classmethod
    def get_de_by_id(cls, user_id, data_id) -> DataElement:
        pass
    


