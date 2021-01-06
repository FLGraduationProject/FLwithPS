import parameters as pm

class parameterServer():
    def __init__(self, initial_params):
        self.globalParams = pm.Parameters(initial_params)
        self.update_infos = []

    def upload(self, update_info):
        self.update_infos.append(update_info)

    def update(self):
        print("------------------server updating------------------")
        for info in self.update_infos:
            self.globalParams.update_params(info)
        self.update_info = []