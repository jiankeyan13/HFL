
class BaseUpdater:
    def __init__(self, config=None):
        pass

    def update(self, global_model, aggregated_update):

        global_model.load_state_dict(aggregated_update)