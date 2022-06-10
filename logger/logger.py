import neptune.new as neptune


class NeptuneLogger():
    """
    Neptune logger. Use it the same way you would use neptune run

    e.g
    n = NeptuneLogger()
    n['lr'] = 0.1
    n['loss'].log(0.01)
    """

    def __init__(self):
        self.api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ=="
        self.project = "rm360179/DistilHerBERT"
        self.run = neptune.init(project=self.project, api_token=self.api_token)

    def __setitem__(self, key, val):
        self.run[key] = val

    def __getitem__(self, key):
        return self.run[key]

    def __del__(self):
        self.run.stop()


class DummyLogger:
    def __init__(self):
        pass

    def __setitem__(self, key, val):
        pass

    def __getitem__(self, key):
        return self

    def __del__(self):
        pass

    def log(self, *args, **kwargs):
        pass
