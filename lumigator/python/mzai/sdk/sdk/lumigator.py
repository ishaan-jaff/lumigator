from sdk.client import ApiClient
from sdk.completions import Completions
from sdk.health import Health
from sdk.jobs import Jobs
from sdk.lm_datasets import Datasets


class LumigatorClient:
    def __init__(self, api_host: str):
        self.client = ApiClient(api_host)

        self.completions = Completions(self.client)
        self.jobs = Jobs(self.client)
        self.health = Health(self.client)
        self.datasets = Datasets(self.client)
