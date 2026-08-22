# benchmarks/locustfile.py
import random
import string
from locust import HttpUser, task, between


def random_text(n=20):
    return "".join(random.choices(string.ascii_lowercase + " ", k=n))


class InferenceUser(HttpUser):
    wait_time = between(0.01, 0.05)  # small think-time between requests

    @task
    def infer(self):
        self.client.post("/infer", json={"text": random_text()})

class OverloadUser(HttpUser):
    wait_time = between(0, 0.01)  # near-zero think time = max pressure

    @task
    def infer(self):
        with self.client.post("/infer", json={"text": "x"}, catch_response=True) as r:
            if r.status_code == 503:
                r.success()  # graceful shedding counts as expected behavior, not failure