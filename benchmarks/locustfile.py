"""
Locust load test for the AI Proxy & Dynamic Batching Gateway.

Point it at Nginx (default: http://localhost:8000), not directly at a
gateway replica, so the test also exercises the load-balancing layer.

Three user classes model three real traffic shapes:
  - UniqueQueryUser: every request is novel -> always a cache miss,
    exercises the full Redis Stream + batching path.
  - RepeatedQueryUser: draws from a small fixed pool of prompts -> high
    cache hit rate after warm-up, exercises the caching layer.
  - BurstUser: near-zero think time, used to validate the rate limiter and
    503/429 shedding behavior under saturation.

For a true 1M-request run, launch Locust in distributed mode (one master +
several `--worker` processes, potentially on separate machines) — a single
Locust process is itself CPU-bound on generating load well before the
gateway becomes the bottleneck.
"""
import random
import string

from locust import HttpUser, between, task

FIXED_PROMPT_POOL = [f"cached prompt #{i}" for i in range(20)]


def random_text(n=20):
    return "".join(random.choices(string.ascii_lowercase + " ", k=n))


class UniqueQueryUser(HttpUser):
    weight = 5
    wait_time = between(0.01, 0.05)

    @task
    def infer(self):
        self.client.post("/v1/infer", json={"text": random_text()})


class RepeatedQueryUser(HttpUser):
    weight = 3
    wait_time = between(0.01, 0.05)

    @task
    def infer(self):
        self.client.post("/v1/infer", json={"text": random.choice(FIXED_PROMPT_POOL)})


class BurstUser(HttpUser):
    weight = 1
    wait_time = between(0, 0.005)  # near-zero think time = max pressure

    @task
    def infer(self):
        with self.client.post("/v1/infer", json={"text": "x"}, catch_response=True) as r:
            if r.status_code in (429, 503, 504):
                r.success()  # graceful shedding/backpressure is expected, not a failure
