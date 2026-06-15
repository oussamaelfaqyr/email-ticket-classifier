from prometheus_client import Counter, Histogram


class MetricsExporter:

    def __init__(self):

        self.prediction_counter = Counter(
            "email_predictions_total",
            "Total predictions made",
            ["label"]
        )

        self.latency_histogram = Histogram(
            "prediction_latency_seconds",
            "Prediction latency"
        )

    def record_prediction(self, label: str):
        self.prediction_counter.labels(
            label=label
        ).inc()

    def record_latency(self, latency_seconds: float):
        self.latency_histogram.observe(
            latency_seconds
        )