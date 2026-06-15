import pandas as pd
import os

from evidently import Report
from evidently.presets import DataDriftPreset


class DriftDetector:
    def __init__(self, reference_data_path: str):
        self.reference_data = pd.read_csv(reference_data_path)

    def check_data_drift(self, current_data: pd.DataFrame, report_path: str = None):

        if report_path is None:
            report_path = os.path.join(
                os.path.dirname(__file__),
                "data_drift_report.html"
            )

        # Evidently 0.7+ way
        report = Report(metrics=[DataDriftPreset()])

        snapshot = report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        snapshot.save_html(report_path)

        print("\nDataset drift detected (check report):", report_path)

        return True


if __name__ == "__main__":

    reference_path = r"C:\Users\21270\Desktop\email-ticket-classifier\data\processed\train.csv"
    current_path = r"C:\Users\21270\Desktop\email-ticket-classifier\data\processed\test.csv"

    detector = DriftDetector(reference_path)
    current_data = pd.read_csv(current_path)

    detector.check_data_drift(current_data)