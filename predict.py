
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "learnmate_model.joblib"

saved = joblib.load(MODEL_PATH)
model = saved["model"]
FEATURES = saved["features"]
LABELS = saved["labels"]


def predict_recommendation(
    previous_accuracy,
    accuracy,
    attempts,
    response_time,
    difficulty
):
    """Return a learning recommendation from quiz-performance data."""

    values = {
        "previous_accuracy": previous_accuracy,
        "accuracy": accuracy,
        "attempts": attempts,
        "response_time": response_time,
        "difficulty": difficulty,
    }

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        for value in values.values()
    ):
        raise ValueError("All inputs must be numbers.")

    if not 0 <= previous_accuracy <= 1:
        raise ValueError("previous_accuracy must be between 0 and 1.")
    if not 0 <= accuracy <= 1:
        raise ValueError("accuracy must be between 0 and 1.")
    if attempts < 0 or response_time < 0:
        raise ValueError("attempts and response_time cannot be negative.")
    if difficulty not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2, or 3.")

    input_data = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    if prediction not in LABELS:
        raise ValueError("The model returned an unknown label.")

    return prediction


if __name__ == "__main__":
    result = predict_recommendation(
        previous_accuracy=0.50,
        accuracy=0.60,
        attempts=2,
        response_time=70,
        difficulty=2,
    )
    print("LearnMate recommendation:", result)
