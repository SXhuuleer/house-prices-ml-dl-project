from pathlib import Path

def save_submission(submission, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(output_path, index=False)

    return output_path