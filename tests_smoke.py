import importlib.util
import sys
from pathlib import Path


def load_graders_module():
    graders_path = Path(__file__).parent / "server" / "graders.py"
    spec = importlib.util.spec_from_file_location("graders", graders_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["graders"] = module
    spec.loader.exec_module(module)
    return module


def test_grader_score_range():
    graders = load_graders_module()
    result = graders.CRMTaskGrader.grade(
        expected_intent="card_fraud",
        required_slots=["customer_id", "last4_card"],
        collected_slots={"customer_id": "123", "last4_card": "9999"},
        required_workflows={"verify_identity", "freeze_card"},
        optional_workflows={"reissue_card"},
        workflows_taken=["card_fraud", "verify_identity", "freeze_card"],
        finalized=True,
        used_handoff=False,
        step_count=4,
        max_steps=10,
        avg_response_quality=0.9,
    )
    assert 0.0 <= result.score <= 1.0


if __name__ == "__main__":
    test_grader_score_range()
    print("smoke ok")
