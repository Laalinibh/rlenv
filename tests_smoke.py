import importlib.util
import sys
import types
from pathlib import Path
from pydantic import BaseModel


def _install_openenv_stubs():
    """Install lightweight stubs so models.py and env.py can import openenv types."""
    if "openenv" in sys.modules:
        return

    # Stub base classes matching openenv signatures
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0
        metadata: dict = {}

    class State:
        def __init__(self, episode_id="", step_count=0, **kw):
            self.episode_id = episode_id
            self.step_count = step_count
            for k, v in kw.items():
                setattr(self, k, v)

    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = False
        def reset(self, **kw): ...
        def step(self, action): ...

    # Build module hierarchy
    openenv = types.ModuleType("openenv")
    core = types.ModuleType("openenv.core")
    env_server = types.ModuleType("openenv.core.env_server")
    types_mod = types.ModuleType("openenv.core.env_server.types")
    interfaces_mod = types.ModuleType("openenv.core.env_server.interfaces")

    types_mod.Action = Action
    types_mod.Observation = Observation
    types_mod.State = State
    interfaces_mod.Environment = Environment

    openenv.core = core
    core.env_server = env_server
    env_server.types = types_mod
    env_server.interfaces = interfaces_mod

    sys.modules["openenv"] = openenv
    sys.modules["openenv.core"] = core
    sys.modules["openenv.core.env_server"] = env_server
    sys.modules["openenv.core.env_server.types"] = types_mod
    sys.modules["openenv.core.env_server.interfaces"] = interfaces_mod


_install_openenv_stubs()


def load_env_module():
    """Load the environment module directly, bypassing server/__init__.py"""
    # Pre-load dependencies so the except-block imports work
    load_models_module()
    # Ensure 'server' package exists in sys.modules
    if "server" not in sys.modules:
        import types as _t
        sys.modules["server"] = _t.ModuleType("server")
    # Load task_bank
    tb_path = Path(__file__).parent / "server" / "task_bank.py"
    tb_spec = importlib.util.spec_from_file_location("server.task_bank", tb_path)
    tb_mod = importlib.util.module_from_spec(tb_spec)
    assert tb_spec and tb_spec.loader
    sys.modules["server.task_bank"] = tb_mod
    tb_spec.loader.exec_module(tb_mod)
    # Load graders
    gr_path = Path(__file__).parent / "server" / "graders.py"
    gr_spec = importlib.util.spec_from_file_location("server.graders", gr_path)
    gr_mod = importlib.util.module_from_spec(gr_spec)
    assert gr_spec and gr_spec.loader
    sys.modules["server.graders"] = gr_mod
    gr_spec.loader.exec_module(gr_mod)
    # Load environment
    env_path = Path(__file__).parent / "server" / "customer_relationship_environment.py"
    spec = importlib.util.spec_from_file_location("customer_relationship_environment", env_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_models_module():
    models_path = Path(__file__).parent / "models.py"
    spec = importlib.util.spec_from_file_location("models", models_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["models"] = module
    spec.loader.exec_module(module)
    return module


def load_graders_module():
    load_models_module()  # graders imports from models
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
        usefulness_100=75.0,
        tool_failures=0,
        repair_loops=0,
        compliance_disclosures=1,
        auth_checkpoints=1,
        policies_referenced=2,
        total_policies=2,
        high_risk=True,
    )
    assert 0.0 <= result.score <= 1.0
    assert "compliance" in result.breakdown


def test_grader_zero_score():
    """Empty trajectory should produce a low but valid score."""
    graders = load_graders_module()
    result = graders.CRMTaskGrader.grade(
        expected_intent="card_fraud",
        required_slots=["customer_id", "last4_card"],
        collected_slots={},
        required_workflows={"verify_identity", "freeze_card"},
        optional_workflows=set(),
        workflows_taken=[],
        finalized=False,
        used_handoff=True,
        step_count=10,
        max_steps=10,
        usefulness_100=0.0,
        tool_failures=3,
        repair_loops=3,
        high_risk=True,
    )
    assert 0.0 <= result.score <= 1.0
    assert result.score < 0.3  # should be low


def test_grader_perfect_score():
    """Perfect trajectory should produce a high score."""
    graders = load_graders_module()
    result = graders.CRMTaskGrader.grade(
        expected_intent="card_fraud",
        required_slots=["customer_id", "last4_card"],
        collected_slots={"customer_id": "C-123", "last4_card": "4532"},
        required_workflows={"verify_identity", "freeze_card"},
        optional_workflows={"reissue_card"},
        workflows_taken=["card_fraud", "verify_identity", "freeze_card", "reissue_card"],
        finalized=True,
        used_handoff=False,
        step_count=4,
        max_steps=10,
        usefulness_100=90.0,
        tool_failures=0,
        repair_loops=0,
        compliance_disclosures=2,
        auth_checkpoints=1,
        policies_referenced=2,
        total_policies=2,
        high_risk=True,
    )
    assert result.score > 0.75


def test_normalized_usefulness():
    """Usefulness formula produces values in [0, 100]."""
    graders = load_graders_module()
    u = graders.CRMTaskGrader.normalized_usefulness(
        correctness=2.0, completeness=2.0, clarity=2.0,
        actionability=2.0, safety=2.0,
    )
    assert u == 100.0

    u_zero = graders.CRMTaskGrader.normalized_usefulness(
        correctness=0.0, completeness=0.0, clarity=0.0,
        actionability=0.0, safety=0.0,
    )
    assert u_zero == 0.0


def test_session_satisfaction_hat():
    """Session satisfaction proxy returns value in [0, 1]."""
    graders = load_graders_module()
    sat = graders.CRMTaskGrader.session_satisfaction_hat(
        task_success=1.0, avg_usefulness_01=1.0,
        cost=0.0, tool_failures=0.0,
        repair_loops=0.0, handoff_penalty=0.0,
    )
    assert 0.0 <= sat <= 1.0
    assert sat > 0.8  # should be high

    sat_bad = graders.CRMTaskGrader.session_satisfaction_hat(
        task_success=0.0, avg_usefulness_01=0.0,
        cost=1.0, tool_failures=1.0,
        repair_loops=1.0, handoff_penalty=1.0,
    )
    assert 0.0 <= sat_bad <= 1.0
    assert sat_bad < 0.2  # should be low


def test_environment_reset_step():
    """Full environment reset and step cycle works."""
    models = load_models_module()
    env_mod = load_env_module()
    CustomerRelationshipEnvironment = env_mod.CustomerRelationshipEnvironment
    CRMAction = models.CRMAction

    env = CustomerRelationshipEnvironment()

    # Test all 3 tasks
    for task_id in ["easy_card_freeze", "medium_dispute_retention", "hard_business_churn_prevention"]:
        obs = env.reset(task_id=task_id)
        assert obs.done is False
        assert obs.task.task_id == task_id
        assert len(obs.required_slots) > 0
        assert len(obs.compliance_policies) > 0
        assert obs.account_context.customer_name != ""
        assert len(obs.knowledge_base) > 0

        # Take one step
        action = CRMAction(
            action_type="analyze",
            intent="card_fraud" if task_id == "easy_card_freeze" else "fee_dispute",
            response_text="Let me verify your identity for security.",
            confidence=0.8,
            event_type="auth_checkpoint",
        )
        obs2 = env.step(action)
        assert obs2.reward != 0.0  # should have some reward signal
        assert 0.0 <= obs2.progress_score <= 1.0


def test_environment_full_easy_trajectory():
    """Run a complete easy task trajectory and check final grade."""
    models = load_models_module()
    env_mod = load_env_module()
    CustomerRelationshipEnvironment = env_mod.CustomerRelationshipEnvironment
    CRMAction = models.CRMAction

    env = CustomerRelationshipEnvironment()
    obs = env.reset(task_id="easy_card_freeze")

    # Step 1: Analyze
    obs = env.step(CRMAction(
        action_type="analyze", intent="card_fraud",
        response_text="I understand you're reporting unauthorized transactions. Let me verify your identity to secure your account immediately.",
        extracted_slots={"customer_id": "C-98210", "last4_card": "4532"},
        confidence=0.9, event_type="auth_checkpoint",
    ))

    # Step 2: verify_identity workflow
    obs = env.step(CRMAction(
        action_type="workflow", workflow="verify_identity",
        response_text="Identity verified. Per our Fraud Response SLA policy, I'll freeze your card right away to protect your account.",
        confidence=0.95, event_type="compliance_disclosure",
    ))

    # Step 3: freeze_card workflow
    obs = env.step(CRMAction(
        action_type="workflow", workflow="freeze_card",
        response_text="Your card ending in 4532 is now frozen. Under Regulation E, you have zero liability since you reported within 2 business days.",
        confidence=0.95, event_type="compliance_disclosure",
    ))

    # Step 4: finalize
    obs = env.step(CRMAction(
        action_type="finalize",
        response_text="Your compromised card has been frozen and a replacement will arrive in 3-5 business days. Is there anything else I can help you with?",
        confidence=0.95, event_type="confirmation",
    ))

    assert obs.done is True
    grade = obs.metadata["grade"]
    assert grade["score"] > 0.6
    assert grade["compliance"] > 0.0


if __name__ == "__main__":
    test_grader_score_range()
    test_grader_zero_score()
    test_grader_perfect_score()
    test_normalized_usefulness()
    test_session_satisfaction_hat()
    test_environment_reset_step()
    test_environment_full_easy_trajectory()
    print("all 7 smoke tests passed")
