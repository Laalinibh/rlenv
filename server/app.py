from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with `pip install -e .`"
    ) from e

try:
    from ..models import CRMAction, CRMObservation
    from .customer_relationship_environment import CustomerRelationshipEnvironment
except ImportError:
    from models import CRMAction, CRMObservation
    from server.customer_relationship_environment import CustomerRelationshipEnvironment


app = create_app(
    CustomerRelationshipEnvironment,
    CRMAction,
    CRMObservation,
    env_name="customer_relationship",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
