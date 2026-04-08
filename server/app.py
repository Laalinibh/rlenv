from __future__ import annotations

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


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
