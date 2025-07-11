"""/engine_manifest API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.utility import hash_long_string


def test_get_engine_manifest_200(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    response = client.get("/engine_manifest")
    assert response.status_code == 200
    response_json = response.json()
    response_json.pop("dependency_licenses", None)  # dependency_licenses を除外
    assert snapshot_json == hash_long_string(response_json)
