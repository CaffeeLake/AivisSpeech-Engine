"""/audio_query API のテスト。"""

from fastapi.testclient import TestClient
from syrupy.assertion import SnapshotAssertion

from test.utility import round_floats


def test_post_audio_query_200(
    client: TestClient, snapshot_json: SnapshotAssertion
) -> None:
    response = client.post(
        "/audio_query", params={"text": "テストです", "speaker": 888753760}
    )
    assert response.status_code == 200
    assert snapshot_json == round_floats(response.json(), round_value=2)
