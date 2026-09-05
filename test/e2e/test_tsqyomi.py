"""tsqyomi による読み分けから音声合成までの E2E テスト。"""

from io import BytesIO

import numpy as np
import pyopenjtalk
import pytest
import soundfile
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from pyopenjtalk.tsqyomi import diagnostics

from test.e2e.single_api.utils import get_first_style_id
from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.model import AccentPhrase


@pytest.mark.parametrize(
    "endpoint", ["/audio_query", "/audio_query_from_preset", "/accent_phrases"]
)
@pytest.mark.parametrize(
    ("text", "surface", "baseline_reading", "selected_reading", "expected_reading"),
    [
        ("人気のない店", "人気", "ニンキ", "ヒトケ", "ヒトケノナイミセ"),
        ("一寸です", "一寸", "チョット", "イッスン", "イッスンデス"),
    ],
)
def test_tsqyomi_reading_and_synthesis(
    client_with_default_model: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    text: str,
    surface: str,
    baseline_reading: str,
    selected_reading: str,
    expected_reading: str,
) -> None:
    """実モデルで読みが変わる文章を各 API で解析し、その音素列から音声を生成する。"""

    client = client_with_default_model
    style_id = get_first_style_id(client)

    # pyopenjtalk-plus の回帰症例のうち、モデル無効時には別の読みになる文章を使う
    baseline = "".join(
        feature["pron"] for feature in pyopenjtalk.run_frontend(text, use_tsqyomi=False)
    )
    assert baseline.startswith(baseline_reading)
    assert not baseline.startswith(selected_reading)

    # API の実行スレッドで発生した診断を収集し、実際にモデルが読みを選んだことを確認する
    records: list[diagnostics.TargetDiagnostic] = []
    original_record = diagnostics.record

    def record_diagnostic(diagnostic: diagnostics.TargetDiagnostic) -> None:
        """本来の診断処理を実行し、検査用にも記録する。"""

        original_record(diagnostic)
        records.append(diagnostic)

    monkeypatch.setattr(diagnostics, "record", record_diagnostic)
    params: dict[str, str | int] = {"text": text, "speaker": style_id}
    # テスト用のプリセットでも、読み分けは同じ実モデルを通る
    if endpoint == "/audio_query_from_preset":
        params = {"text": text, "preset_id": 1}
    response = client.post(endpoint, params=params)
    assert response.status_code == 200
    assert any(
        record.surface == surface
        and record.outcome == "applied"
        and record.selected_pronunciation == selected_reading
        and record.score_margin is not None
        for record in records
    )

    # アクセント句 API の結果は、音声合成用クエリのアクセント句へそのまま設定する
    if endpoint == "/accent_phrases":
        phrases = TypeAdapter(list[AccentPhrase]).validate_python(response.json())
        query_response = client.post(
            "/audio_query", params={"text": text, "speaker": style_id}
        )
        assert query_response.status_code == 200
        query = AudioQuery.model_validate(query_response.json())
        query.accent_phrases = phrases
    else:
        query = AudioQuery.model_validate(response.json())
        phrases = query.accent_phrases

    assert (
        "".join(mora.text for phrase in phrases for mora in phrase.moras)
        == expected_reading
    )

    # プリセット内のサンプル話者とは独立に、インストール済みの話者で実際に合成する
    synthesis = client.post(
        "/synthesis", params={"speaker": style_id}, json=query.model_dump()
    )
    assert synthesis.status_code == 200
    assert synthesis.headers["content-type"] == "audio/wav"
    wave, sample_rate = soundfile.read(BytesIO(synthesis.content))
    assert sample_rate == query.outputSamplingRate
    assert len(wave) > 0
    assert np.isfinite(wave).all()
    assert np.max(np.abs(wave)) > 0


def test_explicit_kana_preserves_reading(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """読みを直接指定する API は、指定した読みを維持する。"""

    records: list[diagnostics.TargetDiagnostic] = []
    original_record = diagnostics.record

    def record_diagnostic(diagnostic: diagnostics.TargetDiagnostic) -> None:
        """本来の診断処理を実行し、検査用にも記録する。"""

        original_record(diagnostic)
        records.append(diagnostic)

    monkeypatch.setattr(diagnostics, "record", record_diagnostic)
    response = client.post(
        "/accent_phrases",
        params={"text": "ニ'ンキ", "speaker": 0, "is_kana": True},
    )
    assert response.status_code == 200
    phrases = TypeAdapter(list[AccentPhrase]).validate_python(response.json())
    assert "".join(mora.text for phrase in phrases for mora in phrase.moras) == "ニンキ"
    assert records == []
