"""ユーザー辞書 API による読み保護と tsqyomi の読み分けの統合テスト。"""

from pathlib import Path

import pyopenjtalk
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from pyopenjtalk.tsqyomi import diagnostics

from voicevox_engine.tts_pipeline.model import AccentPhrase
from voicevox_engine.user_dict.model import UserDictWord


@pytest.mark.parametrize("has_shared_dictionary", [False, True])
def test_user_dictionary_reading_protection_lifecycle(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_shared_dictionary: bool,
) -> None:
    """登録・変更・インポート・最後の語の削除を読み分け結果まで検証する。"""

    # 実際にコンパイルした共有辞書を置き、利用者辞書と異なる保護設定も検証する
    ## コストは利用者辞書の優先度10より高くし、登録後は利用者の読みが最良経路に残るようにする
    if has_shared_dictionary is True:
        shared_csv_path = tmp_path / "shared.csv"
        shared_dic_path = tmp_path / "default_dictionaries" / "shared.dic"
        shared_csv_path.write_text(
            "人気,,,8000,名詞,一般,*,*,*,*,人気,ニンキ,ニンキ,0/3,*\n",
            encoding="utf-8",
        )
        pyopenjtalk.mecab_dict_index(str(shared_csv_path), str(shared_dic_path))

    # テスト用の登録語を API で削除し、利用者辞書が空の状態から比較する
    words_response = client.get("/user_dict", params={"enable_compound_accent": True})
    assert words_response.status_code == 200
    words_adapter = TypeAdapter(dict[str, UserDictWord])
    initial_words = words_adapter.validate_json(words_response.content)
    for word_uuid in initial_words:
        response = client.delete(f"/user_dict_word/{word_uuid}")
        assert response.status_code == 204

    # FastAPI の実行スレッドから診断を収集し、モデル推論と本来の診断処理をそのまま実行する
    recorded_diagnostics: list[diagnostics.TargetDiagnostic] = []
    original_record = diagnostics.record

    def record_diagnostic(diagnostic: diagnostics.TargetDiagnostic) -> None:
        """実際の推論が報告した診断をテスト側にも保存する。"""

        recorded_diagnostics.append(diagnostic)
        original_record(diagnostic)

    monkeypatch.setattr(diagnostics, "record", record_diagnostic)

    def assert_reading(
        expected_reading: str,
        expected_outcome: diagnostics.TargetDiagnosticOutcome,
    ) -> None:
        """応答の読みと tsqyomi の保護・適用結果を合わせて確認する。"""

        recorded_diagnostics.clear()
        response = client.post(
            "/accent_phrases",
            params={"text": "人気のない店", "speaker": 0},
        )
        assert response.status_code == 200
        accent_phrases = TypeAdapter(list[AccentPhrase]).validate_json(response.content)
        reading = "".join(
            mora.text for phrase in accent_phrases for mora in phrase.moras
        )
        assert reading == expected_reading
        target_diagnostics = [
            diagnostic
            for diagnostic in recorded_diagnostics
            if diagnostic.surface == "人気"
        ]
        assert len(target_diagnostics) == 1
        assert target_diagnostics[0].outcome == expected_outcome

    try:
        # 共有辞書の「ニンキ」も候補に含め、文脈に応じた「ヒトケ」をモデルが選ぶ
        assert_reading("ヒトケノナイミセ", "applied")

        # 文脈による読み分けと異なる読みを登録し、利用者の指定を優先する
        response = client.post(
            "/user_dict_word",
            params={
                "surface": "人気",
                "pronunciation": "ニンキ",
                "accent_type": 0,
                "priority": 10,
            },
        )
        assert response.status_code == 200
        word_uuid = TypeAdapter(str).validate_json(response.content)
        assert_reading("ニンキノナイミセ", "reading_protected")

        # 登録直後の辞書を保存し、後で同じ UUID の読みをインポートで復元する
        words_response = client.get(
            "/user_dict", params={"enable_compound_accent": True}
        )
        assert words_response.status_code == 200
        exported_words = words_adapter.validate_json(words_response.content)
        assert set(exported_words) == {word_uuid}

        # 登録済みの読みを変更した直後も、新しい読みが保護される
        response = client.put(
            f"/user_dict_word/{word_uuid}",
            params={
                "surface": "人気",
                "pronunciation": "ニンキモノ",
                "accent_type": 0,
                "priority": 10,
            },
        )
        assert response.status_code == 204
        assert_reading("ニンキモノノナイミセ", "reading_protected")

        # インポートによる上書きでも読みと保護設定を更新する
        response = client.post(
            "/import_user_dict",
            params={"override": True},
            json=words_adapter.dump_python(exported_words, mode="json"),
        )
        assert response.status_code == 204
        assert_reading("ニンキノナイミセ", "reading_protected")

        # 最後の語を削除すると利用者の読み保護が解除され、文脈による読み分けへ戻る
        response = client.delete(f"/user_dict_word/{word_uuid}")
        assert response.status_code == 204
        assert_reading("ヒトケノナイミセ", "applied")
        words_response = client.get("/user_dict")
        assert words_response.status_code == 200
        assert words_adapter.validate_json(words_response.content) == {}
    finally:
        # グローバル辞書を内蔵辞書へ戻し、後続テストの解析条件を揃える
        pyopenjtalk.unset_user_dict()
