"""読みとアクセント句の解析結果が、音声合成用クエリの API まで届くことを確かめる E2E テスト。"""

import pytest
from fastapi.testclient import TestClient


def _get_accent_phrases(client: TestClient, text: str) -> list[tuple[str, int]]:
    """/accent_phrases API の結果を、アクセント句ごとのモーラのカナとアクセント核の位置の組にする。"""

    response = client.post("/accent_phrases", params={"text": text, "speaker": 0})
    assert response.status_code == 200
    return [
        ("".join(mora["text"] for mora in phrase["moras"]), phrase["accent"])
        for phrase in response.json()
    ]


@pytest.mark.parametrize(
    ("text", "expected_reading"),
    [
        # 「シークヮーサー」は、辞書に登録した稀な音節の「クヮ」のまま読む (「クワ」へ置き換えない)
        ("シークヮーサー味が好きです。", "シイクヮアサアアジガスキデス."),
        # 「二・二六事件」は「ニーテン」と小数点のように読まず、数字を伸ばした「ニーニーロク」で読む
        ("二・二六事件について調べる。", "ニイニイロクジケンニツイテシラベル."),
        # 「日本放送協会」の「日本」は、固有名詞として定着した「ニッポン」で読む
        ("日本放送協会のニュース。", "ニッポンホオソオキョオカイノニュウス."),
        # 人数を尋ねる「何人」は、国籍を尋ねる「ナニジン」ではなく「ナンニン」で読む
        ("何人ですか。", "ナンニンデスカ."),
        # 争いをやめる意味の「止める」は「ヤメル」で読む
        ("戦いを止める。", "タタカイオヤメル."),
        # 8 の後で促音にも読める助数詞は、促音にしない「ハチフン」を既定にする
        ("8分待った。", "ハチフンマッタ."),
        # 書画を書く「色紙」は、tsqyomi が文脈から「イロガミ」ではなく「シキシ」を選ぶ
        ("色紙に書く。", "シキシニカク."),
    ],
)
def test_reading_reaches_accent_phrases(
    client: TestClient, text: str, expected_reading: str
) -> None:
    """
    pyopenjtalk-plus の辞書・後処理・tsqyomi が決めた読みが、正規化と g2p を通った後の /accent_phrases API の結果にそのまま現れることを確認する。
    どれも依存ライブラリが古い版のままだと別の読みになる文なので、依存の更新が反映されていないときにここで止まる。
    """

    reading = "".join(mora for mora, _ in _get_accent_phrases(client, text))
    assert reading == expected_reading


def test_spaced_numbers_are_not_joined(client: TestClient) -> None:
    """
    「65 1032号機」のように空白で区切った2つの数字は、「ロクジューゴマン…」と1つの数にまとめず、別々の数として位取りして読むことを確認する。
    後ろの数の「センサンジュー」も、1つのアクセント句として続けて読む。
    """

    assert _get_accent_phrases(client, "65 1032号機") == [
        ("ロクジュウゴ'", 4),
        ("センサンジュウ", 3),
        ("ニゴオキ", 2),
    ]


def test_unknown_katakana_word_joins_following_auxiliary(client: TestClient) -> None:
    """
    辞書にないカタカナ語「ヒホ」を名詞として扱い、後ろの「です」と1つのアクセント句「ヒ＼ホデス」にまとめて読むことを確認する。
    名詞として扱わないと「です」と別のアクセント句に分かれ、核の位置もずれる。
    """

    assert _get_accent_phrases(client, "こんにちは、ヒホです") == [
        ("コンニチワ,", 5),
        ("ヒホデス", 1),
    ]
