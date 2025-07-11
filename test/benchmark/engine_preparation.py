"""AivisSpeech Engine へアクセス可能なクライアントの生成"""

import warnings
from pathlib import Path
from typing import Literal

import httpx
from fastapi.testclient import TestClient

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.app.application import generate_app
from voicevox_engine.core.core_initializer import initialize_cores
from voicevox_engine.engine_manifest import load_manifest
from voicevox_engine.library.library_manager import LibraryManager
from voicevox_engine.preset.preset_manager import PresetManager
from voicevox_engine.setting.setting_manager import SettingHandler
from voicevox_engine.tts_pipeline.song_engine import make_song_engines_from_cores
from voicevox_engine.tts_pipeline.tts_engine import make_tts_engines_from_cores
from voicevox_engine.user_dict.user_dict_manager import UserDictionary
from voicevox_engine.utility.path_utility import engine_manifest_path, get_save_dir


def _generate_engine_fake_server(root_dir: Path) -> TestClient:
    aivm_manager = AivmManager(get_save_dir() / "Models")
    core_manager = initialize_cores(
        voicevox_dir=root_dir, use_gpu=False, enable_mock=False
    )
    tts_engines = make_tts_engines_from_cores(core_manager)
    song_engines = make_song_engines_from_cores(core_manager)
    setting_loader = SettingHandler(Path("./not_exist.yaml"))
    preset_manager = PresetManager(get_save_dir() / "presets.yaml")
    user_dict = UserDictionary()
    engine_manifest = load_manifest(engine_manifest_path())
    library_manager = LibraryManager(
        # get_save_dir() / "installed_libraries",
        # AivisSpeech では利用しない LibraryManager によるディレクトリ作成を防ぐため、get_save_dir() 直下を指定
        get_save_dir(),
        engine_manifest.supported_vvlib_manifest_version,
        engine_manifest.brand_name,
        engine_manifest.name,
        engine_manifest.uuid,
    )
    app = generate_app(
        tts_engines=tts_engines,
        song_engines=song_engines,
        aivm_manager=aivm_manager,
        core_manager=core_manager,
        setting_loader=setting_loader,
        preset_manager=preset_manager,
        character_info_dir=root_dir / "resources" / "character_info",
        user_dict=user_dict,
        engine_manifest=engine_manifest,
        library_manager=library_manager,
    )
    return TestClient(app)


ServerType = Literal["localhost", "fake"]


def generate_client(
    server: ServerType, root_dir: Path | None
) -> TestClient | httpx.Client:
    """
    AivisSpeech Engine へアクセス可能なクライアントを生成する。

    `server=localhost` では http://localhost:10101 へのクライアントを生成する。
    `server=fake` ではネットワークを介さずレスポンスを返す疑似サーバーを生成する。
    """
    if server == "fake":
        if root_dir is None:
            warn_msg = "root_dirが未指定であるため、自動的に `VOICEVOX/vv-engine` を `root_dir` に設定します。"
            warnings.warn(warn_msg, stacklevel=2)
            root_dir = Path("VOICEVOX/vv-engine")
        return _generate_engine_fake_server(root_dir)
    elif server == "localhost":
        return httpx.Client(base_url="http://localhost:10101")
    else:
        raise Exception(f"{server} はサポートされていないサーバータイプです")
