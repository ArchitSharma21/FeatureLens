from __future__ import annotations

import os
import socket
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('FEATURELENS_EAGER_LOAD', '0')

# The UI smoke validates Gradio construction + launch and intentionally does not
# load Qwen weights. Allow it to run in a lightweight developer environment
# where Transformers itself is absent; normal application execution still
# requires requirements.txt.
try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType('transformers')

    class _UnavailableAutoClass:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):  # pragma: no cover - safety guard
            raise RuntimeError(
                'Transformers is not installed. Install requirements.txt before model inference.'
            )

    stub.AutoModelForCausalLM = _UnavailableAutoClass
    stub.AutoTokenizer = _UnavailableAutoClass
    sys.modules['transformers'] = stub

from app import CSS, THEME, demo  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def main() -> None:
    port = _free_port()
    demo.launch(
        css=CSS,
        theme=THEME,
        ssr_mode=False,
        prevent_thread_lock=True,
        server_name='127.0.0.1',
        server_port=port,
        show_error=True,
    )
    print(f'FeatureLens UI launch smoke: PASS (port {port})')
    demo.close()


if __name__ == '__main__':
    main()
