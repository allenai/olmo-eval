"""Launch the installed SWE-ReX server with responsive command execution."""

import base64
import shlex
from pathlib import Path


def server_command(executable: str, token: str, port: int, *, install_pipx: bool = False) -> str:
    """Preserve the server interpreter and shell environment while installing its adapter."""
    source = Path(__file__).with_name("_http_command.py").read_bytes()
    payload = base64.b64encode(source + b"\ninstall()\n").decode()
    patch = (
        f"import base64; exec(compile(base64.b64decode({payload!r}), "
        "'<olmo-http-command>', 'exec'))\n"
    )
    encoded = base64.b64encode(patch.encode()).decode()
    # A temporary console script retains the installed entrypoint's shebang.
    # This works for an isolated SWE-ReX venv as well as an activated task venv.
    fallback = "exit 1"
    if executable == "swerex-remote":
        setup = (
            "python3 -m pip install pipx && python3 -m pipx ensurepath && " if install_pipx else ""
        )
        pipx = "python3 -m pipx" if install_pipx else "pipx"
        fallback = (
            setup + f"{pipx} install swe-rex && "
            f'original="$({pipx} environment --value PIPX_BIN_DIR)/swerex-remote"'
        )
    return (
        f"original=$(command -v {shlex.quote(executable)}) || "
        f"{{ {fallback}; }} || exit 1; "
        'case "$(head -n 1 "$original")" in *python*) ;; '
        "*) echo 'SWE-ReX bootstrap requires a Python-shebang entrypoint' >&2; exit 1;; esac; "
        "bootstrap=$(mktemp -d) || exit 1; "
        "trap 'rm -rf \"$bootstrap\"' EXIT; "
        '{ head -n 1 "$original"; '
        f"printf %s {shlex.quote(encoded)} | base64 -d; "
        'tail -n +2 "$original"; } > "$bootstrap/server" && '
        'chmod 700 "$bootstrap/server" && '
        f'"$bootstrap/server" --auth-token {shlex.quote(token)} --port {port}'
    )
