{ pkgs }:
let
  runtime = import ./runtime.nix { inherit pkgs; };
in
pkgs.writeShellApplication {
  name = "topic-modeling";
  runtimeInputs = runtime.packages;
  text = ''
    ${runtime.shellEnvironment}
    export UV_PYTHON=${pkgs.lib.getExe pkgs.python313}
    export UV_PYTHON_DOWNLOADS=never
    if [ ! -f src/topic_modeling_streamlit/bertopic_app.py ]; then
      echo "Run this command from the TopicModelingStreamlit checkout." >&2
      exit 1
    fi
    if [ -z "''${ACCELERATOR:-}" ]; then
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        ACCELERATOR=cuda
      elif [ -e /dev/kfd ]; then
        ACCELERATOR=rocm
      elif [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
        ACCELERATOR=mlx
      else
        ACCELERATOR=cpu
      fi
    fi
    uv sync --locked --extra "$ACCELERATOR"
    exec uv run --no-sync streamlit run src/topic_modeling_streamlit/bertopic_app.py "$@"
  '';
}
