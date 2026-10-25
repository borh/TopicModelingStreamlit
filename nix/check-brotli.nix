{ pkgs }:
let
  fixed = import ./brotli-fix.nix { inherit pkgs; };
in
pkgs.runCommand "topic-modeling-bundled-brotli"
  {
    nativeBuildInputs = [
      pkgs.python3
      pkgs.glibc.bin
    ];
    buildInputs = [
      fixed.pillow
      fixed.pyarrow
    ];
  }
  ''
    python ${../bin/check_bundled_brotli.py} ${fixed.wheelhouse}/*.whl > $out
  ''
