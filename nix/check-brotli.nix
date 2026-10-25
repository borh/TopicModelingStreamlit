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
  }
  ''
    python ${../bin/check_bundled_brotli.py} ${fixed.wheelhouse}/*.whl > $out
  ''
