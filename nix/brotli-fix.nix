{ pkgs }:
let
  brotli = pkgs.brotli.overrideAttrs (old: {
    version = "1.2.1";
    src = pkgs.fetchFromGitHub {
      owner = "google";
      repo = "brotli";
      tag = "v1.2.1";
      hash = "sha256-llY72cYffDv0ugsH0p89xIPEbElV4L8u7U/Fe8hT3gI=";
    };
  });
  freetype = pkgs.freetype.override { inherit brotli; };
  arrow-cpp = pkgs.arrow-cpp.override { inherit brotli; };
  pillow = pkgs.python313Packages.pillow.override { inherit freetype; };
  pyarrow = pkgs.python313Packages.pyarrow.override { inherit arrow-cpp; };
in
{
  inherit brotli pillow pyarrow;
  wheelhouse = pkgs.runCommand "topic-modeling-brotli-1.2.1-wheels" { } ''
    mkdir -p $out
    cp ${pillow.dist}/*.whl $out/
    cp ${pyarrow.dist}/*.whl $out/
  '';
}
