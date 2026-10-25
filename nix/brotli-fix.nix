{ pkgs }:
let
  lib = pkgs.lib;
  minimumBrotli = "1.2.1";
  upstreamBrotli = pkgs.brotli;
  brotli =
    if lib.versionAtLeast upstreamBrotli.version minimumBrotli then
      upstreamBrotli
    else
      upstreamBrotli.overrideAttrs (_old: {
        version = minimumBrotli;
        src = pkgs.fetchFromGitHub {
          owner = "google";
          repo = "brotli";
          tag = "v${minimumBrotli}";
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
  wheelhouse = pkgs.runCommand "topic-modeling-brotli-fixed-wheels" { } ''
    mkdir -p $out
    cp ${pillow.dist}/*.whl $out/
    cp ${pyarrow.dist}/*.whl $out/
  '';
}
