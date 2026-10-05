{ pkgs }:
let
  dictionaries = {
    unidic-cwj = {
      hash = "sha256-lNvcTlhqSWoLPF104eS7r08Dw5SQYRwzTQXy6DxdfjM=";
      suffix = "";
    };
    unidic-csj = {
      hash = "sha256-W0toSrgD8R+KLWos1XuaR3qJhiiYLXtwyd1sUvuYFAM=";
      suffix = "";
    };
    unidic-novel = {
      hash = "sha256-19wqA64F2CJFYaHzZTMCxOcwYoOTd6C98GHa5b/RUH8=";
      suffix = "v";
    };
  };
in
pkgs.lib.mapAttrs (
  name: dictionary:
  pkgs.${name} or (pkgs.stdenv.mkDerivation {
    pname = name;
    version = "2512";
    src = pkgs.fetchzip {
      url = "https://clrd.ninjal.ac.jp/unidic_archive/2512/${name}-${dictionary.suffix}202512.zip";
      inherit (dictionary) hash;
      stripRoot = false;
    };
    phases = [
      "unpackPhase"
      "installPhase"
    ];
    installPhase = ''
      runHook preInstall
      install -d $out/share/mecab/dic/${name}
      install -m 644 dicrc *.def *.bin *.dic $out/share/mecab/dic/${name}
      runHook postInstall
    '';
  })
) dictionaries
