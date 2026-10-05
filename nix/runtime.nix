{ pkgs }:
let
  dictionaries = import ./dictionaries.nix { inherit pkgs; };
  environment = {
    MECAB_DICDIR_CWJ = "${dictionaries.unidic-cwj}/share/mecab/dic/unidic-cwj";
    MECAB_DICDIR_CSJ = "${dictionaries.unidic-csj}/share/mecab/dic/unidic-csj";
    MECAB_DICDIR_NOVEL = "${dictionaries.unidic-novel}/share/mecab/dic/unidic-novel";
  }
  // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
    LD_LIBRARY_PATH =
      pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
        pkgs.bzip2
        pkgs.xz
        pkgs.zstd
      ]
      + ":/run/opengl-driver/lib";
  };
in
{
  inherit environment;
  packages = [
    pkgs.uv
    pkgs.python313
    pkgs.mecab
    pkgs.sentencepiece
  ]
  ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ pkgs.jumanpp ]
  ++ builtins.attrValues dictionaries;
  shellEnvironment = pkgs.lib.concatStringsSep "\n" (
    pkgs.lib.mapAttrsToList (
      name: value:
      "export ${name}=${pkgs.lib.escapeShellArg value}"
      + pkgs.lib.optionalString (name == "LD_LIBRARY_PATH") "\${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ) environment
  );
}
