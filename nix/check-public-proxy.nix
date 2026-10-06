{ pkgs }:
let
  publicConfig = import ./public-proxy.nix {
    name = "topic-modeling-gensim";
    upstream = "127.0.0.1:18082";
  };
  configuration =
    network:
    pkgs.writeText "topic-proxy-test" ''
      {
        admin off
        auto_https off
      }
      http://127.0.0.1:18081 {
        ${builtins.replaceStrings [ "133.1.0.0/16" ] [ network ] publicConfig}
      }
    '';
in
pkgs.runCommand "topic-modeling-public-proxy-contracts"
  {
    nativeBuildInputs = [
      pkgs.caddy
      pkgs.python3
    ];
  }
  ''
    export HOME=$TMPDIR
    python ${./test_public_proxy.py} ${configuration "133.1.0.0/16"} ${configuration "127.0.0.0/8"}
    touch $out
  ''
