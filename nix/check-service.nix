{
  nixpkgs,
  pkgs,
  module,
}:
let
  evaluated = nixpkgs.lib.nixosSystem {
    system = pkgs.stdenv.hostPlatform.system;
    modules = [
      module
      {
        boot.isContainer = true;
        system.stateVersion = "26.05";
        services.topic-modeling = {
          enable = true;
          accelerator = "cuda";
          gpu = "GPU-test";
          gensim.enable = true;
        };
      }
    ];
  };
  config = evaluated.config;
  bertopic = config.systemd.services.topic-modeling-bertopic;
  gensim = config.systemd.services.topic-modeling-gensim;
  prepare = config.systemd.services.topic-modeling-prepare;
  lib = nixpkgs.lib;
in
assert lib.hasInfix "--locked --no-dev --no-editable --extra cuda" prepare.script;
assert bertopic.requires == [ "topic-modeling-prepare.service" ];
assert lib.hasInfix "--server.address=127.0.0.1" bertopic.serviceConfig.ExecStart;
assert lib.hasInfix "--server.baseUrlPath=topic-modeling-bertopic" bertopic.serviceConfig.ExecStart;
assert lib.hasInfix "--server.port=3332" gensim.serviceConfig.ExecStart;
assert bertopic.environment.CUDA_VISIBLE_DEVICES == "GPU-test";
assert bertopic.environment.HF_HOME == "/var/lib/topic-modeling/huggingface";
assert bertopic.serviceConfig.WorkingDirectory == "/var/lib/topic-modeling";
assert bertopic.serviceConfig.ProtectHome;
pkgs.runCommand "topic-modeling-service-contracts" { } "touch $out"
