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
          publicAccess = true;
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
assert bertopic.environment.TOPIC_MODELING_PUBLIC == "1";
assert bertopic.serviceConfig.MemoryMax == "12G";
assert bertopic.serviceConfig.CPUQuota == "200%";
assert bertopic.serviceConfig.TasksMax == 256;
assert
  config.systemd.services.topic-modeling-bertopic-public.environment.TOPIC_MODELING_READ_ONLY == "1";
assert
  config.systemd.services.topic-modeling-gensim-public.environment.TOPIC_MODELING_APP == "gensim";
assert lib.hasInfix "--server.port=3333"
  config.systemd.services.topic-modeling-bertopic-public.serviceConfig.ExecStart;
assert lib.hasInfix "--server.baseUrlPath=topic-modeling-gensim"
  config.systemd.services.topic-modeling-gensim-public.serviceConfig.ExecStart;
assert config.systemd.services.topic-modeling-gensim-public.serviceConfig.MemoryMax == "1G";
assert
  config.systemd.services.topic-modeling-gensim-public.serviceConfig.User == "topic-modeling-public";
assert config.systemd.services.topic-modeling-gensim-public.serviceConfig.StateDirectory == [ ];
assert config.systemd.services.topic-modeling-gensim-public.serviceConfig.ReadWritePaths == [ ];
assert builtins.elem "/var/lib/topic-modeling/cache"
  config.systemd.services.topic-modeling-gensim-public.serviceConfig.InaccessiblePaths;
assert lib.hasInfix "--client.showErrorDetails=none" bertopic.serviceConfig.ExecStart;
assert lib.hasInfix "--server.maxUploadSize=1" gensim.serviceConfig.ExecStart;
pkgs.runCommand "topic-modeling-service-contracts" { } "touch $out"
