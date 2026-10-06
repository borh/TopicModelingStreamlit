{ source }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.topic-modeling;
  runtime = import ./runtime.nix { inherit pkgs; };
  state = "/var/lib/topic-modeling";
  environment =
    runtime.environment
    // {
      UV_PYTHON = lib.getExe pkgs.python313;
      UV_PYTHON_DOWNLOADS = "never";
      UV_PROJECT_ENVIRONMENT = "${state}/venv";
      UV_CACHE_DIR = "${state}/uv-cache";
      HF_HOME = "${state}/huggingface";
      XDG_CACHE_HOME = "${state}/cache";
      XDG_CONFIG_HOME = "${state}/config";
      OMP_NUM_THREADS = "2";
      OPENBLAS_NUM_THREADS = "2";
      MKL_NUM_THREADS = "2";
      POLARS_MAX_THREADS = "2";
      NUMBA_NUM_THREADS = "2";
      TOKENIZERS_PARALLELISM = "false";
      TOPIC_MODELING_PUBLIC = if cfg.publicAccess then "1" else "0";
    }
    // lib.optionalAttrs (cfg.accelerator == "cuda") {
      CUDA_VISIBLE_DEVICES = cfg.gpu;
    }
    // lib.optionalAttrs (cfg.accelerator == "rocm") {
      HIP_VISIBLE_DEVICES = cfg.gpu;
    };
  serviceConfig = {
    User = "topic-modeling";
    Group = "topic-modeling";
    SupplementaryGroups = [
      "video"
      "render"
    ];
    WorkingDirectory = state;
    StateDirectory = "topic-modeling";
    StateDirectoryMode = "0750";
    UMask = "0027";
    NoNewPrivileges = true;
    PrivateTmp = true;
    ProtectHome = true;
    ProtectSystem = "strict";
    ReadWritePaths = [ state ];
    BindReadOnlyPaths = map (corpus: "-${cfg.corpusDir}/${corpus}:${state}/${corpus}") [
      "Aozora-Bunko-Fiction-Selection-2022-05-30"
      "standard-ebooks-selection"
    ];
  };
  apps = {
    bertopic = {
      port = 3331;
      file = "bertopic_app.py";
    };
  }
  // lib.optionalAttrs cfg.gensim.enable {
    gensim = {
      port = 3332;
      file = "gensim_app.py";
    };
  };
in
{
  options.services.topic-modeling = {
    enable = lib.mkEnableOption "the topic modeling Streamlit service";
    accelerator = lib.mkOption {
      type = lib.types.enum [
        "cpu"
        "cuda"
        "rocm"
      ];
      default = "cpu";
      description = "Locked uv accelerator extra to install.";
    };
    gpu = lib.mkOption {
      type = lib.types.str;
      default = "0";
      description = "Physical GPU index or UUID exposed to the service.";
    };
    corpusDir = lib.mkOption {
      type = lib.types.str;
      default = "/data/topic-modeling";
      description = "Directory containing the Aozora and Standard Ebooks corpus directories.";
    };
    publicAccess = lib.mkEnableOption "campus-only computation behind a proxy that overwrites X-Topic-Client-IP";
    gensim.enable = lib.mkEnableOption "the Gensim app on port 3332";
  };
  config = lib.mkIf cfg.enable {
    users.users.topic-modeling = {
      isSystemUser = true;
      group = "topic-modeling";
    };
    users.groups.topic-modeling = { };
    systemd.services = {
      topic-modeling-prepare = {
        description = "Install the pinned topic modeling Python environment";
        wants = [ "network-online.target" ];
        after = [ "network-online.target" ];
        restartTriggers = [
          source
          pkgs.python313
        ];
        path = runtime.packages;
        inherit environment;
        serviceConfig = serviceConfig // {
          Type = "oneshot";
          RemainAfterExit = true;
          TimeoutStartSec = "30min";
        };
        script = ''
          uv sync --locked --no-dev --no-editable --extra ${cfg.accelerator} --project ${source}
        '';
      };
    }
    // lib.mapAttrs' (
      name: app:
      lib.nameValuePair "topic-modeling-${name}" {
        description = "${name} topic modeling";
        wantedBy = [ "multi-user.target" ];
        requires = [ "topic-modeling-prepare.service" ];
        after = [ "topic-modeling-prepare.service" ];
        restartTriggers = [
          source
          pkgs.python313
        ];
        path = runtime.packages;
        inherit environment;
        serviceConfig = serviceConfig // {
          MemoryHigh = "8G";
          MemoryMax = "12G";
          CPUQuota = "200%";
          TasksMax = 256;
          ProtectKernelTunables = true;
          ProtectControlGroups = true;
          RestrictSUIDSGID = true;
          RestrictAddressFamilies = [
            "AF_UNIX"
            "AF_INET"
            "AF_INET6"
          ];
          ExecStart = "${state}/venv/bin/python -m streamlit run ${source}/src/topic_modeling_streamlit/${app.file} --server.headless=true --server.address=127.0.0.1 --server.port=${toString app.port} --server.baseUrlPath=topic-modeling-${name} --server.fileWatcherType=none --browser.gatherUsageStats=false --client.showErrorDetails=none --server.maxUploadSize=1 --server.maxMessageSize=16";
          Restart = "on-failure";
          RestartSec = 5;
        };
      }
    ) apps;
  };
}
