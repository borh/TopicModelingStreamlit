{ pkgs, config, lib, ... }:
let
  unidic-cwj = (
    let
      pname = "unidic-cwj";
      version = "202302";
    in
    pkgs.stdenv.mkDerivation {
      inherit pname version;

      src = pkgs.fetchzip {
        url = "https://ccd.ninjal.ac.jp/unidic_archive/2302/${pname}-${version}.zip";
        name = "${pname}-${version}.zip";
        sha256 = "sha256-VJEbOf6WWg5e0MqoLIzXWeBqf0zw7idMMqoeTHOzsEw=";
        stripRoot = false;
      };

      phases = [ "unpackPhase" "installPhase" ];
      installPhase = ''
        runHook preInstall
        install -d $out/share/mecab/dic/$pname
        install -m 644 dicrc *.def *.bin *.dic $out/share/mecab/dic/$pname
        runHook postInstall
      '';
    }
  );
in
{
  env.UNIDIC_CWJ_PATH = "${unidic-cwj}/share/mecab/dic/unidic-cwj";
  packages = [
    pkgs.mecab
    pkgs.jumanpp
    pkgs.sentencepiece
    unidic-cwj
  ] ++ lib.optionals pkgs.stdenv.isDarwin [
    pkgs.llvmPackages_14.stdenv
    pkgs.llvmPackages_14.stdenv.cc
  ] ++ lib.optionals (!pkgs.stdenv.isDarwin) [
    # Build deps
    pkgs.stdenv
    pkgs.stdenv.cc
    # Dev
    pkgs.playwright
    pkgs.playwright-driver
    pkgs.black
  ] ++ lib.optionals (!config.container.isBuilding) [
    pkgs.git
  ];

  env.LD_LIBRARY_PATH = lib.mkIf pkgs.stdenv.isLinux (
    lib.makeLibraryPath (with pkgs; [
      gcc-unwrapped.lib
      linuxPackages_latest.nvidia_x11
      zlib
      cmake
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      cudaPackages.libcublas
      cudaPackages.libcurand
      cudaPackages.libcufft
      cudaPackages.libcusparse
      cudaPackages.cuda_nvtx
      cudaPackages.cuda_cupti
      cudaPackages.cuda_nvrtc
      cudaPackages.nccl
    ])
  );
  env.CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
  env.CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";

  enterShell = ''
    if [ ! -d 65_novel ]; then
      wget -c https://clrd.ninjal.ac.jp/unidic_archive/2203/UniDic-202203_65_novel.zip
      ${pkgs.unzip}/bin/unzip -xn UniDic-202203_65_novel.zip
      mv 65_novel/.dicrc 65_novel/dicrc
    fi

    if [ ! -d Aozora-Bunko-Fiction-Selection-2022-05-30 ]; then
      wget -c https://nlp.lang.osaka-u.ac.jp/files/Aozora-Bunko-Fiction-Selection-2022-05-30.zip
      ${pkgs.unzip}/bin/unzip -xn Aozora-Bunko-Fiction-Selection-2022-05-30.zip
    fi

    if [ ! -d standard-ebooks-selection ]; then
      wget -c https://nlp.lang.osaka-u.ac.jp/files/standard-ebooks-selection.zip
      ${pkgs.unzip}/bin/unzip -xn standard-ebooks-selection.zip
    fi

    test_hosts=(kurent speely)
    current_hostname=$(hostname)
    for hostname in "''${test_hosts[@]}"
    do
      if [ "$current_hostname" == "$hostname" ]; then
        export PLAYWRIGHT_BROWSERS_PATH=$(nix build --print-out-paths nixpkgs#playwright-driver.browsers)
      break 
    fi
    done

    export TOKENIZERS_PARALLELISM=false
    export ZTIP=$(ip -o -4 addr show ztyqb7r673 | awk '{ split($4, ip_addr, "/"); print ip_addr[1] }')
  '';

  languages.python = {
    enable = true;
    package = # pkgs.python311;
      (pkgs.python311.withPackages (p: with p; [
        playwright
        # jupyter
        # ipykernel
        # gensim
        # pandas
        # numpy
        # plotly
        # matplotlib
        # fugashi
        # pyldavis
        # umap-learn
        # hdbscan
      ]));
    poetry = {
      enable = true;
      activate.enable = true;
    };
  };
  languages.javascript.enable = true;

  pre-commit.hooks = {
    shellcheck.enable = true;
    black.enable = true;
  };

  # Run with:
  # devenv container bertopic --docker-run

  containers.bertopic.name = "topic-modeling-bertopic-streamlit";
  containers.bertopic.startupCommand = config.processes.serve-bertopic.exec;
  processes.serve-bertopic.exec = config.scripts.serve-bertopic.exec;

  containers.gensim.name = "topic-modeling-gensim-streamlit";
  containers.gensim.startupCommand = config.processes.serve-gensim.exec;
  processes.serve-gensim.exec = config.scripts.serve-gensim.exec;

  scripts.serve-bertopic.exec = "poetry run streamlit run topic-modeling-bertopic-streamlit.py --server.port 3331 --server.address $ZTIP --browser.serverAddress nlp.lang.osaka-u.ac.jp --browser.serverPort 443 --server.baseUrlPath topic-modeling-bertopic --server.headless true";
  scripts.serve-gensim.exec = "poetry run streamlit run topic-modeling-gensim-streamlit.py --server.port 3332 --server.address $ZTIP --browser.serverAddress nlp.lang.osaka-u.ac.jp --browser.serverPort 443 --server.baseUrlPath topic-modeling-gensim --server.headless true";

  scripts.test-1.exec = "pytest --numprocesses 1 -s playwright_test.py";
  scripts.stress-test.exec = "pytest --numprocesses 15 -s playwright_test.py";

  # See full reference at https://devenv.sh/reference/options/
}
