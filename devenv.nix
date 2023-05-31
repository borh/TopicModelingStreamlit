{ pkgs, config, lib, ... }:

{
  packages = [
    pkgs.mecab
    pkgs.jumanpp
    pkgs.sentencepiece
  ] ++ lib.optionals pkgs.stdenv.isDarwin [
    pkgs.llvmPackages_14.stdenv
    pkgs.llvmPackages_14.stdenv.cc
  ] ++ lib.optionals (!pkgs.stdenv.isDarwin) [
    # build deps
    pkgs.stdenv
    pkgs.stdenv.cc
    pkgs.stdenv.cc.cc.lib # runtime dep; comment out when building packages
    pkgs.cudatoolkit
    pkgs.cudaPackages.cudnn
    pkgs.cudaPackages.nccl
    # Dev
    pkgs.playwright
    pkgs.playwright-driver
    pkgs.black
  ] ++ lib.optionals (!config.container.isBuilding) [
    pkgs.git
  ];
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

    export PLAYWRIGHT_BROWSERS_PATH=$(nix build --print-out-paths nixpkgs#playwright-driver.browsers)
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

  scripts.serve-bertopic.exec = "streamlit run topic-modeling-bertopic-streamlit.py --server.port 3331 --server.address $ZTIP --browser.serverAddress nlp.lang.osaka-u.ac.jp --browser.serverPort 443 --server.baseUrlPath topic-modeling-bertopic --server.headless true";
  scripts.serve-gensim.exec = "streamlit run topic-modeling-gensim-streamlit.py --server.port 3332 --server.address $ZTIP --browser.serverAddress nlp.lang.osaka-u.ac.jp --browser.serverPort 443 --server.baseUrlPath topic-modeling-gensim --server.headless true";

  scripts.test-1.exec = "pytest --numprocesses 1 -s playwright_test.py";
  scripts.stress-test.exec = "pytest --numprocesses 15 -s playwright_test.py";

  # See full reference at https://devenv.sh/reference/options/
}
