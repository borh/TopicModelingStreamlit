{ pkgs, config, ... }:

{
  packages = [
    pkgs.git
    pkgs.mecab
    pkgs.black
    pkgs.sentencepiece
    pkgs.stdenv
    pkgs.gcc
    pkgs.stdenv.cc.cc.lib
    pkgs.cudatoolkit
    pkgs.cudaPackages.cudnn
    pkgs.cudaPackages.nccl
  ];
  enterShell = ''
    if [ ! -d 65_novel ]; then
      wget -c https://clrd.ninjal.ac.jp/unidic_archive/2203/UniDic-202203_65_novel.zip
      unzip -xn UniDic-202203_65_novel.zip
      mv 65_novel/.dicrc 65_novel/dicrc
    fi

    if [ ! -d Aozora-Bunko-Fiction-Selection-2022-05-30 ]; then
      wget -c https://nlp.lang.osaka-u.ac.jp/files/Aozora-Bunko-Fiction-Selection-2022-05-30.zip
      unzip -xn Aozora-Bunko-Fiction-Selection-2022-05-30.zip
    fi

    export TOKENIZERS_PARALLELISM=false
    export ZTIP=$(ip -o -4 addr show ztyqb7r673 | awk '{ split($4, ip_addr, "/"); print ip_addr[1] }')
  '';

  languages.python = {
    enable = true;
    package = pkgs.python311;
    #  (pkgs.python311.withPackages (p: with p; [
    #    jupyter
    #    ipykernel
    #    gensim
    #    pandas
    #    numpy
    #    plotly
    #    matplotlib
    #    fugashi
    #    pyldavis
    #    umap-learn
    #    hdbscan
    #  ]));
    poetry = {
      enable = true;
      activate.enable = true;
    };
  };

  pre-commit.hooks = {
    shellcheck.enable = true;
    black.enable = true;
  };

  # Run with:
  # devenv container serve --docker-run

  containers."serve".name = "topic-modeling-streamlit";
  containers."serve".startupCommand = config.processes.serve.exec;

  scripts."serve-bertopic".exec = "streamlit run topic-modeling-bertopic-streamlit.py  --server.port 3331 --server.address $ZTIP --browser.serverAddress nlp.lang.osaka-u.ac.jp --browser.serverPort 443 --server.baseUrlPath tmb-$(hostname)";
  scripts."serve-gensim".exec = "streamlit run topic-modeling-gensim-streamlit.py  --server.port 3332 --server.address $ZTIP --browser.serverAddress nlp.lang.osaka-u.ac.jp --browser.serverPort 443 --server.baseUrlPath tmg-$(hostname)";

  # See full reference at https://devenv.sh/reference/options/
}
