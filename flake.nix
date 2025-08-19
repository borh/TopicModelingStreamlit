{
  description = "Example Python development environment for Zero to Nix";

  # Flake inputs
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
  };

  # Flake outputs
  outputs =
    { self, nixpkgs }:
    let
      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      # Helper to provide system-specific attributes
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs allSystems (
          system:
          f {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      # Development environment output
      devShells = forAllSystems (
        { pkgs }:
        {
          default =
            let
              unidic-cwj =
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

                  phases = [
                    "unpackPhase"
                    "installPhase"
                  ];
                  installPhase = ''
                    runHook preInstall
                    install -d $out/share/mecab/dic/$pname
                    install -m 644 dicrc *.def *.bin *.dic $out/share/mecab/dic/$pname
                    runHook postInstall
                  '';
                };
              unidic-csj =
                let
                  pname = "unidic-csj";
                  version = "202302";
                in
                pkgs.stdenv.mkDerivation {
                  inherit pname version;

                  src = pkgs.fetchzip {
                    url = "https://ccd.ninjal.ac.jp/unidic_archive/2302/${pname}-${version}.zip";
                    name = "${pname}-${version}.zip";
                    sha256 = "sha256-EVvrj6iM4UQtSu0jIVZ2jh8OI4HXcScMD+31tTQqcTk=";
                    stripRoot = false;
                  };

                  phases = [
                    "unpackPhase"
                    "installPhase"
                  ];
                  installPhase = ''
                    runHook preInstall
                    install -d $out/share/mecab/dic/$pname
                    install -m 644 dicrc *.def *.bin *.dic $out/share/mecab/dic/$pname
                    runHook postInstall
                  '';
                };

              unidic-novel =
                let
                  pname = "unidic-novel";
                  version = "202308";
                in
                pkgs.stdenv.mkDerivation {
                  inherit pname version;

                  src = pkgs.fetchzip {
                    url = "https://ccd.ninjal.ac.jp/unidic_archive/2308/${pname}-v${version}.zip";
                    name = "${pname}-v${version}.zip";
                    sha256 = "sha256-oKgx/u4HMiwIupWyL95zq2rL4oKQC965kY1lycLm2XE=";
                    stripRoot = false;
                  };

                  phases = [
                    "unpackPhase"
                    "installPhase"
                  ];
                  installPhase = ''
                    cd $pname
                    runHook preInstall
                    install -d $out/share/mecab/dic/$pname
                    install -m 644 dicrc *.def *.bin *.dic $out/share/mecab/dic/$pname
                    runHook postInstall
                  '';
                };
            in
            pkgs.mkShell {
              shellHook = ''
                export TSIP=$(ip -o -4 addr show tailscale0 | awk '{ split($4, ip_addr, "/"); print ip_addr[1] }')

                export AZURE_API_VERSION="2024-12-01-preview"
                export AZURE_AI_API_KEY=$(cat /run/agenix/azure-ai-eastus2-key)
                export AZURE_AI_API_BASE=$(cat /run/agenix/azure-ai-base)
                export AZURE_API_KEY=$(cat /run/agenix/azure-ai-eastus2-key)
                export AZURE_API_BASE=$(cat /run/agenix/azure-ai-base)

                export OPENAI_API_VERSION="2024-02-15-preview"
                export OPENAI_API_KEY=$(cat /run/agenix/openai-api)

                export AWS_ACCESS_KEY_ID=$(cat /run/agenix/aws-access-key-id)
                export AWS_SECRET_ACCESS_KEY=$(cat /run/agenix/aws-secret-access-key)
                export AWS_REGION_NAME=us-west-2

                export OPENROUTER_API_KEY=$(cat /run/agenix/openrouter-key)

                export YOUTUBE_API_KEY=$(cat /run/agenix/youtube-data-api)

                export HF_TOKEN=$(cat /run/agenix/hf-token)

                export MECAB_DICDIR_CWJ=${unidic-cwj}/share/mecab/dic/unidic-cwj
                export MECAB_DICDIR_CSJ=${unidic-csj}/share/mecab/dic/unidic-csj
                export MECAB_DICDIR_NOVEL=${unidic-novel}/share/mecab/dic/unidic-novel
              '';
              # The Nix packages provided in the environment
              packages = [
                pkgs.mecab
                pkgs.jumanpp
                pkgs.sentencepiece
                unidic-cwj
                unidic-csj
                unidic-novel
                # We just use uv.
                # # Python plus helper tools
                # (python.withPackages (
                #   ps: with ps; [
                #     virtualenv # Virtualenv
                #     pip # The pip installer
                #   ]
                # ))
              ];
            };
        }
      );
    };
}
