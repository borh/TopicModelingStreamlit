{
  description = "Topic modeling with Streamlit, uv, and native NLP tools";

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
      checks.x86_64-linux.service = import ./nix/check-service.nix {
        inherit nixpkgs;
        pkgs = import nixpkgs { system = "x86_64-linux"; };
        module = self.nixosModules.default;
      };
      checks.x86_64-linux.public-proxy = import ./nix/check-public-proxy.nix {
        pkgs = import nixpkgs { system = "x86_64-linux"; };
      };
      lib.publicCaddyConfig = import ./nix/public-proxy.nix;
      nixosModules.default = import ./nix/service.nix { source = self; };
      formatter = forAllSystems ({ pkgs }: pkgs.nixfmt);

      packages = forAllSystems (
        { pkgs }: {
          default = import ./nix/launcher.nix { inherit pkgs; };
        }
      );
      apps = forAllSystems (
        { pkgs }: {
          default = {
            type = "app";
            program = pkgs.lib.getExe self.packages.${pkgs.stdenv.hostPlatform.system}.default;
          };
        }
      );

      # Development environment output
      devShells = forAllSystems (
        { pkgs }:
        {
          default =
            let
              runtime = import ./nix/runtime.nix { inherit pkgs; };
            in
            pkgs.mkShell {
              shellHook = ''
                load_agenix_secret() {
                  if ! printenv "$1" >/dev/null 2>&1 && [ -r "$2" ]; then
                    secret_value=$(cat "$2")
                    export "$1=$secret_value"
                    unset secret_value
                  fi
                }

                if command -v ip >/dev/null 2>&1 && ip link show tailscale0 >/dev/null 2>&1; then
                  export TSIP=$(ip -o -4 addr show tailscale0 | awk '{ split($4, ip_addr, "/"); print ip_addr[1] }')
                fi

                load_agenix_secret HF_TOKEN /run/agenix/hf-token
                if [ -z "$AZURE_API_VERSION" ]; then
                  export AZURE_API_VERSION="2024-12-01-preview"
                fi
                if [ -z "$AZURE_API_BASE" ]; then
                  export AZURE_API_BASE="https://admin-m6rf6uyv-eastus2.services.ai.azure.com/"
                fi
                if [ -z "$AZURE_API_KEY" ]; then
                  load_agenix_secret AZURE_API_KEY /run/agenix/azure-ai-eastus2-key
                fi
                if [ -z "$AZURE_AI_API_KEY" ]; then
                  load_agenix_secret AZURE_AI_API_KEY /run/agenix/azure-ai-eastus2-key
                fi
                if [ -z "$AZURE_AI_API_BASE" ]; then
                  export AZURE_AI_API_BASE="https://admin-m6rf6uyv-eastus2.services.ai.azure.com/"
                fi
                if [ -z "$OPENAI_API_VERSION" ]; then
                  export OPENAI_API_VERSION="2024-02-15-preview"
                fi
                if [ -z "$OPENAI_API_KEY" ]; then
                  load_agenix_secret OPENAI_API_KEY /run/agenix/openai-api
                fi
                load_agenix_secret AWS_ACCESS_KEY_ID /run/agenix/aws-access-key-id
                load_agenix_secret AWS_SECRET_ACCESS_KEY /run/agenix/aws-secret-access-key
                if [ -z "$AWS_REGION_NAME" ]; then
                  export AWS_REGION_NAME="us-west-2"
                fi
                load_agenix_secret OPENROUTER_API_KEY /run/agenix/openrouter-key
                load_agenix_secret YOUTUBE_API_KEY /run/agenix/youtube-data-api
                ${runtime.shellEnvironment}
              '';
              # The Nix packages provided in the environment
              packages = runtime.packages;
            };
        }
      );
    };
}
