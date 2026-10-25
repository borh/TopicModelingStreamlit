{ pkgs }:
let
  lib = pkgs.lib;
  securityPatch = pkgs.fetchurl {
    url = "https://github.com/google/brotli/commit/0ed4268a4401abcb4472096340477cf4af7d1a2b.patch";
    hash = "sha256-gzyh1x7thb3jwhxjpu4xATh3Qbiqi5qCJRNAlKZ3thk=";
  };
  speelyBrotli = pkgs.brotli.overrideAttrs (old: {
    patches = (old.patches or [ ]) ++ [ securityPatch ];
  });
  speelyPkgs = pkgs // {
    brotli = speelyBrotli;
  };
  fixed = import ./brotli-fix.nix { pkgs = speelyPkgs; };
  upstreamPatches = speelyBrotli.patches;
  fixedPatches = fixed.brotli.patches;
in
{
  inherit (fixed) brotli;
  check =
    pkgs.runCommand "brotli-patch-phase-check"
      {
        nativeBuildInputs = [
          pkgs.patch
          pkgs.coreutils
          pkgs.gnugrep
        ];
      }
      ''
        set -eu
        cp -a ${fixed.brotli.src} red
        cp -a ${fixed.brotli.src} green
        chmod -R u+w red green
        upstream="${lib.concatStringsSep " " (map toString upstreamPatches)}"
        fixed="${lib.concatStringsSep " " (map toString fixedPatches)}"
        red_log=red.log
        set +e
        : > "$red_log"
        for patch_file in $upstream; do
          echo "RED applying $patch_file" >> "$red_log"
          patch -p1 -d red < "$patch_file" >> "$red_log" 2>&1
          rc=$?
          echo "RED_RC=$rc" >> "$red_log"
          if [ "$rc" -ne 0 ]; then
            break
          fi
        done
        set -e
        if ! grep -Eiq 'reversed|previously applied' "$red_log"; then
          echo "RED did not fail with reversed or previously applied" >&2
          cat "$red_log" >&2
          exit 1
        fi
        if ! grep -q '0ed4268a' "$red_log"; then
          echo "RED did not reach the 0ed4268a patch" >&2
          cat "$red_log" >&2
          exit 1
        fi
        for patch_file in $fixed; do
          patch -p1 -d green < "$patch_file"
        done
        touch $out
      '';
}
