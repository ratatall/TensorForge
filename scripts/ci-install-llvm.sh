#!/usr/bin/env bash
# Ubuntu 24.04 GitHub runner setup; not used by local build.sh.
set -euo pipefail
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_CODENAME" == noble ]] || { echo 'Expected Ubuntu 24.04 (noble)' >&2; exit 1; }
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg cmake ninja-build python3
key_file="$(mktemp)"
trap 'rm -f "$key_file"' EXIT
curl --fail --silent --show-error --retry 3 https://apt.llvm.org/llvm-snapshot.gpg.key -o "$key_file"
gpg --batch --yes --dearmor --output "${key_file}.gpg" "$key_file"
sudo install -m 644 "${key_file}.gpg" /usr/share/keyrings/tensorforge-llvm.gpg
rm -f "${key_file}.gpg"
# LLVM 23 is in the unversioned development channel until its release branch
# exists. Prefer the release branch once available; never select a different major.
llvm_suite=llvm-toolchain-noble-23
if ! curl --fail --silent --show-error --retry 3 "https://apt.llvm.org/noble/dists/${llvm_suite}/Release" -o /dev/null; then
  llvm_suite=llvm-toolchain-noble
fi
printf 'deb [signed-by=/usr/share/keyrings/tensorforge-llvm.gpg] https://apt.llvm.org/noble/ %s main\n' "$llvm_suite" | sudo tee /etc/apt/sources.list.d/tensorforge-llvm.list
sudo apt-get update
sudo apt-get install -y clang-23 llvm-23-dev libclang-rt-23-dev libzstd-dev libedit-dev libffi-dev zlib1g-dev libxml2-dev
[[ "$(llvm-config-23 --version)" == 23.* ]]
