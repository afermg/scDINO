{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    pynng-flake.url = "github:afermg/pynng";
    pynng-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      with pkgs;
      rec {
        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              packages.nahual
              pp.torch
              pp.torchvision
              pp.numpy
              pp.pillow
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/scdino.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        packages = {
          nahual = pkgs.python3.pkgs.callPackage ./nix/nahual.nix {
            pynng = inputs.pynng-flake.packages.${system}.pynng;
          };
        };

        devShells = {
          default =
            let
              # `timm` is only referenced by a docstring in
              # `pyscripts/vision_transformer.py`; skip it to avoid the lengthy
              # timm test suite when entering the dev shell on nixos-unstable.
              python_with_pkgs = python3.withPackages (pp: [
                packages.nahual
                pp.torch
                pp.torchvision
                pp.tifffile
                pp.scikit-image
                pp.scikit-learn
                pp.numpy
                pp.pyyaml
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
              ];
              shellHook = ''
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
