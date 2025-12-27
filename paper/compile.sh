#!/bin/bash
# Compile the paper using the latexmk installed in the pinn-dev container
docker exec -w /workspaces/pykt-toolkit/paper/latex pinn-dev latexmk -pdf paper.tex
