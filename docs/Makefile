# Disable numba, so jitted functions are nicely documented.
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=$(HOME)

# Command line options.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = emg3d
SOURCEDIR     = .
BUILDDIR      = _build

# Will also be triggered if "make" is provided without argument.
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
