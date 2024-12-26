# Minimal makefile for Sphinx documentation
#

<<<<<<< HEAD
<<<<<<< HEAD
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
=======
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
>>>>>>> parent of 1fb22fd (update docs file)
SOURCEDIR     = .
BUILDDIR      = _build
=======
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
>>>>>>> parent of e0ffd02 (Update docs)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm -rf $(BUILDDIR)/*
	@rm -rf $(SOURCEDIR)/api/_autosummary