# pylib_template
# Makefile to build and install the EXAMPLE python module and commands.
# Target: install or empty (->install), clean
# mikkio 2019

#############################
# change the following vars to your environment.
#pylib = ~/nlp/tools/pythonlib
pylib = ~/mylocal/pythonlib
bindir = ~/mylocal/scripts/mywork
# Warning: The make ignores just only white spaces immediately after an equal (=) symbol.

#############################
# packeage and command install
VPATH = src bin
.PHONY: install
pylibname = scoring.py
pylibfiles = $(pylib)/$(pylibname)
commandname = score
binfiles = $(addprefix $(bindir)/,$(commandname))

install: $(pylibfiles) $(binfiles)
#install: $(binfiles)

# the pattern rule for install *.{so,py,sh,R} files
$(pylibfile)/%.py: $(pylibname)
	cp $< $@

#$(bindir)/%.py: src/%.py
$(bindir)/%: bin/%
	cp $< $@

$(bindir)/%.sh: %.sh
	cp $< $@

$(bindir)/%.R: %.R
	cp $< $@

#############################
# clean
.PHONY: clean

clean:
	rm -f $(pylibfiles)
	rm -f $(binfiles)
