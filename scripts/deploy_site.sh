#!/bin/sh
# Publish site/ to GitHub Pages: push its contents to the gh-pages branch.
#   sh scripts/deploy_site.sh
set -e
cd "$(git rev-parse --show-toplevel)"
git subtree split --prefix site -b gh-pages-build >/dev/null
git push -f origin gh-pages-build:gh-pages
git branch -D gh-pages-build >/dev/null
