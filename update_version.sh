#!/bin/bash

# Set the new version here:
NEW_VERSION="1.0.6"

# Update version in pyproject.toml (under [project] section)
sed -i -E "s/^(version *= *\").*(\" *)$/\1${NEW_VERSION}\2/" pyproject.toml

# Update version fallback in multivarious/__init__.py
# This assumes line like: __version__ = "1.0.1" or __version__ = '1.0.1'
sed -i -E "s/^(    __version__ *= *[\"'])([^\"']*)([\"'])/\1${NEW_VERSION}\3/" multivarious/__init__.py

echo "Updated version to $NEW_VERSION in pyproject.toml and multivarious/__init__.py"

