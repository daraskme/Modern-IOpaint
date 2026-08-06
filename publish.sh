#!/usr/bin/env bash
set -e

pushd ./web_app
rm -r -f dist
npm run build
popd
rm -r -f ./modern_iopaint/web_app
cp -r web_app/dist ./modern_iopaint/web_app

rm -r -f dist
python3 setup.py sdist bdist_wheel
