# CatBoost Wrapper for Go

Simple wrapper
of [CatBoost C library](https://tech.yandex.com/catboost/doc/dg/concepts/c-plus-plus-api_dynamic-c-pluplus-wrapper-docpage/)
for prediction

## Installation

CatBoost library is assumed to be installed, and all its includes and library files are assumed to be found in
corresponding paths. One way to do it is just download pre-built files:

```bash
sudo download-catboost.sh
```

If everything above is properly configured then a simple `go get` command will do the trick:

```bash
go get -u github.com/zhaochenxiao90/catboost-go
```
