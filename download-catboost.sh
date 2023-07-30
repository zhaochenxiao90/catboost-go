#!/usr/bin/env bash

#download prebuilt catboost lib (if needed) from github

unameOut="$(uname -s)"
case "${unameOut}" in
Linux*) machine=Linux ;;
Darwin*) machine=Mac ;;
CYGWIN*) machine=Cygwin ;;
MINGW*) machine=MinGw ;;
*) machine="UNKNOWN:${unameOut}" ;;
esac

echo "Detecting OS: ${machine}"

#the current latest version of catboost
catboost_version='v1.2'

if [ "${machine}" == 'Linux' ]; then
  wget -nv -P /usr/local/include -F https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model_interface/c_api.h
  wget -nv -P /usr/local/include -F https://github.com/catboost/catboost/raw/master/catboost/libs/model_interface/model_calcer_wrapper.h
  wget -nv -P /usr/local/include -F https://github.com/catboost/catboost/raw/master/catboost/libs/model_interface/wrapped_calcer.h
  wget -nv -P /usr/local/lib -F https://github.com/catboost/catboost/releases/download/$catboost_version/libcatboostmodel.so
  ln -fs /usr/local/lib/libcatboostmodel.so /usr/local/lib/libcatboostmodel.so.1
  ldconfig
  echo "Done"
elif [ "${machine}" == 'Mac' ]; then
  # catboost support universal binary for mac; so we can use the same file for both x86 and arm
  wget -nv -P /usr/local/include -F https://raw.githubusercontent.com/catboost/catboost/master/catboost/libs/model_interface/c_api.h
  wget -nv -P /usr/local/include -F https://github.com/catboost/catboost/raw/master/catboost/libs/model_interface/model_calcer_wrapper.h
  wget -nv -P /usr/local/include -F https://github.com/catboost/catboost/raw/master/catboost/libs/model_interface/wrapped_calcer.h
  wget -nv -P /usr/local/lib -F https://github.com/catboost/catboost/releases/download/$catboost_version/libcatboostmodel.dylib
  chmod +x /usr/local/lib/libcatboostmodel.dylib
  ln -fs /usr/local/lib/libcatboostmodel.dylib /usr/local/lib/libcatboostmodel.dylib.1
  echo "Done"
else
  echo "there is no pre-built files for ${machine} now. You need to run build catboost manually"
fi
