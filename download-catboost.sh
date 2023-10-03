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

catboost_version='v1.2.2'

rm /usr/local/include/c_api.h
rm /usr/local/include/model_calcer_wrapper.h
rm /usr/local/include/wrapped_calcer.h

if [ "${machine}" == 'Linux' ]; then
  wget -nv -P /usr/local/include -F https://raw.githubusercontent.com/catboost/catboost/$catboost_version/catboost/libs/model_interface/c_api.h
  wget -nv -P /usr/local/include -F https://github.com/catboost/catboost/raw/$catboost_version/catboost/libs/model_interface/model_calcer_wrapper.h
  wget -nv -P /usr/local/include -F https://github.com/catboost/catboost/raw/$catboost_version/catboost/libs/model_interface/wrapped_calcer.h
  rm /usr/local/lib/libcatboostmodel.so*
  wget -nv -P /usr/local/lib -F https://github.com/catboost/catboost/releases/download/$catboost_version/libcatboostmodel.so
  ln -fs /usr/local/lib/libcatboostmodel.so /usr/local/lib/libcatboostmodel.so.1
  ldconfig
  echo "Done"
elif [ "${machine}" == 'Mac' ]; then
  wget -nv -P /usr/local/include -F https://raw.githubusercontent.com/catboost/catboost/$catboost_version/catboost/libs/model_interface/c_api.h
  wget -nv -P /usr/local/include -F https://github.com/catboost/catboost/raw/$catboost_version/catboost/libs/model_interface/model_calcer_wrapper.h
  wget -nv -P /usr/local/include -F https://github.com/catboost/catboost/raw/$catboost_version/catboost/libs/model_interface/wrapped_calcer.h
  rm /usr/local/lib/libcatboostmodel.dylib*
  wget -nv -P /usr/local/lib -F https://github.com/catboost/catboost/releases/download/$catboost_version/libcatboostmodel.dylib
  ln -fs /usr/local/lib/libcatboostmodel.dylib /usr/local/lib/libcatboostmodel.dylib.1
  chmod +x /usr/local/lib/libcatboostmodel.dylib*
  echo "Done"
  # if you see error Reason: no LC_RPATH load command; then you need to add /usr/local/lib to DYLD_LIBRARY_PATH
  # export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
else
  echo "there is no pre-built files for ${machine} now. You need to run build catboost manually"
fi
