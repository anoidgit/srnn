CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)

FILE(GLOB luasrc *.lua)

ADD_TORCH_PACKAGE(srnn "" "${luasrc}")
INSTALL(DIRECTORY "units" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/srnn")
