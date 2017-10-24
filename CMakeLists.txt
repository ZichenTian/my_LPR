#这是对CMake工具最低版本要求，这里我们要检查下我们的CMake工具的版本信息，我们可以使用命令“cmake --version”查看
cmake_minimum_required(VERSION 2.6)
project(LPR)#这是建立一个工程项目，括号里面时工程名,工程名我们可以任意给，最后程序编译出来的可执行文件就是这个名字
FIND_PACKAGE(OpenCV  REQUIRED )#用Cmake查找opencv包
set(CMAKE_CXX_STANDARD 11)#用C++11标准库编译
set(SOURCE_FILES main.cpp)
add_executable(LPR ${SOURCE_FILES})#这里括号里面的两个参数分别是工程项目名和我们要编译文件名的意思，记住中间一空格键隔开
TARGET_LINK_LIBRARIES(LPR ${OpenCV_LIBS})#这是我们链接到OpenCV库的环节，我们只要更改前面第一个参数位我们的工程项目名即可
