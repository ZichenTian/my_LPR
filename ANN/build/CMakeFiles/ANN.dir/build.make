# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tzc/Learning/OpenCV/ANN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tzc/Learning/OpenCV/ANN/build

# Include any dependencies generated for this target.
include CMakeFiles/ANN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ANN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ANN.dir/flags.make

CMakeFiles/ANN.dir/main.cpp.o: CMakeFiles/ANN.dir/flags.make
CMakeFiles/ANN.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tzc/Learning/OpenCV/ANN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ANN.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ANN.dir/main.cpp.o -c /home/tzc/Learning/OpenCV/ANN/main.cpp

CMakeFiles/ANN.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ANN.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tzc/Learning/OpenCV/ANN/main.cpp > CMakeFiles/ANN.dir/main.cpp.i

CMakeFiles/ANN.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ANN.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tzc/Learning/OpenCV/ANN/main.cpp -o CMakeFiles/ANN.dir/main.cpp.s

CMakeFiles/ANN.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/ANN.dir/main.cpp.o.requires

CMakeFiles/ANN.dir/main.cpp.o.provides: CMakeFiles/ANN.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/ANN.dir/build.make CMakeFiles/ANN.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/ANN.dir/main.cpp.o.provides

CMakeFiles/ANN.dir/main.cpp.o.provides.build: CMakeFiles/ANN.dir/main.cpp.o


# Object files for target ANN
ANN_OBJECTS = \
"CMakeFiles/ANN.dir/main.cpp.o"

# External object files for target ANN
ANN_EXTERNAL_OBJECTS =

ANN: CMakeFiles/ANN.dir/main.cpp.o
ANN: CMakeFiles/ANN.dir/build.make
ANN: /usr/lib64/libopencv_videostab.so.3.1.0
ANN: /usr/lib64/libopencv_superres.so.3.1.0
ANN: /usr/lib64/libopencv_stitching.so.3.1.0
ANN: /usr/lib64/libopencv_shape.so.3.1.0
ANN: /usr/lib64/libopencv_photo.so.3.1.0
ANN: /usr/lib64/libopencv_face.so.3.1.0
ANN: /usr/lib64/libopencv_calib3d.so.3.1.0
ANN: /usr/lib64/libopencv_features2d.so.3.1.0
ANN: /usr/lib64/libopencv_flann.so.3.1.0
ANN: /usr/lib64/libopencv_video.so.3.1.0
ANN: /usr/lib64/libopencv_objdetect.so.3.1.0
ANN: /usr/lib64/libopencv_ml.so.3.1.0
ANN: /usr/lib64/libopencv_highgui.so.3.1.0
ANN: /usr/lib64/libopencv_videoio.so.3.1.0
ANN: /usr/lib64/libopencv_imgcodecs.so.3.1.0
ANN: /usr/lib64/libopencv_imgproc.so.3.1.0
ANN: /usr/lib64/libopencv_core.so.3.1.0
ANN: CMakeFiles/ANN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tzc/Learning/OpenCV/ANN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ANN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ANN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ANN.dir/build: ANN

.PHONY : CMakeFiles/ANN.dir/build

CMakeFiles/ANN.dir/requires: CMakeFiles/ANN.dir/main.cpp.o.requires

.PHONY : CMakeFiles/ANN.dir/requires

CMakeFiles/ANN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ANN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ANN.dir/clean

CMakeFiles/ANN.dir/depend:
	cd /home/tzc/Learning/OpenCV/ANN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tzc/Learning/OpenCV/ANN /home/tzc/Learning/OpenCV/ANN /home/tzc/Learning/OpenCV/ANN/build /home/tzc/Learning/OpenCV/ANN/build /home/tzc/Learning/OpenCV/ANN/build/CMakeFiles/ANN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ANN.dir/depend

