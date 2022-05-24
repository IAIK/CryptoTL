cmake_minimum_required(VERSION 3.12)

project(HE_TL VERSION 4.0 LANGUAGES C CXX)

set(EXECUTABLE main)
# Executable will be in ../bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HE_TL_SOURCE_DIR}/bin)

include(CheckCCompilerFlag)
function(check_c_compiler_flag_and_add flag result)
  check_c_compiler_flag("${flag}" ${result})
  if(${result})
    add_compile_options("${flag}")
  endif()
endfunction()

check_c_compiler_flag_and_add(-pedantic CC_SUPPORTS_PEDANTIC)
check_c_compiler_flag_and_add(-Wall CC_SUPPORTS_WALL)
check_c_compiler_flag_and_add(-Wextra CC_SUPPORTS_WEXTRA)
check_c_compiler_flag_and_add(-Wshadow CC_SUPPORTS_WSHADOW)
check_c_compiler_flag_and_add(-Werror=vla CC_SUPPORTS_ERRORVLA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
endif()

OPTION(SERVER_ONLY "Perform Classification with the server network only" OFF)
OPTION(STORE_INPUTS "Stores the outputs of the lower layers" OFF)
OPTION(ACCURATE_PARAMS "Uses more accurate HE parameters" OFF)

if (SERVER_ONLY)
  add_compile_definitions("SERVER_ONLY")
endif()
if (STORE_INPUTS)
  add_compile_definitions("STORE_INPUTS")
endif()
if (ACCURATE_PARAMS)
  add_compile_definitions("ACCURATE_PARAMS")
endif()


add_executable(${EXECUTABLE} main.cpp)
target_sources(${EXECUTABLE}
  PRIVATE
    Client.cpp
    Server.cpp
)

# Import Microsoft SEAL
find_package(SEAL 4.0 REQUIRED
    # Optionaly providing a path so this can be built without installing SEAL
    PATHS ${HE_TL_SOURCE_DIR}/SEAL/build/cmake
)

target_link_libraries(${EXECUTABLE} SEAL::seal)

add_custom_target(run
  COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EXECUTABLE}
  DEPENDS ${EXECUTABLE}
)
