cmake_minimum_required(VERSION 3.16)
project(snapkitty_mlc C)

set(CMAKE_C_STANDARD 11)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O2)
    if(UNIX AND NOT APPLE)
        add_compile_options(-D_DEFAULT_SOURCE)
    endif()
endif()

# ── Library ────────────────────────────────────────────────────────
add_library(sk_mlc STATIC
    src/sk_arena.c
    src/sk_random.c
    src/sk_matrix.c
    src/sk_model.c
)
target_include_directories(sk_mlc PUBLIC include)
if(UNIX)
    target_link_libraries(sk_mlc PUBLIC m)  # -lm for math.h
endif()

# ── MNIST example ──────────────────────────────────────────────────
add_executable(mnist examples/mnist.c)
target_link_libraries(mnist PRIVATE sk_mlc)

# ── Install ────────────────────────────────────────────────────────
install(TARGETS sk_mlc ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/sk_mlc)
