from conan import ConanFile
from conan.tools.cmake import cmake_layout


class RTODRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("opencv/4.10.0")
        self.requires("nlohmann_json/3.11.3")

    def layout(self):
        cmake_layout(self)
