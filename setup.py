from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='illumination_cuda',
    version='0.0',
    packages=find_packages(),
    license='MIT License',
    ext_modules=[
        CUDAExtension(
            name='illuminationCUDA',
            sources=[
                'src/illuminationCUDA.cpp',
                'src/illuminationCUDAKernel.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)