from setuptools import find_packages, setup

# Retrieve description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="fastai_category_encoders",
    version="0.0.3",
    url="https://github.com/kireygroup/fastai-category-encoders",
    license="MIT",
    author="Riccardo Sayn",
    author_email="riccardo.sayn@kireygroup.com",
    description="Category encoders integrated with Fast.ai ",
    packages=find_packages(),
    install_requires=["fastai>=2.1.5", " category-encoders"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    keywords=["encoding", "fastai", "pytorch", "python", "pandas", "data"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
