from setuptools import setup, find_packages

# Retrieve description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

version = "<project-version>"
if version == "<project-version>":
    print(
        "You should customize project variables inside setup.py before trying to build this project."
    )
    return

setup(
    name="fastai_category_encoders",
    version=version,
    url="<project-url>",
    license="<license>",
    author="<author>",
    author_email="<author-email>",
    description="<project-description>",
    packages=find_packages(),
    install_requires=[["fastai", " fastai2", " category-encoders"]],
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
)
