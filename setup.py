import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="soft_dtw",
    version="0.0.1",
    author='Sarthak Yadav',
    description="Soft-DTW and Soft-DTW Divergence in Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SarthakYadav/jax-soft-dtw-div",
    # package_dir={"": ""},
    packages=[
        "soft_dtw",
    ],
    python_requires=">=3.8"
)