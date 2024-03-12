from setuptools import setup, find_packages

setup(
    name="easyjailbreak",
    version="0.1.2",
    description="Easy Jailbreak toolkit",
    author="ZenithX",
    author_email="waltersumbon@gmail.com",
    url="https://github.com/EasyJailbreak/EasyJailbreak",
    packages=find_packages(include=('easyjailbreak*',)),
    install_requires=[
        'transformers>=4.34.0',
        'protobuf',
        'sentencepiece',
        'datasets',
        'torch>=2.0',
        'openai>=1.0.0',
        'numpy',
        'pandas',
        'accelerate',
        'fschat',
        'jsonlines',
        'einops',
        'nltk',
        'transformers_stream_generator',
    ],
    python_requires=">=3.9",
    keywords=['jailbreak', 
              'llm security', 
              'llm safety benchmark',
              'large language model',
              'jailbreak framework',
              'jailbreak prompt',
              'discrete optimization'
             ],
    license='GNU General Public License v3.0',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3"
    ]
)
