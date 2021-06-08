import io
from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf8') as f:
    reqs = f.read()

setup(
    name='densephrases',
    version='1.0',
    description='Learning Dense Representations of Phrases at Scale',
    long_description=readme,
    license=license,
    url='https://github.com/princeton-nlp/DensePhrases',
    keywords=['phrase', 'embedding', 'retrieval', 'nlp', 'open-domain', 'qa'],
    python_requires='>=3.7',
    install_requires=reqs.strip().split('\n'),
)
