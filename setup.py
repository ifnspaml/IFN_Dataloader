from distutils.core import setup

setup(
    name='dataloader',
    version='0.1',
    packages=['dataloader/data_preprocessing','dataloader/definitions','dataloader/eval','dataloader/file_io','dataloader/pt_data_loader',],
    long_description=open('README.md').read(),
)