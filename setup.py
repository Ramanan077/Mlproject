from setuptools import setup, find_packages

def get_requirements(file_path):
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .') 

setup(
    name='mlproject',
    version='0.0.1',
    author='Ram',
    author_email='ramanan270727@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    ) 

