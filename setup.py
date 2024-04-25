from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT ='-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return required libraries to be installed as a List
    '''
    requirements=[]
    with open(file_path)as file_obj:
        requirements=file_obj.readline()
        requirements=[req.replace("\n","")for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
setup (         
name='MLProject',
version='0.0.1',
author='vanivasudevan',
author_email='vasudevan.vani@gmail.com',
packages=find_packages(),
install_requires = get_requirements('requirements.txt')
)