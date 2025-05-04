
from typing import List

def get_requirements(file_path:str)->List[str]:
    print('file path is----->', file_path)
    requiremets=[]
    with open(file_path,'r')  as data:
            requiremets=data.readlines()
            requiremets=(req.replace("\n","")for req in requiremets)


get_requirements('requirements.txt')