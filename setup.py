

from setuptools import find_packages,setup
from typing import List

requirement_lst :List[str] = []
def get_requirements()->List[str]:
    try:
        with open('requirements.txt','r')as file:
            lines = file.readlines()
            
            for line in lines:
                requirement = line.strip()

                if requirement and requirement!= '-e.':
                    requirement_lst.append(requirement)
    except FileExistsError:
        print("requirements.txt file not found")

    return requirement_lst 

print(get_requirements())

setup(
    name="network_security",  # Your package name
    version="0.0.1",
    author="Kareem Al Chorbai",  # Your name
    author_email="your.email@example.com",  # Your email
    description="A network security analysis package",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/network-security",  
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
    },
)