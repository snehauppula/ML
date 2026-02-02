from setuptools import setup, find_packages

def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    # Remove editable installs and empty lines
    requirements = [req for req in requirements if req and not req.startswith('-e')]
    return requirements

setup(
    name='my_package',
    version='0.1.0',
    author='Uppula Sneha',
    author_email='snehauppula23@gmail.com',
    description='A sample Python package',
    packages=find_packages(),
    install_requires=get_requirements(),
    entry_points={
        'console_scripts': [
            'my_command=my_package.module:main_function',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
    