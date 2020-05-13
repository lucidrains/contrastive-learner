from setuptools import setup, find_packages

setup(
  name = 'contrastive_learner',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'Self-supervised contrastive learning made simple',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/contrastive-learner',
  keywords = ['self-supervised learning', 'artificial intelligence'],
  install_requires=[
      'torch',
      'kornia'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)