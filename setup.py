from setuptools import setup, find_packages

setup(name='pairfinance-pkg',
      description='Package for Pair Finance email service',
      version='0.1.3',
      maintainer='Shahid',
      maintainer_email='skshahidur@gmail.com',
      license='Shahid',
      packages=find_packages(exclude=['unit_test', 'test_environment']),
      package_data={'pairfinance': ['model_files_v0.pkl']},
      zip_safe=False,
      # test_suite='test_model',
      python_requires='>=3',
      install_requires=['pandas==0.24.2',
                        'numpy',
                        'matplotlib==2.2.3',
                        'scikit-learn[alldeps]>=0.21.0',
                        'seaborn==0.9.0',
                        'xgboost==0.90',
                        'requests',
                        'glob3',
                        'pymysql']
      )



