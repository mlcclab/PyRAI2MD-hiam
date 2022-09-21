######################################################
#
# PyRAI2MD 2 module for building extension
#
# Author Jingbai Li
# Aug 30 2022
#
######################################################

import os
import shutil
import subprocess

def verify_ext():
    curr_path = os.getcwd()
    script_path = os.path.realpath(__file__)
    package_path = '/'.join(script_path.split('/')[: -3])
    setup_path = '%s/PyRAI2MD/Dynamics/Propagators/setup_fssh.py' % package_path

    os.chdir(package_path)

    print('\n PyRAI2MD: Updating required extension files\n')
    print('\n PyRAI2MD: Compiling C library for fssh.pyx\n')
    subprocess.run(['python3', setup_path, 'build_ext', '--inplace'])
    print('\n\n PyRAI2MD: Complete\n\n')

    shutil.rmtree('%s/build' % package_path)

    os.chdir(curr_path)

    return None




