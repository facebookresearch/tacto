import os
import nox

BASE = os.path.abspath(os.path.dirname(__file__))

def install_tacto(session):
    session.chdir(BASE)
    cmd = "pip install -e .".split(' ')
    session.run(*cmd)

@nox.session
def tests(session):
    install_tacto(session)
    session.install('pytest')
    session.run('pytest')
