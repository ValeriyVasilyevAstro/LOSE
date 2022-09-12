activate_venv=. venv/bin/activate
python=python3
pip=$(python) -m pip
virtualenv=venv

init_local: setup_local install_deps_local

setup_local:
	$(pip) install --upgrade pip
	$(pip) install virtualenv
	$(python) -m virtualenv venv


install_deps_local:
	$(activate_venv) &&	$(pip) install -r requirements.txt