test:
	- rm *.npz
	python3 setup.py test
	
develop:
	sudo python3 setup.py develop

install:
	sudo apt install python3-numpy python3-matplotlib python3-healpy python3-astropy python3-h5py python3-scipy python3-svgwrite

lint:
	pylint --extension-pkg-whitelist=numpy --ignored-modules=numpy --extension-pkg-whitelist=astropy gridless

test2:
	#python3 -m unittest  disko.tests.test_gridless.TestGridless.test_from_pos
#	python3 -m unittest  disko.tests.test_subsphere
	python3 setup.py test -s disko.tests.test_gridless.TestGridless.test_gamma_size

test_upload:
	rm -rf tart2ms.egg-info dist
	python3 setup.py sdist
	twine upload --repository testpypi dist/*

upload:
	rm -rf tart2ms.egg-info dist
	python3 setup.py sdist
	twine upload --repository pypi dist/*