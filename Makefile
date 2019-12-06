test:
	- rm *.npz
	python3 setup.py test
	
develop:
	sudo python3 setup.py develop

install:
	sudo apt install python3-numpy python3-matplotlib python3-healpy python3-astropy python3-h5py python3-scipy python3-svgwrite

lint:
	pylint --extension-pkg-whitelist=numpy --ignored-modules=numpy --extension-pkg-whitelist=astropy disko

test2:
	#python3 -m unittest  disko.tests.test_gridless.TestGridless.test_from_pos
#	python3 -m unittest  disko.tests.test_subsphere
	python3 setup.py test -s disko.tests.test_disko_ms

ngc1194:
	disko --fov 0.3 --ms ../tart2ms/docker/NGC1194.split.ms --SVG --arcmin 0.2 --tikhonov --nvis 3000

cygnus:
	disko --fov 0.3 --ms ../tart2ms/docker/cyg2052.ms --SVG --arcmin 0.2 --tikhonov --nvis 3000
	mv disko_2015_11_15_20_35_44_.svg cygnus_pointless.svg

test_upload:
	rm -rf tart2ms.egg-info dist
	python3 setup.py sdist
	twine upload --repository testpypi dist/*

upload:
	rm -rf tart2ms.egg-info dist
	python3 setup.py sdist
	twine upload --repository pypi dist/*
