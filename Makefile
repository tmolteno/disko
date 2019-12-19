test:
	- rm *.npz
	python3 setup.py test
	
develop:
	sudo python3 setup.py develop

install:
	sudo apt install python3-casacore python3-numpy python3-matplotlib python3-healpy python3-astropy python3-h5py python3-scipy python3-svgwrite

lint:
	pylint --extension-pkg-whitelist=numpy --ignored-modules=numpy --extension-pkg-whitelist=astropy disko

test2:
	#python3 -m unittest  disko.tests.test_gridless.TestGridless.test_from_pos
#	python3 -m unittest  disko.tests.test_subsphere
	python3 setup.py test -s disko.tests.test_disko_ms


ngc1194:
	disko --fov 0.3 --ms ../tart2ms/docker/NGC1194.split.ms --SVG --arcmin 0.3 --tikhonov --nvis 3000

cygnus:
	/usr/bin/time -v disko --fov 0.3 --ms /home/tim/astro/cyg2052.ms --SVG --arcmin 0.2 --tikhonov --nvis 3000 --alpha 0.0025 --title 'cygnus'
#Mem 4G for 22260 x 3000
#Mem 16G 90000 x 3000
#Mem 160G 90000 x 30000
#Mem 1600G 90000 x 300000
# FOV 0.3 0.89 alpha=0.1
# FOV 0.3 0.98 alpha=0.01
# FOV 0.3 0.993 alpha=0.005
# FOV 0.3 0.997 alpha=0.0025
# FOV 0.3 __ alpha=0.001  # Doesn't converge

#        Maximum resident set size (kbytes): 3956904
#         Maximum resident set size (kbytes): 2903484


cygnus_center:
	disko --fov 0.02 --ms ../tart2ms/docker/cyg2052.ms --SVG --arcmin 0.012 --tikhonov --nvis 2000 --alpha 0.1 --title 'cygnus_center'
	
tart:
	/usr/bin/time -v disko --fov 155 --ms ../tart2ms/test.ms --SVG --arcmin=90 --alpha=0.01 --tikhonov

# Memory 4800x276 456212 
#	 19328x276 640932 ->  458364 for 
test_upload:
	rm -rf tart2ms.egg-info dist
	python3 setup.py sdist
	twine upload --repository testpypi dist/*

upload:
	rm -rf tart2ms.egg-info dist
	python3 setup.py sdist
	twine upload --repository pypi dist/*
