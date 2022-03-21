TIME=/usr/bin/time -v

test:
	- rm *.npz
	pytest-3  # python3 setup.py test
	
develop:
	pip3 install -e .

install:
	sudo apt install python3-casacore python3-numpy python3-matplotlib python3-healpy python3-astropy python3-h5py python3-scipy python3-svgwrite python3-dask

lint:
	pylint --extension-pkg-whitelist=numpy --ignored-modules=numpy --extension-pkg-whitelist=astropy disko

test2:
	#python3 -m unittest  disko.tests.test_gridless.TestGridless.test_from_pos
	#	python3 -m unittest  disko.tests.test_subsphere
	pytest-3 -k test_multivariate

svd:
	rm -f *.npz
	${TIME} disko_svd  --file test_data/test_data.json  --nside 16

bayes:
	#rm -f *.npz
	${TIME} disko_bayes --fov 155 --ms test_data/test.ms  --mu --PNG --SVG --arcmin=90  --dir test_out --title 'bayes_tart' --sigma-v=0.15
#	${TIME} disko --fov 155 --ms ../tart2ms/test.ms --SVG --arcmin=120  --title 'tart' --tikhonov --alpha=0.01

ngc1194:
	disko --fov 0.3 --ms ../tart2ms/docker/NGC1194.split.ms --SVG --arcmin 0.3 --tikhonov --nvis 3000

adaptive:
	rm -f round*.vtk
	${TIME} disko --fov 0.05 --ms /home/tim/astro/cyg2052.ms --SVG --arcmin 0.07 --arcmax=0.1 --tikhonov --nvis 2000 --alpha 0.015 --title 'acygnus' --adaptive 50
	

# Requires memory_profiler pip3 install memory_profiler
cygnus:
	/usr/bin/time -v disko_bayes --fov 0.05 --ms /home/tim/astro/cyg2052.ms --SVG --mu --arcmin 0.025 --nvis 1500 --title 'cygnus'
	
dask:
	disko --fov 0.5 --ms /home/tim/astro/cyg2052.ms --SVG --arcmin 0.25 --tikhonov --nvis 2000 --dask

# Mem 2520  (0.1 arcmin)   1726672
# Mem 9940  (0.05 arcmin)  6080840 / 5858460
# Mem 39480 (0.025 arcmin)

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
	${TIME} disko --fov 155 --ms test_data/test.ms --SVG --arcmin=60 --arcmax=90 --alpha=0.0025 --tikhonov  --title 'tart'

sphere:
	disko --nside 64 --ms ../tart2ms/test.ms --SVG --PNG --PDF --SVG --show-sources --alpha=0.0025 --tikhonov  --title 'sphere'

mf:
	rm -f disko.log
	disko --fov 155 --ms test_data/test.ms --SVG --arcmin=90 --arcmax=190 --alpha=-0.56 --fista --matrix-free --title 'mf'

## 2000 0.0696
## 4000 0.0281
## 4000 0.035 # 3.546681e-02
## 16000
NV_CYG=16000 
#	1281930
mf_cyg:
	rm -f disko.log
	disko --fov 0.05 --ms ~/astro/cyg2052.ms --SVG --arcmin=0.01 --alpha=-0.0695 --nvis ${NV_CYG} --fista --matrix-free --title 'mf_cyg' --niter 150
	

profile:
	python3 -m cProfile -o disko.prof ./bin/disko --fov 155 --ms ../tart2ms/test.ms --SVG --arcmin=120 --alpha=0.25 --matrix-free --lsqr
	python3 prof.py

sequential: step1 step2

step1:
	disko_bayes --nside 24 --ms test_data/test.ms  --dir test_out --title 'bayes_tart' --sigma-v=0.15 --posterior post.h5

step2:
	disko_bayes --nside 24 --ms test_data/test.ms  --mu --pcf --var --PNG  --dir test_out --title 'bayes_tart_2' --sigma-v=0.15 --prior post.h5

h5:
	disko_bayes --hdf test_data/vis_2021-03-25_20_50_23.568474.hdf --nside 20 --sigma-v=0.15 --mu --pcf --var --PNG --title 'sequential' --dir=seq_out
	
# Memory 4800x276 456212 
#	 19328x276 640932 ->  458364 for 
test_upload:
	rm -rf disko.egg-info dist
	python3 setup.py sdist
	twine upload --repository testpypi dist/*

upload:
	rm -rf disko.egg-info dist
	python3 setup.py sdist
	twine upload --repository pypi dist/*
