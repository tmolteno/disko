ms:
	tart2ms --ms test.ms --clobber --hdf vis_2021-03-25_20_50_23.568474.hdf

# Synthetic TART measurement set generated with the tartball tool
# (https://github.com/tmolteno/tartball). Consumed by
# TestSequentialSyntheticMs in disko/tests/test_sequential.py: a known spot
# model is predicted onto an nside=16 healpix sphere with the 24-antenna
# layout, complex Gaussian noise (0.05) is added, and the 276 baselines are
# written as a single-timestamp measurement set.
TARTBALL_DIR ?= $(HOME)/tart-telescope/tartball

synth.ms:
	cd $(TARTBALL_DIR) && uv run -p 3.13 tartball \
		--model context/spotless_model_2026_06_18_09_57_45_UTC.json \
		--info context/info.json \
		--antenna-positions context/antenna_positions.json \
		--ms $(CURDIR)/synth.ms \
		--nside 16 --noise 0.05 --clobber
