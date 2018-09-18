all:
    # install pycocotools locally
	pip install -r requirements.txt
	python setup.py build_ext --inplace
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	python setup.py build_ext install
	rm -rf build