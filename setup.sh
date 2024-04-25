conda create -n pointattn python=3.8
conda activate pointattn
python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install h5py==3.6.0
python -m pip install matplotlib==3.4.3
python -m pip install munch==2.5.0
python -m pip install open3d==0.13.0
python -m pip install pyyaml

# then, compile the utils as the README.md says
cd utils/ChamferDistancePytorch/chamfer3D || exit
python setup.py install
cd ../../..

cd utils/mm3d_pn2  || exit
python setup.py build_ext --inplace
cd ../..

# The beginning of the end, we have to install things that for some reason are not in the requirements.txt
python -m pip install transforms3d
python -m pip install tensorpack
python -m pip install tensorflow

# At last, the troubling part
python -m pip install openmim
python -m pip install mmcv==0.6.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
python -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html

