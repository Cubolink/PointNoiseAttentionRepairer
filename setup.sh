# installing with torch
conda create -n pointattn2 python=3.10 pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# installing torch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d


# installing pytorch-scatter. Please make sure torch is not downgraded to CPU.
conda install pytorch-scatter -c pyg  # TRY THIS FIRST. It shouldn't downgrade torch, for me it didnt at least (But it did when I was trying pytorch 2.x)
# conda install pytorch-scatter=2.1.1 -c pyg  # USE THIS ONE ONLY IF THE OTHER DOWNGRADE THE TORCH'S CUDA VERSION (You will have to check manually with torch.version.cuda)
# conda install pytorch_scatter -c conda-forge  # USE THIS ONE ONLY IF THE OTHER TRIED TO DOWNGRADE TORCH.


# installing other dependencies (1)
conda install trimesh -c conda-forge
conda install tensorboardx
conda install pandas scikit-image scipy
conda install pyembree -c conda-forge
conda install cython imageio numpy numpy-base matplotlib matplotlib-base pillow pytest pyyaml tqdm


# installing other dependencies (2)
conda install h5py
conda install plyfile -c conda-forge
conda install pyyaml


# installing other dependencies (3)
conda install tensorpack -c hcc


# installing other dependencies (4)
conda install transforms3d -c conda-forge
conda install munch -c conda-forge
conda install scikit-learn


# installing open3d
python -m pip install open3d==0.18.0  # an outdated version in conda caused me a problem with sklearn, that derived in failing installing the mmdet3d

# stuff originally compiled from convolutional_occupancy_networks, but here installed with pip/conda
conda install pykdtree -c conda-forge  # may take some time
conda install pymcubes -c conda-forge
# libmesh seems to be uneeded, because it's used in evaluation
# libmise is used in generation, so it's NEEDED. I will install it later tho.
# libsimplify is not used in my experiments, but it is in generation
# libvoxelide is irrelevant, it's in generation but I don't handle voxels


# installing tensorflow
python -m pip install tensorflow  # this downgraded h5py to 3.11


# installing mim's related stuff
python -m pip install openmim
python -m mim install mmcv mmdet
python -m mim install mmsegmentation

python -m pip install mmdet3d  # works now, with open3d 0.18


# then, compile the utils as the README.md says
cd utils/ChamferDistancePytorch/chamfer3D || exit
python setup.py install
cd ../../..