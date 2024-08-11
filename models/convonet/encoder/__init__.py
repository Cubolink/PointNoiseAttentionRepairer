from models.convonet.encoder import (
    pointnet,
    # voxels,
    # pointnetpp
)


encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
}
