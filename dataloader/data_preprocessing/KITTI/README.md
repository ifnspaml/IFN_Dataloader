This is a guide on how to create the KITTI dataset from scratch:

KITTI data is now downloaded and processed:

python3 download_kitti.py

This will download all necessary archives, bring them to a reasonable 
folder structure and generate the depth from the point clouds

Postprocessing the depth maps with the Code from the NYU depth toolbox

matlab -nodisplay -nosplash -nodesktop -r "run('gen_kitti_interp.m');exit;"

Will generate interpolations for the sparse depth maps with the colorization method 
of Levin et al., the types of interpolation have to be realized inside the data loader




