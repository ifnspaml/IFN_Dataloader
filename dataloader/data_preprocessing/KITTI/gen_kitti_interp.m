ground_path = 'C:\Users\Klingner\Desktop\Datasets\KITTI';
sparse_depth_fold = 'Depth';
interp_depth_fold = 'Depth_processed';
img_fold = 'Raw_data';
sparse_path = fullfile(ground_path, sparse_depth_fold);
interp_path = fullfile(ground_path, interp_depth_fold);
img_path = fullfile(ground_path, img_fold);
FileList = dir(fullfile(sparse_path, '**', '*.png'));
array_dim = size(FileList);
num_imgs = array_dim(1);

parfor i = 1:num_imgs
    sparse_depth = depth_read(fullfile(FileList(i).folder, FileList(i).name));

    splitted_paths = strsplit(FileList(i).folder, sparse_depth_fold);
    end_path = splitted_paths{2};

    img_file_path = fullfile(img_path, end_path, FileList(i).name);
    rgb_img = double(imread(img_file_path))/255.;

    interp_file_dir = fullfile(interp_path, end_path);
    interp_file_path = fullfile(interp_path, end_path, FileList(i).name);
    if ~isdir(interp_file_dir)
        mkdir(interp_file_dir)
    end
    depth_interp = fill_depth_colorization(rgb_img, sparse_depth, 1.0);
    depth_interp = uint16(depth_interp*256);

    sparse_depth = uint16(sparse_depth*256);
    imwrite(depth_interp, 'depth_interp.png')
    imwrite(sparse_depth, 'depth_gt.png')
    imwrite(rgb_img, 'depth_img.png')
    imwrite(depth_interp, interp_file_path);
    disp(i)
end

sparse_depth_fold = 'Depth_improved';
interp_depth_fold = 'Depth_processed_improved';
img_fold = 'Raw_data';
sparse_path = fullfile(ground_path, sparse_depth_fold);
interp_path = fullfile(ground_path, interp_depth_fold);
img_path = fullfile(ground_path, img_fold);
FileList = dir(fullfile(sparse_path, '**', '*.png'));
array_dim = size(FileList);
num_imgs = array_dim(1);

parfor i = 1:num_imgs
    sparse_depth = depth_read(fullfile(FileList(i).folder, FileList(i).name));

    splitted_paths = strsplit(FileList(i).folder, sparse_depth_fold);
    end_path = splitted_paths{2};

    img_file_path = fullfile(img_path, end_path, FileList(i).name);
    rgb_img = double(imread(img_file_path))/255.;

    interp_file_dir = fullfile(interp_path, end_path);
    interp_file_path = fullfile(interp_path, end_path, FileList(i).name);
    if ~isdir(interp_file_dir)
        mkdir(interp_file_dir)
    end
    depth_interp = fill_depth_colorization(rgb_img, sparse_depth, 1.0);
    depth_interp = uint16(depth_interp*256);

    sparse_depth = uint16(sparse_depth*256);
    imwrite(depth_interp, 'depth_interp.png')
    imwrite(sparse_depth, 'depth_gt.png')
    imwrite(rgb_img, 'depth_img.png')
    imwrite(depth_interp, interp_file_path);
    disp(i)
end


