from datasets.SomethingSomething import STHV1
from datasets.SomethingSomething import STHV2


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['SomethingSomethingV1', 'SomethingSomethingV2']

    if opt.dataset == 'SomethingSomethingV1':
        training_data = STHV1(
            opt.video_path,
            opt.annotation_path,
            opt.proposal_path,
            'something-something-v1-train.txt',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            n_box_per_frame=opt.n_box_per_frame)
    elif opt.dataset == 'SomethingSomethingV2':
        training_data = STHV2(
            opt.video_path,
            opt.annotation_path,
            opt.proposal_path,
            'something-something-v2-train.txt',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            n_box_per_frame=opt.n_box_per_frame)
            
    return training_data

def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['SomethingSomethingV1', 'SomethingSomethingV2']

    if opt.dataset == 'SomethingSomethingV1':
        validation_data = STHV1(
            opt.video_path,
            opt.annotation_path,
            opt.proposal_path,
            'something-something-v1-validation.txt',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            n_box_per_frame=opt.n_box_per_frame)
    elif opt.dataset == 'SomethingSomethingV2':
        validation_data = STHV2(
            opt.video_path,
            opt.annotation_path,
            opt.proposal_path,
            'something-something-v2-validation.txt',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            n_box_per_frame=opt.n_box_per_frame)
    
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['SomethingSomethingV1', 'SomethingSomethingV2']

    if opt.dataset == 'SomethingSomethingV1':
        test_data = STHV1(
            opt.video_path,
            opt.annotation_path,
            opt.proposal_path,
            opt.test_subset,
            opt.n_test_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            n_box_per_frame=opt.n_box_per_frame)
    elif opt.dataset == 'SomethingSomethingV2':
        test_data = STHV2(
            opt.video_path,
            opt.annotation_path,
            opt.proposal_path,
            opt.test_subset,
            opt.n_test_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            n_box_per_frame=opt.n_box_per_frame)

    return test_data
