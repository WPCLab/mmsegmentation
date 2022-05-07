from mmseg.datasets import MapillaryDataset

if __name__ == '__main__':
    data_root = '/nfs/volume-807-2/darrenwang/mapillary'
    img_dir = 'training/images'
    ann_dir = 'training/labels'
    dataset = MapillaryDataset(data_root=data_root, img_dir=img_dir, ann_dir=ann_dir)
    for step, data in enumrate(dataset):
        print(step, data.keys())