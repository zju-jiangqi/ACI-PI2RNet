import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs


def create_lmdb_for_div2k():
    """Create lmdb files for DIVPano
    """
    # GT images
    folder_path = './datasets/DIVPano/Train/HQ'
    lmdb_path = './datasets/DIVPano_lmdb/DIVPano_Train_HR.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LR images
    folder_path = './datasets/DIVPano/Train/LQ'
    lmdb_path = './datasets/DIVPano_lmdb/DIVPano_Train_LR.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # GT images
    folder_path = './datasets/DIVPano/Val/HQ'
    lmdb_path = './datasets/DIVPano_lmdb/DIVPano_Val_HR.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # AS images
    folder_path = './datasets/DIVPano/Val/LQ'
    lmdb_path = './datasets/DIVPano_lmdb/DIVPano_Train_LR.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)






def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        help=("Options: 'DIV2K' "
              'You may need to modify the corresponding configurations in codes.'))
    args = parser.parse_args()
    dataset = args.dataset.lower()
    if dataset == 'div2k':
        create_lmdb_for_div2k()
    else:
        raise ValueError('Wrong dataset.')
