from data_provider.data_loader import Dataset_Lake
from torch.utils.data import DataLoader

data_dict = {
    'Lake': Dataset_Lake
}


def data_provider(args, flag, gt=None):
    Data = data_dict[args['dataset']]
    # timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args['batch_size']
        freq = args['freq']
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args['freq']
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args['batch_size']
        freq = args['freq']

    if gt is not None:
        root_path=args['gt_root_path']
        data_path=args['gt_source_filename']
    else:
        root_path=args['root_path']
        data_path=args['source_filename']
    
    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[args['seq_len'], args['label_len'], args['pred_len']],
        timeenc=1,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader
