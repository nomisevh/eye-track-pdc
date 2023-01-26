from argparse import ArgumentParser
from enum import Enum

from torch.cuda import is_available as cuda_is_available

from src.dataset import KIDataset
from src.entrypoint import handle_rocket, handle_inception_time, handle_inception_former
from src.processor import Leif


class Model(Enum):
    ROCKET = 'rocket'
    INCEPTION_TIME = 'inception_time'
    INCEPTION_FORMER = 'inception_former'


def main(args):
    use_train_data = args.data_partition == 'train'
    group_by_trail = args.model == Model.INCEPTION_FORMER

    # TODO load config yaml for processor
    processor_config = ...
    if args.processor_config == 'leif':
        preprocessor = Leif(train=use_train_data, config=processor_config)
    else:
        raise NotImplementedError

    if args.data_source == 'ki':
        # Todo use group by trail
        ds = KIDataset(data_processor=preprocessor, train=use_train_data)
    else:
        raise NotImplementedError

    kwargs = {'dataset': ds,
              'intention': args.intention,
              'model_config': args.model_config,
              'train_config': args.train_config,
              'seed': args.seed}

    if args.model == Model.ROCKET:
        handler = handle_rocket
    elif args.model == Model.INCEPTION_TIME:
        kwargs['device'] = args.device
        handler = handle_inception_time
    elif args.model == Model.INCEPTION_FORMER:
        kwargs['device'] = args.device
        handler = handle_inception_former
    else:
        raise NotImplementedError

    handler(**kwargs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='rocket')
    parser.add_argument('--intention', type=str, default='train')
    parser.add_argument('--data_partition', type=str, default='train')
    parser.add_argument('--data_source', type=str, default='ki')
    parser.add_argument('--processor_config', type=str, default='leif')
    parser.add_argument('--model_config', type=str, default='default')
    parser.add_argument('--train_config', type=str, default='default')
    parser.add_argument('--device', default='cuda' if cuda_is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    arguments = parser.parse_args()

    main(arguments)
