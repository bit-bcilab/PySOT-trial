
from configs.DataPath import get_base_name
from .vot import VOTDataset, VOTLTDataset
from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .nfs import NFSDataset
from .trackingnet import TrackingNetDataset
from .got10k import GOT10kDataset
from .itb import ITBDataset

DATASETS = {'OTB50': OTBDataset,
            'OTB100': OTBDataset,
            'LaSOT': LaSOTDataset,

            'GOT-10k': GOT10kDataset,
            'VOT2016': VOTDataset,
            'VOT2018': VOTDataset,
            'VOT2019': VOTDataset,
            'VOT2020': OTBDataset,
            'ITB': ITBDataset,
            'NFS30': NFSDataset,
            'NFS240': NFSDataset,
            'VOT2018-LT': VOTLTDataset,
            'TrackingNet': TrackingNetDataset,

            'UAV123': UAVDataset,
            'UAV20L': UAVDataset,
            'UAV10fps': UAVDataset,
            'UAVDT': OTBDataset,
            'DTB70': OTBDataset,

            'VOT2017-TIR': VOTDataset,
            'LSOTB-TIR': OTBDataset,
            'PTB-TIR': OTBDataset}


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        # name = kwargs['name']
        # base_name = get_base_name(name)
        # dataset = DATASETS[base_name](**kwargs)
        # dataset.base_name = base_name
        # dataset.save = 'derive'
        try:
            name = kwargs['name']
            base_name = get_base_name(name)
            dataset = DATASETS[base_name](**kwargs)
            dataset.base_name = base_name
            dataset.save = 'derive'
        except:
            raise Exception("unknow dataset {}".format(kwargs['name']))

        return dataset
