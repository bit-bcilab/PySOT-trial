

from pysot.datasets.siamrpnpp_dataset import SiamRPNppDataset
from pysot.datasets.siamban_dataset import SiamBANDataset
from pysot.datasets.siamcar_dataset import SiamCARDataset
from pysot.datasets.siamgat_dataset import SiamGATDataset


DATASETS = {
            'SiamRPNpp': SiamRPNppDataset,
            'SiamMask': SiamRPNppDataset,
            'SiamAtt': SiamRPNppDataset,
            'SiamBAN': SiamBANDataset,
            'SiamCAR': SiamCARDataset,
            'SiamGAT': SiamGATDataset,
           }


def build_dataset(name, cfg):
    return DATASETS[name](cfg)
