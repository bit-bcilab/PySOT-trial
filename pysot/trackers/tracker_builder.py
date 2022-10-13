from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from pysot.trackers.siamrpn_tracker import SiamRPNTracker
from pysot.trackers.siammask_tracker import SiamMaskTracker
from pysot.trackers.siamatt_tracker import SiamAttTracker
from pysot.trackers.siamban_tracker import SiamBANTracker
from pysot.trackers.siamcar_tracker import SiamCARTracker
from pysot.trackers.siamcarm_tracker import SiamCARMTracker
from pysot.trackers.siamgat_tracker import SiamGATTracker
from pysot.trackers.transt_tracker import TransTracker


TRACKS = {
          'SiamRPNpp': SiamRPNTracker,
          'SiamMask': SiamMaskTracker,
          'SiamAtt': SiamAttTracker,
          'SiamBAN': SiamBANTracker,
          'SiamCAR': SiamCARTracker,
          'SiamCARM': SiamCARMTracker,
          'SiamGAT': SiamGATTracker,
          'TransT': TransTracker
         }


def build_tracker(cfg, model):
    return TRACKS[cfg.TRACK.TYPE](cfg, model)
