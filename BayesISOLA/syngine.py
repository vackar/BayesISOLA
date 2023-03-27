import os
from obspy.clients.syngine import Client
from obspy import UTCDateTime
import numpy as np

class generate_query:
    """
    models: http://ds.iris.edu/ds/products/syngine/#earth
    """

    def __init__(self):
        self.bulk = []
        # sources [m_rr, m_tt, m_pp, m_rt,m_rp, m_tp]
        source1 = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        source2 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        source3 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        source4 = [1.0, -1.0, 0, 0, 0, 0]
        source5 = [1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
        source6 = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        self.sources = [source1, source2, source3, source4, source5, source6]
        self.dir_names = ["GFs1", "GFs2", "GFs3", "GFs4", "GFs5", "GFs6"]

    def do_query_simple(self, model, sourcelatitude: float, sourcelongitude: float, sourcedepthinmeters: float, origintime: UTCDateTime, starttime : UTCDateTime, endtime: UTCDateTime, output_root_path):
        client = Client()
        for count, sc in enumerate(self.sources):
            path_for_GFs = os.path.join(output_root_path, self.dir_names[count])
            if not os.path.exists(path_for_GFs):
                os.makedirs(path_for_GFs)
            st = client.get_waveforms_bulk(model=model, bulk=self.bulk, sourcelatitude=sourcelatitude,sourcelongitude=sourcelongitude,
                                    sourcedepthinmeters=sourcedepthinmeters, origintime=origintime, starttime=starttime,endtime=endtime,
                                    units="velocity", sourcemomenttensor=sc)
            for tr in st:
                file_name = tr.id
                name = os.path.join(path_for_GFs, file_name)
                tr.write(name, format="MSEED")

    def do_query_grid(self, model, sourcelatitude: float, sourcelongitude: float, sourcedepthinmeters: float, origintime: UTCDateTime, starttime: UTCDateTime, endtime: UTCDateTime, output_root_path, Lx, Ly, Lz, dx = 2, dy = 2, dz = 2):
        client = Client()
        depths = np.arange(int(sourcedepthinmeters-Lz), int(sourcedepthinmeters+Lz), dz)
        longs = np.arange(sourcelongitude - (Lx/112), sourcelongitude + (Lx/112), (dx/112))
        lats = np.arange(sourcelatitude - (Ly / 112), sourcelatitude + (Ly / 112), (dy/112))
        # First loop over source --> loop around source point
        for count, sc in enumerate(self.sources):
            path_for_GFs = os.path.join(output_root_path,self.dir_names[count])
            if not os.path.exists(path_for_GFs):
                os.makedirs(path_for_GFs)
            for zz in depths: # fixed depth ready to fix lat
                for yy in lats: # fixed longs ready to fix lat
                    for xx in longs: # loop over lats
                        print(zz, yy, xx)
                        st = client.get_waveforms_bulk(model=model, bulk=self.bulk, sourcelatitude=yy, sourcelongitude=xx, sourcedepthinmeters=zz, origintime=origintime, starttime=starttime, endtime=endtime, units="velocity", sourcemomenttensor=sc)
                        # saving
                        for tr in st:
                            file_name = tr.id + "." + str("{:.1f}".format(zz))+"_"+str("{:.4f}".format(yy))+"_"+\
                                        str("{:.4f}".format(xx))
                            name = os.path.join(path_for_GFs, file_name)
                            print("saving ", name)
                            tr.write(name, format="MSEED")
