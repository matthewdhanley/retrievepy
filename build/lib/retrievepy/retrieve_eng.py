import requests
import json
import pprint


class RetrieveEng:
    def __init__(self):
        print("init")
        self._catalog_url = 'http://dsweb.lasp.colorado.edu/ops/sorce/webtcad/latis/catalog.json'
        self._latis_url = 'http://dsweb.lasp.colorado.edu/ops/sorce/webtcad/latis/AnalogTelemetryItem.json?TMID=3796&time%3E=2018-11-27T12:37:41.688Z&time%3C=2018-11-28T12:37:41.688Z&binave(50000)&exclude_missing()'
        self._catalog = self._get_tlm_catalog()

    def _get_tlm_catalog(self):
        r = requests.get(self._catalog_url)
        tlm_dict = json.loads(r.text)
        return tlm_dict

    def _get_TMID(self):
        """
        gets the corresponding TMID from the latis catalog
        :param tlm_name: string formatted as pkt.mnemonic
        :return: integer TMID
        """

    def _get_data(self, tlmid, st, et):
        pass


if __name__ == "__main__":
    test = RetrieveEng()
    test._get_tlm_catalog()
