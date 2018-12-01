import requests
import json
import pprint
import pylogger
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import configparser

LOGGER = pylogger.get_logger()

class RetrieveEng:
    def __init__(self):
        """ PRIVATE """
        self.__catalog_url = ''
        self.__latis_base_url = ''
        self._latis_url = None
        self._catalog = self.__get_tlm_catalog()
        self._access_url = None
        self._access_url_raw = None
        self._start_time_url = None
        self._end_time_url = None
        self._start_time = None
        self._end_time = None
        self._latis_mnemonic = None
        self._raw = False
        self._data = None
        self._time = None
        self._epoch = dt.datetime(1970, 1, 1)
        self.__config = None
        self.__config_file = 'latis.cfg'  # file containing latis api urls

        """ PUBLIC """
        self.mnemonic = None
        self.data = None
        self.time = None
        self.info = None
        self.exclude_missing = True

    """
    PRIVATE METHODS
    """

    def __get_config_parser(self):
        self.__config = configparser.ConfigParser()
        self.__config.read(self.__config_file)

    def __parse_config_file(self):
        self.__catalog_url = self.__config['latis']['catalog_url']
        self.__base_url = self.__config['latis']['base_url']

    def __get_tlm_catalog(self):
        """
        Gets the telemetry catalog from LaTiS
        :return: LaTiS response in a dict formed from the JSON
        """
        r = requests.get(self.__catalog_url)
        tlm_dict = json.loads(r.text)
        return tlm_dict

    def __get_access_url(self):
        """
        gets the corresponding access url from the latis catalog
        :param tlm_name: string formatted as pkt.mnemonic. Example : 'aac.mtbcmdyenable'
        :return: string access url. Example : 'DiscreteTelemetryItem.jsond?TMID=15113'
        """
        # Save access url in the event that the mnemonic is not found
        old_access_url = self._access_url
        old_access_url_raw = self._access_url_raw
        self._access_url = None
        self._access_url_raw = None

        catalogs = self._catalog['SORCE_Catalog']['catalog']
        for catalog in catalogs:
            for entry in catalog['dataset']:
                if self._latis_mnemonic == entry['name']:
                    self._access_url = entry['distribution'][0]['accessURL']
                    self._access_url_raw = entry['distribution'][1]['accessURL']

        if not self._access_url:
            LOGGER.warn('Could not find tlm item "'+self.mnemonic+'".')
            self._access_url = old_access_url
            self._access_url_raw = old_access_url_raw
            return False

        return True

    def __reformat_input(self, input):
        words = input.split()
        if len(words) != 2:
            LOGGER.error("Invalid input "+input)
            return False
        self._latis_mnemonic = words[0] + '.' + words[1]
        return True

    def _get_dt(self, time):
        try:
            year = time[0]
        except IndexError:
            LOGGER.error('Input HOUR incorrect')
            return False

        try:
            doy = time[1]
        except IndexError:
            LOGGER.error('Input DOY incorrect')
            return False

        try:
            hour = time[2]
        except IndexError:
            LOGGER.info('Assuming time correlation occurs at midnight')
            hour = 0

        try:
            minute = time[3]
        except IndexError:
            minute = 0

        try:
            second = time[4]
        except IndexError:
            second = 0

        new_date = dt.datetime(year=year, month=1, day=1) + dt.timedelta(days=doy-1, minutes=minute, hours=hour, seconds=second)
        return new_date

    def __set_start_end_dt(self, start, end):
        self._start_time = self._get_dt(start)
        self._end_time = self._get_dt(end)
        return True

    def __set_times_url(self):
        if self._start_time:
            self._start_time_url = '&time>='+self._start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            return False

        if self._start_time:
            self._end_time_url = '&time<='+self._end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            return False

        return True

    def __build_url(self):
        success = self.__set_times_url()

        if not success:
            return False

        success = self.__get_access_url()

        if not success:
            return False

        # Check if requesting raw data
        if self._raw:
            self._latis_url = self.__latis_base_url + self._access_url
        else:
            self._latis_url = self.__latis_base_url + self._access_url_raw

        # Add time componenets to url
        self._latis_url += self._start_time_url + self._end_time_url

        # add exclude_missing routine
        if self.exclude_missing:
            self._latis_url += '&exclude_missing()'

    def __convert_to_eu(self):
        coeffs = self.info[self._latis_mnemonic]['analog_conversions']['coefficients']

        index = 0
        self.data = np.zeros_like(self._data)

        for coeff in coeffs:
            val = list(coeff.values())[0]
            # np.add(self.data, np.multiply(np.power(self._data, index), val))
            self.data += np.multiply(np.power(self._data, index), val)
            index += 1

    def __convert_time_to_dt(self):
        self.time = [self._epoch + dt.timedelta(milliseconds=x) for x in self._time]

    def __get_data(self):
        in_data = requests.get(self._latis_url)
        response_dict = json.loads(in_data.text)
        self.info = response_dict[self._latis_mnemonic]['metadata']
        as_np = np.asarray(response_dict[self._latis_mnemonic]['data'])
        self._data = as_np[:, 1]
        self._time = as_np[:, 0]
        self.__convert_to_eu()
        self.__convert_time_to_dt()
        return True

    """
    PUBLIC METHODS
    """

    def fetch(self, mnemonic, start_time, stop_time, raw=False):
        self.mnemonic = mnemonic
        success = self.__reformat_input(mnemonic)

        if success:
            self.__set_start_end_dt(start_time, stop_time)

        if success:
            success = self.__build_url()

        success = self.__get_data()

    def plot(self):
        # convert from datetime to matplotlib readable
        dates = mdates.date2num(self.time)

        # create the figure
        fig = plt.figure()

        # open one subplot spanning figure
        ax = fig.add_subplot(111)

        # Configure x-ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%j - %H:%M:%S'))  # set x axis format
        ax.grid(True)  # turn on the grid

        plt.plot_date(dates, self.data,
                      # ls='--',
                      color='gray',
                      marker='.',
                      markerfacecolor='green',
                      markeredgecolor='green',
                      markersize=5)

        fig.autofmt_xdate(rotation=50)  # rotate the tick labels

        ax.set_title(self.mnemonic)  # sets the title of the plot

        fig.tight_layout()  # makes it so labels/ticks are not cut off

        # show the plot
        plt.show()


if __name__ == "__main__":
    test = RetrieveEng()
    # test.__get_access_url('aac.mtbcmdyenable')
    test.fetch('xb slrarcur',[2018,300], [2018,301])
    print(test.data)
