from ale.base.data_naif import NaifSpice
from ale.base.label_pds4 import Pds4Label
from ale.base.type_sensor import Framer, PushFrame, RollingShutter
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver
import spiceypy as spice


class JuicePds4LabelNaifSpiceDriver(Framer,  Pds4Label, NaifSpice, NoDistortion, Driver):
    @property
    def sensor_model_version(self):
        return 1

    @property
    def instrument_id(self):
        """
        Returns the short name of the instrument

         Returns
        -------
        : str
          Short name of the instrument
        """

        return "JUICE_JANUS"

    @property
    def instrument_name(self):
        """
        Returns the full name of the instrument

          Returns
        -------
        : str
          Full name of the instrument
        """
        return "JUICE_JANUS"

    @property
    def instrument_host_name(self):
        """
        Returns the full name of the instrument host

        Returns
        -------
        : str
          Full name of the instrument host
        """
        return "JUICE_SPACECRAFT"

    @property
    def instrument_host_id(self):
        """
        Returns the short name of the instrument host

          Returns
        -------
        : str
          Short name of the instrument host
        """
        return "JUICE_SPACECRAFT"

    @property
    def line_exposure_duration(self):
        """
        Line exposure duration returns the time between the exposures for
        subsequent lines.

        Returns
        -------
        : float
          Returns the line exposure duration in seconds from the PDS4 label.
        """
        return 0.2215002 * 1e-3  # Scale to seconds

    @property
    def focal_length(self):
        """
        Returns the focal length of the sensor
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : float
          focal length
        """
        # k = 'INS{}_FOCAL_LENGTH'.format(self.ikid)
        # print(f'searching keyword {k}')
        # return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])

        return 467.0
    

    # Consider expanding this to handle units
    # @property
    # def line_exposure_duration(self):
    #     """
    #     Line exposure duration returns the time between the exposures for
    #     subsequent lines.

    #     Returns
    #     -------
    #     : float
    #       Returns the line exposure duration in seconds from the PDS3 label.
    #     """
    #     return 0.2215002 * 0.001  # Scale to seconds

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[0])

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[1])

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        pixel_size = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0]
        return [0.0, (1 / pixel_size) * 1000.0, 0.0]

    @property
    def focal2pixel_samples(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        pixel_size = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0]
        return [0.0, 0.0, (1 / pixel_size) * 1000.0]
    
    @property
    def line_times(self):
        """
        Line exposure times for the image.
        Generally this will be normalized to [-1, 1] so that the jitter coefficients
        are well conditioned, but it is not necessarily required as long as the
        jitter coefficients are consistent.

        Returns
        -------
        : array
        """
        raise NotImplementedError

import numpy as np

class JuicePds4LabelNaifSpiceDriverPush(PushFrame,  Pds4Label, NaifSpice, NoDistortion, Driver):
    @property
    def sensor_model_version(self):
        return 1
    

    @property
    def interframe_delay(self):
        return float(
            self.label.find(
                ".//juice_janus:Onground_Processing/juice_janus:asw_tick_len",
                namespaces=self._ns,
            ).text
        ) * 1e-3

    # @property
    # def framelet_order_reversed(self):
    #     """
    #     Return a boolean indicating if the framelets are reversed.

    #     Returns
    #     -------
    #     Bool
    #         A boolean indicating if the framelets are reversed.
    #     """
    #     return False


    # @property
    # def framelets_flipped(self):
    #     """
    #     Return a boolean indicating if the framelets are flipped.

    #     Returns
    #     -------
    #     Bool
    #         A boolean indicating if the framelets are flipped.
    #     """

    #     return False
    

    @property
    def instrument_id(self):
        """
        Returns the short name of the instrument

         Returns
        -------
        : str
          Short name of the instrument
        """

        return "JUICE_JANUS"

    @property
    def instrument_name(self):
        """
        Returns the full name of the instrument

          Returns
        -------
        : str
          Full name of the instrument
        """
        return "JUICE_JANUS"

    @property
    def instrument_host_name(self):
        """
        Returns the full name of the instrument host

        Returns
        -------
        : str
          Full name of the instrument host
        """
        return "JUICE_SPACECRAFT"

    @property
    def instrument_host_id(self):
        """
        Returns the short name of the instrument host

          Returns
        -------
        : str
          Short name of the instrument host
        """
        return "JUICE_SPACECRAFT"

    @property
    def line_exposure_duration(self):
        """
        Line exposure duration returns the time between the exposures for
        subsequent lines.

        Returns
        -------
        : float
          Returns the line exposure duration in seconds from the PDS4 label.
        """
        return 0.2215002 * 1e-3  # Scale to seconds

    @property
    def focal_length(self):
        """
        Returns the focal length of the sensor
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : float
          focal length
        """
        # k = 'INS{}_FOCAL_LENGTH'.format(self.ikid)
        # print(f'searching keyword {k}')
        # return float(spice.gdpool('INS{}_FOCAL_LENGTH'.format(self.ikid), 0, 1)[0])

        return 467.0
    

    # Consider expanding this to handle units
    @property
    def line_exposure_duration(self):
        """
        Line exposure duration returns the time between the exposures for
        subsequent lines.

        Returns
        -------
        : float
          Returns the line exposure duration in seconds from the PDS3 label.
        """
        return 0.2215002 * 0.001  # Scale to seconds

    @property
    def detector_center_sample(self):
        """
        Returns the center detector sample. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[0])

    @property
    def detector_center_line(self):
        """
        Returns the center detector line. Expects ikid to be defined. This should
        be an integer containing the Naif Id code of the instrument.

        Returns
        -------
        : float
          Detector line of the principal point
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[1])

    @property
    def focal2pixel_lines(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector lines
        """
        pixel_size = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0]
        return [0.0, (1 / pixel_size) * 1000.0, 0.0]

    @property
    def focal2pixel_samples(self):
        """
        Expects ikid to be defined. This must be the integer Naif id code of the instrument

        Returns
        -------
        : list<double>
          focal plane to detector samples
        """
        pixel_size = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0]
        return [0.0, 0.0, (1 / pixel_size) * 1000.0]
    
    @property
    def line_times(self):
        """
        Line exposure times for the image.
        Generally this will be normalized to [-1, 1] so that the jitter coefficients
        are well conditioned, but it is not necessarily required as long as the
        jitter coefficients are consistent.

        Returns
        -------
        : array
        """
        raise NotImplementedError