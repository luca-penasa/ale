import numpy as np

class LineScanner():
    """
    Mix-in for line scan sensors.
    """

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
        return "USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL"

    @property
    def line_scan_rate(self):
        """
        Expects ephemeris_start_time to be defined. This should be a float
        containing the start time of the image.
        Expects center_ephemeris_time to be defined. This should be a float
        containing the average of the start and end ephemeris times.

        Returns
        -------
        : list
          Start lines
        : list
          Line times
        : list
          Exposure durations
        """
        t0_ephemeris = self.ephemeris_start_time - self.center_ephemeris_time
        return [0.5], [t0_ephemeris], [self.exposure_duration]

    @property
    def ephemeris_time(self):
        """
        Returns an array of times between the start/stop ephemeris times
        based on the number of lines in the image.
        Expects ephemeris start/stop times to be defined. These should be
        floating point numbers containing the start and stop times of the
        images.
        Expects image_lines to be defined. This should be an integer containing
        the number of lines in the image.

        Returns
        -------
        : ndarray
          ephemeris times split based on image lines
        """
        if not hasattr(self, "_ephemeris_time"):
          self._ephemeris_time = np.linspace(self.ephemeris_start_time, self.ephemeris_stop_time, self.image_lines + 1)
        return self._ephemeris_time

    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the number of lines
        times the exposure duration. Expects ephemeris start time, exposure duration
        and image lines to be defined. These should be double precision numbers
        containing the ephemeris start, exposure duration and number of lines of
        the image.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.ephemeris_start_time + (self.image_lines * self.exposure_duration)


class PushFrame():

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
        return "USGS_ASTRO_PUSH_FRAME_SENSOR_MODEL"


    @property
    def ephemeris_time(self):
        """
        Returns an array of times between the start/stop ephemeris times
        based on the number of lines in the image.
        Expects ephemeris start/stop times to be defined. These should be
        floating point numbers containing the start and stop times of the
        images.
        Expects image_lines to be defined. This should be an integer containing
        the number of lines in the image.

        Returns
        -------
        : ndarray
          ephemeris times split based on image lines
        """

        return np.arange(self.ephemeris_start_time + (.5 * self.exposure_duration), self.ephemeris_stop_time + self.interframe_delay, self.interframe_delay)


    @property
    def framelet_height(self):
        return 1


    @property
    def framelet_order_reversed(self):
        return False


    @property
    def framelets_flipped(self):
        return False


    @property
    def num_frames(self):
        return int(self.image_lines // self.framelet_height)

    @property
    def num_lines_overlap(self):
        """
        Returns
        -------
        : int
          For PushFrame sensors, returns how many many lines of a framelet
          overlap with neighboring framelets.
        """
        return 0

    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the number of lines
        times the exposure duration. Expects ephemeris start time, exposure duration
        and image lines to be defined. These should be double precision numbers
        containing the ephemeris start, exposure duration and number of lines of
        the image.

        Returns
        -------
        : double
          Center ephemeris time for an image
        """
        return self.ephemeris_start_time + (self.interframe_delay) * (self.num_frames - 1) + self.exposure_duration



class Framer():
    """
    Mix-in for framing sensors.
    """

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS Frame model
        """
        return "USGS_ASTRO_FRAME_SENSOR_MODEL"

    @property
    def ephemeris_time(self):
        """
        Returns the center ephemeris time for the image which is start time plus
        half of the exposure duration.
        Expects center_ephemeris_time to be defined. This should be a double
        containing the average of the start and stop ephemeris times.

        Returns
        -------
        : double
          Center ephemeris time for the image
        """
        return [self.center_ephemeris_time]

    @property
    def ephemeris_stop_time(self):
        """
        Returns the sum of the starting ephemeris time and the exposure duration.
        Expects ephemeris start time and exposure duration to be defined. These
        should be double precision numbers containing the ephemeris start and
        exposure duration of the image.

        Returns
        -------
        : double
          Ephemeris stop time for an image
        """
        return self.ephemeris_start_time + self.exposure_duration

class Radar():
    """
    Mix-in for synthetic aperture radar sensors.
    """

    @property
    def name_model(self):
        """
        Returns Key used to define the sensor type. Primarily
        used for generating camera models.

        Returns
        -------
        : str
          USGS SAR (synthetic aperture radar) model
        """
        return "USGS_ASTRO_SAR_MODEL"

    @property
    def ephemeris_time(self):
        """
        Returns an array of times between the start/stop ephemeris times
        based on the start/stop times with a timestep (stop - start) / image_lines.
        Expects ephemeris start/stop times to be defined. These should be
        floating point numbers containing the start and stop times of the
        images.

        Returns
        -------
        : ndarray
          ephemeris times split based on image lines
        """
        if not hasattr(self, "_ephemeris_time"):
          self._ephemeris_time = np.linspace(self.ephemeris_start_time, self.ephemeris_stop_time, self.image_lines + 1)
        return self._ephemeris_time

    @property
    def wavelength(self):
        """
        Returns the wavelength used for image acquisition.

        Returns
        -------
        : double
          Wavelength used to create an image in meters
        """
        raise NotImplementedError

    @property
    def line_exposure_duration(self):
        """
        Returns the exposure duration for each line.

        Returns
        -------
        : double
          Exposure duration for a line
        """
        raise NotImplementedError


    @property
    def scaled_pixel_width(self):
        """
        Returns the scaled pixel width

        Returns
        -------
        : double
          Scaled pixel width
        """
        raise NotImplementedError

    @property
    def range_conversion_coefficients(self):
        """
        Returns the range conversion coefficients

        Returns
        -------
        : list
          Coefficients needed for range conversion
        """
        raise NotImplementedError

    @property
    def range_conversion_times(self):
        """
        Returns the times associated with the range conversion coefficients

        Returns
        -------
        : list
          Times for the range conversion coefficients
        """
        raise NotImplementedError

    @property
    def look_direction(self):
        """
        Direction of the look (left or right)

        Returns
        -------
        : string
          left or right
        """
        raise NotImplementedError


class RollingShutter():
    """
    Mix-in for sensors with a rolling shutter.
    Specifically those with rolling shutter jitter.
    """

    @property
    def sample_jitter_coeffs(self):
        """
        Polynomial coefficients for the sample jitter.
        The highest order coefficient comes first.
        There is no constant coefficient, assumed 0.

        Returns
        -------
        : array
        """
        raise NotImplementedError

    @property
    def line_jitter_coeffs(self):
        """
        Polynomial coefficients for the line jitter.
        The highest order coefficient comes first.
        There is no constant coefficient, assumed 0.

        Returns
        -------
        : array
        """
        raise NotImplementedError

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
