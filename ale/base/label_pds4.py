from lxml import etree



class Pds4Label:
    """
    Mix-in for parsing PDS4 labels.
    """

    @property
    def label(self) -> etree.Element:
        """
        Return the PDS4 label.

        Returns
        -------
        pvl.PVLModule
            The cube label as a PVLModule object.

        Raises
        ------
        ValueError
            Raised when an invalid label is provided.
        """
        if not hasattr(self, "_label"):
            try:
                with open(self._file, "r") as file:
                    self._label: etree.Element = etree.fromstring(file.read())
            except Exception:
                self._label: etree.Element = etree.fromstring(self._file)
            except:
                raise ValueError("{} is not a valid label".format(self._file))

            self._ns = {  # shall we read these straight from the xml header?? Guess so.
                "pds": "http://pds.nasa.gov/pds4/pds/v1",
                "img": "http://pds.nasa.gov/pds4/img/v1",
                "psa": "http://psa.esa.int/psa/v1",
                "juice_janus": "http://psa.esa.int/psa/juice/janus/v1",
            }
        return self._label

    @property
    def instrument_id(self):
        """
        Returns the short name of the instrument

         Returns
        -------
        : str
          Short name of the instrument
        """

        return self.label.find(
            ".//pds:Observing_System_Component[pds:type='Instrument']/pds:name",
            namespaces=self._ns,
        ).text

    @property
    def instrument_name(self):
        """
        Returns the full name of the instrument

          Returns
        -------
        : str
          Full name of the instrument
        """
        return self.label.find(
            ".//pds:Observing_System_Component[pds:type='Instrument']/pds:description",
            namespaces=self._ns,
        ).text

    @property
    def sensor_name(self):
        """
        Returns the name of the instrument

        Returns
        -------
        : str
          Name of the sensor
        """
        return self.instrument_name

    @property
    def instrument_host_id(self):
        """
        Returns the short name of the instrument host

          Returns
        -------
        : str
          Short name of the instrument host
        """
        return self.label.find(
            ".//pds:Observing_System_Component[pds:type='Host']/pds:name",
            namespaces=self._ns,
        ).text

    @property
    def instrument_host_name(self):
        """
        Returns the full name of the instrument host

        Returns
        -------
        : str
          Full name of the instrument host
        """
        return self.label.find(
            ".//pds:Observing_System_Component[pds:type='Host']/pds:description",
            namespaces=self._ns,
        ).text

    @property
    def platform_name(self):
        """
        Returns the name of the platform which the instrument is mounted on

        Returns
        -------
        : str
          platform name
        """
        return self.instrument_host_name

    @property
    def spacecraft_name(self):
        """
        Returns the name of the spacecraft

        Returns
        -------
        : str
          Full name of the spacecraft
        """
        return self.label.find(
            ".//pds:Investigation_Area[pds:type='Mission']/pds:name",
            namespaces=self._ns,
        ).text

    @property
    def utc_start_time(self):
        """
        Returns the start time of the image as a UTC string

        Returns
        -------
        : str
          Start time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]
        """
        return self.label.find(
            ".//pds:Time_Coordinates/pds:start_date_time",
            namespaces=self._ns,
        ).text

    @property
    def utc_stop_time(self):
        """
        Returns the stop time of the image as a UTC string

        Returns
        -------
        : str
          Stop time of the image in UTC YYYY-MM-DDThh:mm:ss[.fff]
        """
        return self.label.find(
            ".//pds:Time_Coordinates/pds:stop_date_time",
            namespaces=self._ns,
        ).text

    @property
    def image_lines(self):
        """
        Returns the number of lines in the image.

        Returns
        -------
        : int
          Number of lines in the image
        """
        return int(
            self.label.find(
                ".//pds:Array_2D_Image/pds:Axis_Array[pds:axis_name='Line']/pds:elements",
                namespaces=self._ns,
            ).text
        )

    @property
    def image_samples(self):
        """
        Returns the number of samples in the image.

        Returns
        -------
        : int
          Number of samples in the image
        """
        return int(
            self.label.find(
                ".//pds:Array_2D_Image/pds:Axis_Array[pds:axis_name='Sample']/pds:elements",
                namespaces=self._ns,
            ).text
        )

    @property
    def target_name(self):
        """
        Returns a target name uniquely identifying what an observation was capturing.
        This is most often a body name (e.g., Mars, Moon, Europa). This value is often
        use to acquire Ephemeris data from SPICE files; therefore it should be the same
        name spicelib expects in bodvrd calls.

        Returns
        -------
        : str
          Target name
        """
        return self.label.find(
            ".//pds:Target_Identification/pds:name", namespaces=self._ns
        ).text

    @property
    def sampling_factor(self):
        """
        Returns the summing factor from the PDS4 label. For example a return value of 2
        indicates that 2 lines and 2 samples (4 pixels) were summed and divided by 4
        to produce the output pixel value.

        Returns
        -------
        : int
          Number of samples and lines combined from the original data to produce a single pixel in this image
        """
        return 1  # to be fixed as per pds4 Downsampling(?)

    @property
    def line_summing(self):
        """
        Expects sampling_factor to be defined. This must be an integer
        containing the number of samples and lines combined from the original data

        Returns
        -------
        : int
           Number of detector lines summed to produce each image line
        """
        return self.sampling_factor

    @property
    def sample_summing(self):
        """
        Expects sampling_factor to be defined. This must be an integer
        containing the number of samples and lines combined from the original data

        Returns
        -------
        : int
           Number of detector lines summed to produce each image line
        """
        return self.sampling_factor

    @property
    def downtrack_summing(self):
        """
        Returns the number of detector pixels (normally in the line direction) that
        have been averaged to produce the output pixel

        Returns
        -------
        : int
          Number of downtrack pixels summed together
        """
        return 1  # to be fixed as per pds4 Downsampling(?)

    @property
    def crosstrack_summing(self):
        """
        Returns the number of detector pixels (normally in the sample direction) that
        have been averaged to produce the output pixel

        Returns
        -------
        : int
          Number of crosstrack pixels summed together
        """
        return 1  # to be fixed as per pds4 Downsampling(?)

    @property
    def spacecraft_clock_start_count(self):
        """
        The spacecraft clock start count, frequently used to determine the start time
        of the image.

        Returns
        -------
        : str
          Returns the start clock count string from the PDS4 label.
        """
        return self.label.find(
            ".//psa:Mission_Information/psa:spacecraft_clock_start_count",
            namespaces=self._ns,
        ).text

    @property
    def spacecraft_clock_stop_count(self):
        """
        The spacecraft clock stop count, frequently used to determine the stop time
        of the image.

        Returns
        -------
        : str
          Returns the stop clock count string from the PDS4 label.
        """
        count = self.label.find(
            ".//psa:Mission_Information/psa:spacecraft_clock_stop_count",
            namespaces=self._ns,
        ).text
        if count == "N/A":
            count = None
        return count

    @property
    def exposure_duration(self):
        """
        Returns the exposure duration converted to seconds. If the exposure duration
        is not present in the PDS4 label, then this property returns the
        line exposure duration. Expects line_exposure_duration to be defined. This
        should be a floating point number containing the line exposure duration.

         Returns
         -------
         : float
           Returns the exposure duration in seconds from the PDS4 label.
        """

        exp_node = self.label.find(
            ".//img:Exposure/img:exposure_duration", namespaces=self._ns
        )

        value = float(exp_node.text)
        unit = exp_node.attrib.get("unit", None)

        # The EXPOSURE_DURATION may either be stored as a (value, unit) or just a value
        if unit:
            unit = unit.lower()
            if unit == "ms" or unit == "msec" or unit == "millisecond":
                return value * 0.001
            else:
                return value

        # With no units, assume milliseconds
        else:
            return value * 0.001

    # Consider expanding this to handle units
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
        return 0.0  # Scale to seconds

    @property
    def filter_number(self):
        """
        Returns
        -------
        : int
          Returns the filter number from the PDS4 label
        """
        return int(
            self.label.find(
                ".//img:Optical_Filter/img:filter_number",
                namespaces=self._ns,
            ).text
        )
