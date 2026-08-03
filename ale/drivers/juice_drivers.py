import math
import numpy as np
from ale.base.data_naif import NaifSpice
from ale.base.label_pds4 import Pds4Label
from ale.base.type_sensor import Framer, PushFrame
from ale.base.type_distortion import NoDistortion
from ale.base.base import Driver
import spiceypy as spice


class JuicePds4LabelNaifSpiceDriver(Framer, Pds4Label, NaifSpice, NoDistortion, Driver):
    """
    Driver for JUICE JANUS PDS4 labels using NAIF SPICE data.
    Models JANUS as a standard framing camera (single exposure, no rolling
    shutter correction).  Use JuicePds4LabelNaifSpiceDriverPush for the
    rolling-shutter push-frame approximation.
    """

    @property
    def sensor_model_version(self):
        return 1

    @property
    def instrument_id(self):
        return "JUICE_JANUS"

    @property
    def instrument_name(self):
        return "JUICE_JANUS"

    @property
    def instrument_host_name(self):
        return "JUICE_SPACECRAFT"

    @property
    def instrument_host_id(self):
        return "JUICE_SPACECRAFT"

    @property
    def line_exposure_duration(self):
        """
        Line exposure duration in seconds.

        Returns
        -------
        : float
        """
        return 0.2215002 * 1e-3

    @property
    def focal_length(self):
        """
        Focal length in millimeters.

        The JANUS IK does not define an ``INS<id>_FOCAL_LENGTH`` keyword --
        unlike most framing cameras, JANUS's focal length must be derived
        from pixel pitch and IFOV: ``f = pixel_size / tan(IFOV)``. See
        ``janus_projector.janus.janus_from_spice`` (the projector.git
        reference implementation) for the same derivation, cross-validated
        against real flight data.

        Returns
        -------
        : float
        """
        pixel_size_mm = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0] * 1e-3
        ifov_rad = spice.gdpool("INS{}_IFOV".format(self.ikid), 0, 1)[0]
        return pixel_size_mm / math.tan(ifov_rad)

    @property
    def detector_center_sample(self):
        """
        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[0])

    @property
    def detector_center_line(self):
        """
        Returns
        -------
        : float
          Detector line of the principal point
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[1])

    @property
    def focal2pixel_lines(self):
        """
        Returns
        -------
        : list<double>
          Focal plane to detector lines transform
        """
        pixel_size = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0]
        return [0.0, (1 / pixel_size) * 1000.0, 0.0]

    @property
    def focal2pixel_samples(self):
        """
        Returns
        -------
        : list<double>
          Focal plane to detector samples transform
        """
        pixel_size = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0]
        return [0.0, 0.0, (1 / pixel_size) * 1000.0]


class JuicePds4LabelNaifSpiceDriverPush(PushFrame, Pds4Label, NaifSpice, NoDistortion, Driver):
    """
    Driver for JUICE JANUS PDS4 labels using NAIF SPICE data.
    Models JANUS as a push-frame sensor: each detector row is treated as an
    independent framelet with height 1, and the inter-framelet delay is the
    per-line readout interval (asw_tick_len from the onground-processing
    metadata, converted from milliseconds to seconds).  This is the
    recommended approximation for JANUS rolling-shutter data.
    """

    @property
    def sensor_model_version(self):
        return 1

    @property
    def instrument_id(self):
        return "JUICE_JANUS"

    @property
    def instrument_name(self):
        return "JUICE_JANUS"

    @property
    def instrument_host_name(self):
        return "JUICE_SPACECRAFT"

    @property
    def instrument_host_id(self):
        return "JUICE_SPACECRAFT"

    # --- timing ---

    @property
    def interframe_delay(self):
        """
        Per-line readout interval in seconds, taken from the PDS4 ``img``
        namespace's ``line_readout_time`` field (stored in the label in
        milliseconds).

        The label also carries a ``juice_janus:Onground_Processing/
        juice_janus:asw_tick_len`` field with a different value (~1 ms vs
        ~0.2215 ms here) -- that is the on-board *application software*
        processing tick, not the physical detector row cadence, and using
        it instead produces a per-line pointing error that grows roughly
        linearly across the frame (confirmed empirically: ~145 km at the
        last line vs ~0 at the first, cross-checked against
        janus_projector's independently validated rolling-shutter pipeline
        -- see projector.git's docs/janus_camera_model.md and
        AGENTS.md, which document ``line_readout_time`` /
        ``JANUS_LINE_READOUT_S`` as the correct field, cross-validated to
        ~0.2% geolocation accuracy against an independent LROC WAC
        control).

        Returns
        -------
        : float
        """
        return float(
            self.label.find(
                ".//img:line_readout_time",
                namespaces=self._ns,
            ).text
        ) * 1e-3

    @property
    def line_exposure_duration(self):
        """
        Integration time of a single detector row in seconds.

        Returns
        -------
        : float
        """
        return 0.2215002 * 1e-3

    @property
    def ephemeris_stop_time(self):
        """
        End of exposure: start + all interframe delays + one final integration.

        Returns
        -------
        : float
        """
        return (self.ephemeris_start_time
                + self.num_frames * self.interframe_delay
                + self.line_exposure_duration)

    @property
    def ephemeris_time(self):
        """
        Sample times for the position/quaternion tables, spaced by exactly
        ``interframe_delay`` between consecutive framelets.

        The base ``PushFrame.ephemeris_time`` (``ale.base.type_sensor``)
        does ``np.linspace(ephemeris_start_time, ephemeris_stop_time,
        image_lines + 1)``. Combined with this driver's
        ``ephemeris_stop_time`` above -- which intentionally adds one extra
        ``line_exposure_duration`` on top of ``num_frames *
        interframe_delay`` to report the true end of the last integration
        -- that would space the table's samples by
        ``interframe_delay + line_exposure_duration / num_frames``, not
        ``interframe_delay``. usgscsm's
        ``UsgsAstroPushFrameSensorModel::getImageTime`` assumes the table is
        sampled at exactly ``ephemeris_start_time + frameletNumber *
        interframe_delay`` (plus a separate ``0.5 * exposure_duration``
        mid-exposure bias applied only when evaluating a query time, not
        when spacing the table) -- so the base implementation would
        introduce a systematic per-line timing drift, growing to a full
        ``line_exposure_duration`` (one whole line period) by the last
        line. Overriding here keeps the table's spacing exactly consistent
        with what usgscsm's rolling-shutter timing model expects.

        Returns
        -------
        : ndarray
        """
        if not hasattr(self, "_ephemeris_time"):
            self._ephemeris_time = (
                self.ephemeris_start_time
                + self.interframe_delay * np.arange(self.num_frames + 1)
            )
        return self._ephemeris_time

    # --- push-frame geometry ---

    @property
    def framelet_height(self):
        """
        Each framelet is a single detector row.

        Returns
        -------
        : int
        """
        return 1

    @property
    def framelet_order_reversed(self):
        """
        Framelets are read out from the first line to the last (not reversed).

        Returns
        -------
        : bool
        """
        return False

    @property
    def framelets_flipped(self):
        """
        Framelets are not flipped within the image.

        Returns
        -------
        : bool
        """
        return False

    # --- optics ---

    @property
    def focal_length(self):
        """
        Focal length in millimeters.

        The JANUS IK does not define an ``INS<id>_FOCAL_LENGTH`` keyword --
        unlike most framing cameras, JANUS's focal length must be derived
        from pixel pitch and IFOV: ``f = pixel_size / tan(IFOV)``. See
        ``janus_projector.janus.janus_from_spice`` (the projector.git
        reference implementation) for the same derivation, cross-validated
        against real flight data.

        Returns
        -------
        : float
        """
        pixel_size_mm = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0] * 1e-3
        ifov_rad = spice.gdpool("INS{}_IFOV".format(self.ikid), 0, 1)[0]
        return pixel_size_mm / math.tan(ifov_rad)

    @property
    def detector_center_sample(self):
        """
        Returns
        -------
        : float
          Detector sample of the principal point
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[0])

    @property
    def detector_center_line(self):
        """
        Returns
        -------
        : float
          Detector line of the principal point
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[1])

    @property
    def focal2pixel_lines(self):
        """
        Returns
        -------
        : list<double>
          Focal plane to detector lines transform
        """
        pixel_size = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0]
        return [0.0, (1 / pixel_size) * 1000.0, 0.0]

    @property
    def focal2pixel_samples(self):
        """
        Returns
        -------
        : list<double>
          Focal plane to detector samples transform
        """
        pixel_size = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0]
        return [0.0, 0.0, (1 / pixel_size) * 1000.0]