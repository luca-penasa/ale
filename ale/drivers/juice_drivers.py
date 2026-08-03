import math
import numpy as np
from ale.base.data_naif import NaifSpice
from ale.base.label_pds4 import Pds4Label
from ale.base.type_sensor import Framer, PushFrame, RollingShutter
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

    ** Known broken, kept only for reference/history -- do not use. **
    usgscsm's UsgsAstroPushFrameSensorModel::losToEcf computes
    ``frameletLine = fmod(line, summedFrameletHeight)`` for the focal-plane
    geometry; with framelet_height=1 (the only physically correct value for
    JANUS, since each row is read out independently) this collapses to ~0
    for every line, so imageToGround's ray direction never varies with the
    true detector row -- only per-framelet time does. Confirmed empirically
    by cross-checking against janus_projector's independently validated
    rolling-shutter pipeline: the line-axis ground-point spread came out
    ~15x too small, while the (unaffected) sample-axis spread matched the
    expected FOV geometry. No existing ale driver uses framelet_height=1;
    every real push-frame mission (MRO, LRO WAC) uses genuine multi-row
    framelets (14-16 rows) with real internal geometry, unlike JANUS's
    rolling-shutter readout of an otherwise ordinary framing camera. See
    JuicePds4LabelNaifSpiceDriverRollingShutter for the replacement
    (USGS_ASTRO_FRAME_SENSOR_MODEL + usgscsm's jitter correction).
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


class JuicePds4LabelNaifSpiceDriverRollingShutter(RollingShutter, Framer, Pds4Label, NaifSpice, NoDistortion, Driver):
    """
    Driver for JUICE JANUS PDS4 labels using NAIF SPICE data.

    Models JANUS as a standard framing camera (USGS_ASTRO_FRAME_SENSOR_MODEL:
    full 2D per-pixel pinhole geometry, one reference position/attitude for
    the whole image) with an added rolling-shutter *jitter* correction
    (usgscsm's optional addJitter/removeJitter mechanism) layered on top.

    This replaces JuicePds4LabelNaifSpiceDriverPush -- see that class's
    docstring for why USGS_ASTRO_PUSH_FRAME_SENSOR_MODEL is structurally
    unable to represent JANUS. USGS_ASTRO_FRAME_SENSOR_MODEL's focal-plane
    geometry has no such collapse (it varies correctly with both line and
    sample), and its jitter mechanism corrects the *reference* pixel
    location by a small, per-line, physically-derived pixel-space offset --
    see line_jitter_coeffs / sample_jitter_coeffs / line_times below for how
    that offset is computed from real per-line SPICE attitude.
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
    def interframe_delay(self):
        """
        Per-line readout interval in seconds, from the PDS4 img namespace's
        line_readout_time field. See JuicePds4LabelNaifSpiceDriverPush's
        interframe_delay docstring for why this field (not
        juice_janus:asw_tick_len) is the correct one.

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
    def ephemeris_stop_time(self):
        """
        Start of exposure plus the true total rolling-shutter readout span
        (image_lines rows, one interframe_delay apart) -- used only to
        anchor center_ephemeris_time near the middle of the actual
        acquisition; the frame model itself uses a single reference
        position/attitude at center_ephemeris_time, corrected per-line by
        the jitter polynomial below.

        Returns
        -------
        : float
        """
        return self.ephemeris_start_time + self.image_lines * self.interframe_delay

    @property
    def focal_length(self):
        """
        Focal length in millimeters.

        The JANUS IK does not define an ``INS<id>_FOCAL_LENGTH`` keyword --
        JANUS's focal length must be derived from pixel pitch and IFOV:
        f = pixel_size / tan(IFOV). See janus_projector.janus.janus_from_spice
        (projector.git) for the same derivation, cross-validated against
        real flight data.

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

    # --- rolling-shutter jitter ---

    @property
    def line_times(self):
        """
        Per-line times, normalized to [-1, 1] across the frame (line 0 ->
        -1, line image_lines-1 -> +1). One entry per detector row -- usgscsm
        indexes this array directly by (rounded) line number.

        Returns
        -------
        : list<float>
        """
        n = self.image_lines
        return list(np.linspace(-1.0, 1.0, n))

    def _jitter_fit_coeffs(self, degree=2):
        """
        Fit line/sample jitter polynomial coefficients from real per-line
        SPICE attitude.

        Physical model: at each sampled line i, compute the true camera
        boresight direction in an inertial frame (J2000) using the real
        attitude at et_line(i), then express that direction in the
        *reference* camera frame (the one the frame model actually uses,
        anchored at center_ephemeris_time) by composing with the inverse of
        the reference attitude. For a camera whose pointing barely changes
        during one frame's readout (true for JANUS in typical, non-flyby
        observations -- see JuicePds4LabelNaifSpiceDriverPush's docstring
        for the empirical numbers), this deviation is small and this
        linearised pixel-space treatment is an excellent approximation;
        J2000 is used only as a common intermediate frame (any inertial or
        body frame would cancel identically) so this needs no target-body
        frame resolution. Position drift during the frame is not modeled
        here (usgscsm's FRAME model has no per-line position table to
        correct -- only pixel-space line/sample jitter); for JANUS this is
        the smaller of the two effects (see the same docstring).

        The resulting deviation, projected through the focal length /
        pixel size onto the focal plane, gives a pixel-space (line, sample)
        offset at each sampled time. A polynomial with no constant term
        (matching usgscsm's removeJitter/addJitter convention) is then
        least-squares fit against the normalized line_times -- the
        reference time (t=0, the center line) has zero jitter by
        construction, consistent with dropping the constant term.

        Parameters
        ----------
        degree : int
            Highest polynomial order (no constant term), e.g. 2 fits
            coefficients for t^2 and t^1.

        Returns
        -------
        : (list<float>, list<float>)
          (line_jitter_coeffs, sample_jitter_coeffs), highest order first.
        """
        n_lines = self.image_lines
        n_samples_fit = min(41, n_lines)
        sample_lines = np.linspace(0, n_lines - 1, n_samples_fit)
        sample_times = -1.0 + 2.0 * sample_lines / (n_lines - 1)

        focal_length_mm = self.focal_length
        pixel_size_mm = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0] * 1e-3

        center_et = self.center_ephemeris_time
        r_center = np.array(spice.pxform("JUICE_JANUS", "J2000", center_et))

        d_line = np.zeros(n_samples_fit)
        d_sample = np.zeros(n_samples_fit)
        for k, line in enumerate(sample_lines):
            et = self.ephemeris_start_time + line * self.interframe_delay
            r = np.array(spice.pxform("JUICE_JANUS", "J2000", et))
            d_world = r @ np.array([0.0, 0.0, 1.0])  # camera +Z = boresight
            d_ref = r_center.T @ d_world
            d_line[k] = focal_length_mm * (d_ref[0] / d_ref[2]) / pixel_size_mm
            d_sample[k] = focal_length_mm * (d_ref[1] / d_ref[2]) / pixel_size_mm

        # Design matrix for [t^degree, ..., t^1] -- no constant column.
        basis = np.vstack([sample_times ** p for p in range(degree, 0, -1)]).T
        line_coeffs, *_ = np.linalg.lstsq(basis, d_line, rcond=None)
        sample_coeffs, *_ = np.linalg.lstsq(basis, d_sample, rcond=None)
        return line_coeffs.tolist(), sample_coeffs.tolist()

    @property
    def line_jitter_coeffs(self):
        if not hasattr(self, "_jitter_coeffs"):
            self._jitter_coeffs = self._jitter_fit_coeffs()
        return self._jitter_coeffs[0]

    @property
    def sample_jitter_coeffs(self):
        if not hasattr(self, "_jitter_coeffs"):
            self._jitter_coeffs = self._jitter_fit_coeffs()
        return self._jitter_coeffs[1]
