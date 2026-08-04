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

        Notes
        -----
        The IK's INS<id>_CCD_CENTER is 1-based; usgscsm's detector_center
        (m_detectorLineOrigin/m_detectorSampleOrigin) is used directly as
        the pixel-space origin with no implicit offset, and CSM's own pixel
        convention has the upper-left pixel's centre at (0.5, 0.5) -- i.e.
        0-based. Subtract 1 to convert, matching
        janus_projector.janus.janus_from_spice (projector.git), which
        already does this and is validated against real flight data.
        Confirmed empirically: using the raw 1-based value here produced
        exactly a 1-pixel-scale systematic offset against projector's
        independently validated geolocation.
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[0]) - 1.0

    @property
    def detector_center_line(self):
        """
        Returns
        -------
        : float
          Detector line of the principal point

        Notes
        -----
        See detector_center_sample -- same 1-based-to-0-based correction.
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[1]) - 1.0

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

        Notes
        -----
        The IK's INS<id>_CCD_CENTER is 1-based; usgscsm's detector_center
        (m_detectorLineOrigin/m_detectorSampleOrigin) is used directly as
        the pixel-space origin with no implicit offset, and CSM's own pixel
        convention has the upper-left pixel's centre at (0.5, 0.5) -- i.e.
        0-based. Subtract 1 to convert, matching
        janus_projector.janus.janus_from_spice (projector.git), which
        already does this and is validated against real flight data.
        Confirmed empirically: using the raw 1-based value here produced
        exactly a 1-pixel-scale systematic offset against projector's
        independently validated geolocation.
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[0]) - 1.0

    @property
    def detector_center_line(self):
        """
        Returns
        -------
        : float
          Detector line of the principal point

        Notes
        -----
        See detector_center_sample -- same 1-based-to-0-based correction.
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[1]) - 1.0

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

        Notes
        -----
        The IK's INS<id>_CCD_CENTER is 1-based; usgscsm's detector_center
        (m_detectorLineOrigin/m_detectorSampleOrigin) is used directly as
        the pixel-space origin with no implicit offset, and CSM's own pixel
        convention has the upper-left pixel's centre at (0.5, 0.5) -- i.e.
        0-based. Subtract 1 to convert, matching
        janus_projector.janus.janus_from_spice (projector.git), which
        already does this and is validated against real flight data.
        Confirmed empirically: using the raw 1-based value here produced
        exactly a 1-pixel-scale systematic offset against projector's
        independently validated geolocation.
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[0]) - 1.0

    @property
    def detector_center_line(self):
        """
        Returns
        -------
        : float
          Detector line of the principal point

        Notes
        -----
        See detector_center_sample -- same 1-based-to-0-based correction.
        """
        return float(spice.gdpool("INS{}_CCD_CENTER".format(self.ikid), 0, 2)[1]) - 1.0

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

    def _jitter_fit_coeffs(self, degree=3, n_fit=81, n_validate=40):
        """
        Fit line/sample jitter polynomial coefficients from real per-line
        SPICE position *and* attitude.

        Earlier revision (attitude-only, no reference ground point) was
        found to be insufficient: on a close lunar flyby, spacecraft
        position change during one frame's rolling-shutter readout can
        dominate over attitude change and produce large apparent geometric
        distortion (observed: ~15% frame compression) -- pure boresight
        rotation cannot capture that; it takes an actual displaced
        observer looking at a fixed ground point.

        Physical model: intersect the camera boresight ray at
        center_ephemeris_time with a sphere of the target body's mean
        radius to get a fixed reference ground point G (body-fixed frame).
        Then, for each sampled line i, compute where G would appear in the
        *reference* camera frame (anchored at center_ephemeris_time) if the
        spacecraft's real position and attitude at et_line(i) were used
        instead of the reference ones:

            vec_i        = G - sc_position(et_line_i)          [body-fixed]
            vec_camera_i = R(et_line_i)^T @ vec_i               [camera frame at line i]
            ... reprojected through the *reference* pointing (R_center)
            by definition of "what pixel would this ray hit if the model
            only knew about the reference attitude/position" --
            camera-frame ray direction is what matters, so this is just
            vec_camera_i normalized and projected through the pinhole
            formula directly (no separate re-composition with R_center is
            needed here, unlike the attitude-only case, because G is fixed
            in body-fixed space, not camera space).

        This captures both attitude *and* position drift's combined effect
        on a fixed ground point's apparent pixel location -- the same
        physics janus_projector's own rolling-shutter pipeline uses for
        per-pixel backprojection (see docs/janus_camera_model.md in
        projector.git), just reduced here to a low-order polynomial for
        usgscsm's jitter mechanism. A held-out validation set (n_validate
        points, disjoint from the n_fit fit points) is used to report the
        polynomial fit's own residual, since a poor fit would silently produce
        wrong geometry -- see the module-level get_jitter_fit_diagnostics
        helper to inspect this without constructing a full driver.

        A polynomial with no constant term (matching usgscsm's
        removeJitter/addJitter convention) is least-squares fit against the
        normalized line_times -- the reference time (t=0, the center line)
        is exact by construction (G was defined from the center-time
        boresight), consistent with dropping the constant term.

        Parameters
        ----------
        degree : int
            Highest polynomial order (no constant term), e.g. 3 fits
            coefficients for t^3, t^2, t^1.
        n_fit : int
            Number of lines sampled to fit the polynomial.
        n_validate : int
            Number of additional, disjoint lines sampled to compute a
            held-out residual (see fit_residual_px on the returned dict).

        Returns
        -------
        : (list<float>, list<float>, dict)
          (line_jitter_coeffs, sample_jitter_coeffs, diagnostics) --
          diagnostics has keys 'fit_rms_px', 'fit_max_px', 'validate_rms_px',
          'validate_max_px'.
        """
        n_lines = self.image_lines
        focal_length_mm = self.focal_length
        pixel_size_mm = spice.gdpool("INS{}_PIXEL_SIZE".format(self.ikid), 0, 1)[0] * 1e-3
        body_frame = self.reference_frame
        target = self.target_name
        sc = self.spacecraft_name
        abcorr = self.light_time_correction
        center_et = self.center_ephemeris_time

        radii = spice.bodvrd(target, "RADII", 3)[1]
        mean_radius = float(np.mean(radii))

        r_center = np.array(spice.pxform("JUICE_JANUS", body_frame, center_et))
        pos_center, _ = spice.spkpos(sc, center_et, body_frame, abcorr, target)
        pos_center = np.array(pos_center)

        boresight_center = r_center @ np.array([0.0, 0.0, 1.0])
        # Ray-sphere intersection (nearest root) from pos_center along boresight_center.
        b = np.dot(pos_center, boresight_center)
        c = np.dot(pos_center, pos_center) - mean_radius ** 2
        disc = b * b - c
        if disc < 0:
            raise ValueError(
                "JANUS boresight at center_ephemeris_time does not intersect "
                f"the target body ({target}) -- cannot derive a reference "
                "ground point for the rolling-shutter jitter fit."
            )
        t_hit = -b - math.sqrt(disc)
        ground_point = pos_center + t_hit * boresight_center

        def _pixel_delta(line):
            et = self.ephemeris_start_time + line * self.interframe_delay
            r = np.array(spice.pxform("JUICE_JANUS", body_frame, et))
            pos, _ = spice.spkpos(sc, et, body_frame, abcorr, target)
            vec_body = ground_point - np.array(pos)
            vec_camera = r.T @ vec_body
            d_line = focal_length_mm * (vec_camera[0] / vec_camera[2]) / pixel_size_mm
            d_sample = focal_length_mm * (vec_camera[1] / vec_camera[2]) / pixel_size_mm
            return d_line, d_sample

        def _basis(times):
            return np.vstack([times ** p for p in range(degree, 0, -1)]).T

        fit_lines = np.linspace(0, n_lines - 1, n_fit)
        fit_times = -1.0 + 2.0 * fit_lines / (n_lines - 1)
        d_line = np.zeros(n_fit)
        d_sample = np.zeros(n_fit)
        for k, line in enumerate(fit_lines):
            d_line[k], d_sample[k] = _pixel_delta(line)

        basis = _basis(fit_times)
        line_coeffs, *_ = np.linalg.lstsq(basis, d_line, rcond=None)
        sample_coeffs, *_ = np.linalg.lstsq(basis, d_sample, rcond=None)

        fit_resid = np.hypot(basis @ line_coeffs - d_line, basis @ sample_coeffs - d_sample)

        # Held-out lines, offset from the fit grid so none coincide.
        val_lines = np.linspace(0.5, n_lines - 1.5, n_validate)
        val_times = -1.0 + 2.0 * val_lines / (n_lines - 1)
        dv_line = np.zeros(n_validate)
        dv_sample = np.zeros(n_validate)
        for k, line in enumerate(val_lines):
            dv_line[k], dv_sample[k] = _pixel_delta(line)
        val_basis = _basis(val_times)
        val_resid = np.hypot(
            val_basis @ line_coeffs - dv_line, val_basis @ sample_coeffs - dv_sample
        )

        diagnostics = {
            "fit_rms_px": float(np.sqrt(np.mean(fit_resid ** 2))),
            "fit_max_px": float(fit_resid.max()),
            "validate_rms_px": float(np.sqrt(np.mean(val_resid ** 2))),
            "validate_max_px": float(val_resid.max()),
        }
        return line_coeffs.tolist(), sample_coeffs.tolist(), diagnostics

    @property
    def jitter_fit_diagnostics(self):
        """
        Held-out residual (px) of the jitter polynomial fit against the
        real per-line ground-point reprojection it's fit to -- see
        _jitter_fit_coeffs. A large residual means the polynomial degree
        is too low to represent this observation's actual motion (e.g. a
        fast, close flyby) and the jitter correction will be inaccurate
        even though the ISD will still construct without error.

        Returns
        -------
        : dict
          keys: fit_rms_px, fit_max_px, validate_rms_px, validate_max_px
        """
        if not hasattr(self, "_jitter_coeffs"):
            self._jitter_coeffs = self._jitter_fit_coeffs()
        return self._jitter_coeffs[2]

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
