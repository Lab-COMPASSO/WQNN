export physics

module physics
    function get_total_irradiance(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
        dni, dhi, ghi, albedo)
        poa_sky_diffuse = get_sky_diffuse(surface_tilt, dhi)
        poa_ground_diffuse = get_ground_diffuse(surface_tilt, ghi, albedo)
            
        aoi_value = aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)

        poa_direct = maximum([dni * cosd(aoi_value), 0])
        poa_diffuse = poa_sky_diffuse + poa_ground_diffuse
        poa_global = poa_direct + poa_diffuse
        return poa_global
    end

    function aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        projection = (
            cosd(surface_tilt) * cosd(solar_zenith) +
            sind(surface_tilt) * sind(solar_zenith) *
            cosd(solar_azimuth - surface_azimuth)
        )
        projection = minimum([maximum([projection, -1]), 1])
        return acosd(projection)
    end

    function get_sky_diffuse(surface_tilt, dhi)
        return dhi * (1 + cosd(surface_tilt)) * 0.5
    end

    function get_ground_diffuse(surface_tilt, ghi, albedo)
        return ghi * albedo * (1 - cosd(surface_tilt)) * 0.5
    end
end


# function get_sky_diffuse(surface_tilt, dhi)
    # isotropic
    # Determine diffuse irradiance from the sky on a tilted surface using
    # the isotropic sky model.

    # .. math::

    #    I_{d} = DHI \frac{1 + \cos\beta}{2}

    # Hottel and Woertz's model treats the sky as a uniform source of
    # diffuse irradiance. Thus the diffuse irradiance from the sky (ground
    # reflected irradiance is not included in this algorithm) on a tilted
    # surface can be found from the diffuse horizontal irradiance and the
    # tilt angle of the surface.

    # Parameters
    # ----------
    # surface_tilt : numeric
    #     Surface tilt angle in decimal degrees. Tilt must be >=0 and
    #     <=180. The tilt angle is defined as degrees from horizontal
    #     (e.g. surface facing up = 0, surface facing horizon = 90)

    # dhi : numeric
    #     Diffuse horizontal irradiance in W/m^2. DHI must be >=0.

    # Returns
    # -------
    # diffuse : numeric
    #     The sky diffuse component of the solar radiation.

    # References
    # ----------
    # .. [1] Loutzenhiser P.G. et. al. "Empirical validation of models to
    #    compute solar irradiance on inclined surfaces for building energy
    #    simulation" 2007, Solar Energy vol. 81. pp. 254-267

    # .. [2] Hottel, H.C., Woertz, B.B., 1942. Evaluation of flat-plate solar
    #    heat collector. Trans. ASME 64, 91.

    # return dhi * (1 + cosd(surface_tilt)) * 0.5

# function get_ground_diffuse(surface_tilt, ghi, albedo)
#     # Estimate diffuse irradiance from ground reflections given
#     # irradiance, albedo, and surface tilt.

#     # Function to determine the portion of irradiance on a tilted surface
#     # due to ground reflections. Any of the inputs may be DataFrames or
#     # scalars.

#     # Parameters
#     # ----------
#     # surface_tilt : numeric
#     #     Surface tilt angles in decimal degrees. Tilt must be >=0 and
#     #     <=180. The tilt angle is defined as degrees from horizontal
#     #     (e.g. surface facing up = 0, surface facing horizon = 90).

#     # ghi : numeric
#     #     Global horizontal irradiance. [W/m^2]

#     # albedo : numeric, default 0.25
#     #     Ground reflectance, typically 0.1-0.4 for surfaces on Earth
#     #     (land), may increase over snow, ice, etc. May also be known as
#     #     the reflection coefficient. Must be >=0 and <=1. Will be
#     #     overridden if surface_type is supplied.

#     # surface_type: None or string, default None
#     #     If not None, overrides albedo. String can be one of 'urban',
#     #     'grass', 'fresh grass', 'snow', 'fresh snow', 'asphalt', 'concrete',
#     #     'aluminum', 'copper', 'fresh steel', 'dirty steel', 'sea'.

#     # Returns
#     # -------
#     # grounddiffuse : numeric
#     #     Ground reflected irradiance. [W/m^2]


#     # References
#     # ----------
#     # .. [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
#     #    solar irradiance on inclined surfaces for building energy simulation"
#     #    2007, Solar Energy vol. 81. pp. 254-267.

#     # The calculation is the last term of equations 3, 4, 7, 8, 10, 11, and 12.

#     return ghi * albedo * (1 - cosd(surface_tilt)) * 0.5
# end