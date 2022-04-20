-- The following script download data from AL_ARCHIVE_COMP(XMLLOG) for Oceania ML model update only
-- This SQL takes data with condition:
-- 1. COVER_TYPE='A'  (Comprehensive)
-- 2. AFNTY_BRAND='BUDD'
-- 3. UNDERWRITER_PRODUCT=504 A&G GOLD

-- updated 23 FEB 2022 by Marcus Leong to include all variables


SELECT
  cast(partnerrankposition as int64) partnerrankposition,  --target
  VEHICLE_MODEL,
  cast(POSTCODE as STRING) as POSTCODE_STR,
  VEHICLE_COLOUR,
  cast(VEHICLE_AGE as int64) as VEHICLE_AGE,
  STATE_CODE,
  -- cast(ACCESSORIES_VALUE as int64) as ACCESSORIES_VALUE,
  cast(NUMBER_OF_CYLINDERS as int64) as NUMBER_OF_CYLINDERS,
  cast(RATING_AREA_OWN_DAMAGE as STRING) as RATING_AREA_OWN_DAMAGE,
  cast(METHOD_OF_PARKING as STRING) as METHOD_OF_PARKING,
  cast(USE_CODE as STRING) as USE_CODE,
  FINANCE_TYPE,
  RD_OTHER_VEHICLE,
  REGULAR_DRIVER_IS_YOUNGEST,
  RISK_SUBURB,
  cast(POWER as int64) as POWER,
  cast(WEIGHT as int64) as WEIGHT,
  HOME_OWNERSHIP,
  -- CATCHMENT_AREA,
  cast(ANCAP_RATING as int64) as ANCAP_RATING,
  cast(NEW_PRICE as int64) as NEW_PRICE,
  cast(POWER_TO_WEIGHT_RATIO as float64) as POWER_TO_WEIGHT_RATIO,
  cast(RATING_AREA_WEATHER as STRING) as RATING_AREA_WEATHER,
  cast(RATING_AREA_THIRD_PARTY as STRING) as RATING_AREA_THIRD_PARTY,
  SUBURB_SEQUENCE,
  cast(VEHICLE_LENGTH as int64) as VEHICLE_LENGTH,
  VFACTS_SEGMENT

  FROM
    `pricing-nonprod-687b.ML.AL_ARCHIVE_COMP`
  WHERE
    1=1
    AND AFNTY_BRAND='BUDD' 
    AND UNDERWRITER_PRODUCT=504
    AND COVER_TYPE='A' --Use quotes from Budget Direct Comprehensive only
    and quotecompleted < '2022-01-28'
