-- The following script download data from AL_ARCHIVE_COMP(XMLLOG) for Oceania scoring purpose
-- 1. COVER_TYPE='A'  (Comprehensive)
-- 2. AFNTY_BRAND='BUDD'
-- 3. UNDERWRITER_PRODUCT=504 A&G GOLD

-- updated 01 DEC 2021 by Marcus Leong



SELECT
    quotecompleted as QUOTE_DATE,
    cast(RATING_AREA_OWN_DAMAGE as INT64) RATARE,
    R.Marketing_Fund,
    cast(partnerrankposition as int64) partnerrankposition,  --target
  -- list of all atrributes for existing model
    VEHICLE_MODEL,
    VEHICLE_KMS,
    cast(WE_COST as float64) WE_COST,
    DRIVER_OPTION,
    VEHICLE_COLOUR,
    cast(PC_CORVEH_FIN_PREM as float64) PC_CORVEH_FIN_PREM ,
    cast(TH_FREQ as float64) TH_FREQ,
    cast(HC_FREQ as float64) HC_FREQ,
    cast(NCD as INT64) NCD,
    YD_LICENCE_TYPE,
    cast(POSTCODE as int64) POSTCODE,
    cast(OD_COST as float64) OD_COST,
    cast(WE_FREQ as float64) WE_FREQ,
    cast(HC_COST as float64) HC_COST,
    cast(TP_FREQ as float64) TP_FREQ,
    cast(VEHICLE_AGE as int64) VEHICLE_AGE,
    cast(RD_CLAIM_COUNT_5YR as int64) RD_CLAIM_COUNT_5YR,
    cast(TP_COST as float64) TP_COST,
    cast(POSTCODE as STRING) as POSTCODE_STR,
    STATE_CODE,
    cast(ACCESSORIES_VALUE as int64) as ACCESSORIES_VALUE,
    VEHICLE_MAKE,
    cast(NUMBER_OF_CYLINDERS as int64) as NUMBER_OF_CYLINDERS,
      DRIVE_TYPE,
      cast(RATING_AREA_OWN_DAMAGE as STRING) as RATING_AREA_OWN_DAMAGE,
      cast(INSURED_VALUE as int64) as INSURED_VALUE,
      cast(WINDOWGLASS_VEHICLE_CATEGORY as STRING) as WINDOWGLASS_VEHICLE_CATEGORY,
      cast(METHOD_OF_PARKING as STRING) as METHOD_OF_PARKING,
      cast(USE_CODE as STRING) as USE_CODE,
      FINANCE_TYPE,
      RD_OTHER_VEHICLE,
      REGULAR_DRIVER_IS_YOUNGEST,
      cast(FACTORY_OPTIONS_VALUE as int64) as FACTORY_OPTIONS_VALUE,
      RISK_SUBURB,
      RD_LICENCE_TYPE,
      cast(POWER as int64) as POWER,
      cast(WEIGHT as int64) as WEIGHT,
      HOME_OWNERSHIP,
      CATCHMENT_AREA,
      cast(ANCAP_RATING as int64) as ANCAP_RATING,
      BODY_ENGINE_MODIFIED,
      BULL_BAR_FITTED,
      HAS_PREVIOUS_INSURANCE,
      cast(NEW_PRICE as int64) as NEW_PRICE,
      cast(POWER_TO_WEIGHT_RATIO as float64) as POWER_TO_WEIGHT_RATIO,
      cast(RATING_AREA_THEFT as STRING) as RATING_AREA_THEFT,
      cast(RATING_AREA_THIRD_PARTY as STRING) as RATING_AREA_THIRD_PARTY,
      RATING_AREA_WEATHER,
      SUBURB_SEQUENCE,
      cast(ACCELERATION as float64) as ACCELERATION,
      cast(VEHICLE_LENGTH as int64) as VEHICLE_LENGTH,
      VFACTS_CLASS,
      VFACTS_SEGMENT
  -- end of model attributes  
  FROM
    `pricing-nonprod-687b.ML.AL_ARCHIVE_COMP` L
  left join   `pricing-nonprod-687b.ML.AL_Marketing_Fund` R
  on L.RATING_AREA_OWN_DAMAGE=R.RATARE
  WHERE
    1=1
    AND AFNTY_BRAND='BUDD' 
    AND UNDERWRITER_PRODUCT=504
    AND COVER_TYPE='A' --Use quotes from Budget Direct Comprehensive only
    and quotecompleted >='2022-01-28'