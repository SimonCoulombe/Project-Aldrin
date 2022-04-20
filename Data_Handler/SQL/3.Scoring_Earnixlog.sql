-- The following script download data from AL_ARCHIVE_COMP_ENX(Earnix log) for Oceania scoring purpose
-- This SQL takes data with condition:
-- 1. Adjusted the Gold Comprehensive position as if Oceania is not present

-- Note : The script DOES NOT filtered for margin criteria. We past the full cohort for scoring.

-- updated 04 APR 2022 by Marcus Leong

select
GOLD.* except( partnerrankposition, journeyid, RTEDAT)
,case when OCEA.OCEA_RANKING < GOLD.partnerrankposition then GOLD.partnerrankposition-1
                                                          else GOLD.partnerrankposition 
                                                          end as partnerrankposition
,MF.Marketing_Fund                                                      
from ( 
    -- BUDGET DIRECT GOLD
    SELECT
    VEHMDL
    ,CORPCD
    ,VEHCLR
    ,cast(VEHAGE as INT64) VEHAGE
    ,STACDE
    ,cast(VEHCYL as INT64) VEHCYL
    ,RATARE
    ,MOPCDE
    ,USECDE
    ,FINTYP
    ,OTHVEH
    ,RGDYGD
    ,SBRNMA
    ,cast(VPOWER as float64) as VPOWER
    ,cast(WEIGHT as float64) as WEIGHT
    ,HMEOWN
    ,cast(ANCPRA as int64) as ANCPRA
    ,cast(NEWPRI as float64) as NEWPRI
    ,cast(POWWEI as float64) as POWWEI 
    ,INSNAM
    ,RATHIR
    ,RAWEAT
    ,SUBSEQ
    ,cast(VEHLEN as float64) as VEHLEN
    ,VFACSG
    ,journeyid
    ,partnerrankposition
    ,RTEDAT
    FROM
      `pricing-nonprod-687b.ML.AL_ARCHIVE_COMP_ENX`
    WHERE UNDPRD =4 and AFNCDE="BUDD"
    ) GOLD

inner join (
    -- OCEANIA
    SELECT
    journeyid
    ,RTEDAT
    ,partnerrankposition as OCEA_RANKING
    FROM
      `pricing-nonprod-687b.ML.AL_ARCHIVE_COMP_ENX`
    WHERE UNDPRD =18 and AFNCDE="IHAF"
    ) OCEA
on GOLD.journeyid = OCEA.journeyid
and GOLD.RTEDAT=OCEA.RTEDAT

left join `pricing-nonprod-687b.ML.AL_Marketing_Fund` MF
on GOLD.RATARE = CAST(MF.RATARE as STRING)

where 1=1
-- and MF.Marketing_Fund>-0.03  --inclusive of all portfolio data during scoring
and GOLD.RTEDAT>"2022-03-01"

