-- The following script download data from AL_ARCHIVE_COMP_ENX(Earnix log) for Oceania ML model update only
-- This SQL takes data with condition:
-- 1. For quote above the marketing margin of >-0.03
-- 2. Adjusted the Gold Comprehensive position as if Oceania is not present

-- updated 04 APR 2022 by Marcus Leong


select
GOLD.* except( partnerrankposition, journeyid, RTEDAT)
-- Adjust Gold ranking when Oceania is above Gold Ranking, no impact when Oceania is below Gold
,case when OCEA.OCEA_RANKING < GOLD.partnerrankposition then GOLD.partnerrankposition-1
                                                          else GOLD.partnerrankposition 
                                                          end as partnerrankposition
-- ,MF.Marketing_Fund                                                      
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
    -- ,INSNAM
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

where MF.Marketing_Fund>-0.03

