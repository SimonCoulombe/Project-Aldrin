-- The following script is used to sample data for the Earnix API post request (batch test Earnix response)
-- Note: Please select the cohort you would like to test

-- updated 04 APR 2022 by Marcus Leong

SELECT
CLIENT_NUMBER
,VEHMDL
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
   FROM `pricing-nonprod-687b.ML.AL_ARCHIVE_COMP_ENX` L
 WHERE 1=1
      and UNDPRD =18 
      and AFNCDE="IHAF"
      and RTEDAT='2022-03-15' -- out of sample
      