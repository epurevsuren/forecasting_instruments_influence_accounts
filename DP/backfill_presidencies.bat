@echo off
setlocal
REM ============================================================================
REM  backfill_presidencies.bat  --  PRIME-TIME backfill
REM  Fetches each geopolitical/institutional X account only during its OWN prime
REM  window (tenure intersected with the presidency), NOT the whole 4-year span.
REM  Dates come from influence_accounts.json active_from / active_to; MANUAL dates
REM  are used where a leader's JSON active_from reflects only their CURRENT term
REM  but they ruled earlier in the window (Modi, Netanyahu, Macron, Erdogan,
REM  monarchs, Khamenei ...). "FULL" groups = active the entire window.
REM
REM  Presidents' PERSONAL accounts are excluded (already full): @realDonaldTrump,
REM  @JoeBiden. Accounts already fetched are simply left out of the lists.
REM
REM  RESUMABLE: the retriever saves each month and SKIPS covered months. X
REM  throttles ~600-700 tweets/session, so re-run this .bat until it reports
REM  mostly "skip (already have ...)".
REM ============================================================================
cd /d "%~dp0"

echo ############################################################################
echo #  46th presidency (Biden): 2021-01-20 .. 2025-01-20
echo ############################################################################
REM  Active the WHOLE 46th window (full-term leaders, monarchs, institutions,
REM  office archives). Macron/Modi/Erdogan/monarchs/Khamenei = manual full window.
call :fetch "EmmanuelMacron,narendramodi,JustinTrudeau,RTErdogan,KingAbdullahII,TamimBinHamad,presidentaz,NikolPashinyan,khamenei_ir,WhiteHouse46,VP46archive,PressSec46,NSC_Spox46,WHCEA46archive,IDF,IsraelMFA,mfa_russia,ChineseEmbinUS,NATO,UN,WHO,StateDept,DeptofDefense" 20210120 20250121
REM  Partial tenures within the 46th window (start..end):
REM   BorisJohnson    PM until 2022-09-06
call :fetch "BorisJohnson"    20210120 20220907
REM   jairbolsonaro   President until 2023-01-01
call :fetch "jairbolsonaro"   20210120 20230102
REM   lopezobrador_   President until 2024-10-01
call :fetch "lopezobrador_"   20210120 20241002
REM   jensstoltenberg NATO SecGen until 2024-10-01
call :fetch "jensstoltenberg" 20210120 20241002
REM   MohamedBinZayed UAE President from 2022-05-14
call :fetch "MohamedBinZayed" 20220514 20250121
REM   GiorgiaMeloni   PM from 2022 (election 2022-09-25)
call :fetch "GiorgiaMeloni"   20220925 20250121
REM   LulaOficial     won 2022-10-30, President 2023
call :fetch "LulaOficial"     20221030 20250121
REM   JMilei          elected 2023-11-19
call :fetch "JMilei"          20231119 20250121
REM   KamalaHarris    2024 nominee prime, up to election day 2024-11-05
call :fetch "KamalaHarris"    20240721 20241106

echo.
echo ############################################################################
echo #  45th presidency (Trump I): 2017-01-20 .. 2021-01-20
echo ############################################################################
REM  Active the WHOLE 45th window (full-term leaders, monarchs, institutions,
REM  office archives). Netanyahu/Modi/Erdogan/monarchs/Khamenei = manual full window.
call :fetch "netanyahu,narendramodi,JustinTrudeau,RTErdogan,KingAbdullahII,TamimBinHamad,PresidentRuvi,khamenei_ir,WhiteHouse45,VP45,PressSec45,WHNSC45,WhiteHouseCEA45,IDF,IsraelMFA,mfa_russia,ChineseEmbinUS,NATO,UN,WHO,StateDept,DeptofDefense,USNavy,USArmy,usairforce,jensstoltenberg" 20170120 20210121
REM  Partial tenures within the 45th window (start..end):
REM   EPN            President until 2018-11-30
call :fetch "EPN"             20170120 20181201
REM   theresa_may    PM until 2019-07-24
call :fetch "theresa_may"     20170120 20190725
REM   NikkiHaley     UN Amb until 2018-12-31
call :fetch "NikkiHaley"      20170120 20190101
REM   AbeShinzo      PM until 2020-09-16
call :fetch "AbeShinzo"       20170120 20200917
REM   EmmanuelMacron President from 2017-05-14
call :fetch "EmmanuelMacron"  20170514 20210121
REM   moonriver365   President from 2017-05-10
call :fetch "moonriver365"    20170510 20210121
REM   AmbJohnBolton  NSA 2018-04-09 .. 2019-09-10
call :fetch "AmbJohnBolton"   20180409 20190911
REM   secpompeo      SecState from 2018-04-26
call :fetch "secpompeo"       20180426 20210121
REM   RichardGrenell Amb/DNI from 2018-05-08
call :fetch "RichardGrenell"  20180508 20210121
REM   GiuseppeConteIT PM from 2018-06-01
call :fetch "GiuseppeConteIT" 20180601 20210121
REM   lopezobrador_  President from 2018-12-01
call :fetch "lopezobrador_"   20181201 20210121
REM   jairbolsonaro  President from 2019-01-01
call :fetch "jairbolsonaro"   20190101 20210121
REM   BorisJohnson   PM from 2019-07-24
call :fetch "BorisJohnson"    20190724 20210121
REM   EsperDoD       SecDef 2019-07-23 .. 2020-11-09
call :fetch "EsperDoD"        20190723 20201110
REM   sugawitter     PM from 2020-09-16
call :fetch "sugawitter"      20200916 20210121

echo.
echo ############################################################################
echo #  One pass complete. X rate-limits ~600-700 tweets/session -- RE-RUN this
echo #  .bat until every line reports mostly "skip (already have ...)".
echo ############################################################################
endlocal
goto :eof

:fetch
REM  %~1 = comma-separated handles, %2 = since YYYYMMDD, %3 = until YYYYMMDD
echo.
echo --- %~1   [%2 .. %3] ---
uv run python x_tweets_retriever.py --handles "%~1" --since %2 --until %3
goto :eof
