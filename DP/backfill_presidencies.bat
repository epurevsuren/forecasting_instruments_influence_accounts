@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM  backfill_presidencies.bat
REM  Backfill the "other players" (world leaders, ministries, institutions and
REM  NARA archive handles) on X/Twitter for the two US presidency windows:
REM     45th  Trump I : 2017-01-20 .. 2021-01-20
REM     46th  Biden   : 2021-01-20 .. 2025-01-20
REM  x_tweets.csv currently only has these accounts from 2024-11-01 onward.
REM
REM  EXCLUDED on purpose: the two presidents' PERSONAL accounts are already full
REM    - @realDonaldTrump  -> trump_tweets.csv (2016-2021)
REM    - @JoeBiden         -> already backfilled to 2025
REM  (Their OFFICIAL office accounts @POTUS45 / @POTUS46Archive ARE included --
REM   different, official-statement content. Delete them from the lists below if
REM   you consider them redundant.)
REM
REM  RESUMABLE: x_tweets_retriever.py saves each month immediately and SKIPS
REM  months already in x_tweets.csv. X throttles ~600-700 tweets per session,
REM  so ONE pass will not finish -- just re-run this .bat until both windows
REM  report mostly "skip (already have ...)". Use --refetch on the retriever to
REM  force-refill a partially fetched month.
REM ============================================================================
cd /d "%~dp0"

REM ---- 45th presidency: leaders/officials/institutions active 2017-2021 ----
set "H45=netanyahu,EmmanuelMacron,narendramodi,AbeShinzo,sugawitter,JustinTrudeau,theresa_may,BorisJohnson,moonriver365,GiuseppeConteIT,jairbolsonaro,lopezobrador_,EPN,RTErdogan,KingAbdullahII,TamimBinHamad,PresidentRuvi,khamenei_ir,WhiteHouse45,VP45,PressSec45,WHNSC45,WhiteHouseCEA45,secpompeo,NikkiHaley,AmbJohnBolton,RichardGrenell,EsperDoD,IDF,IsraelMFA,mfa_russia,ChineseEmbinUS,NATO,UN,WHO,StateDept,DeptofDefense,USNavy,USArmy,usairforce,jensstoltenberg"

REM ---- 46th presidency: leaders/officials/institutions active 2021-2025 ----
set "H46=POTUS46Archive,netanyahu,EmmanuelMacron,narendramodi,ZelenskyyUa,JustinTrudeau,BorisJohnson,Keir_Starmer,GiorgiaMeloni,lopezobrador_,Claudiashein,JMilei,LulaOficial,jairbolsonaro,RTErdogan,KingAbdullahII,TamimBinHamad,MohamedBinZayed,NicolasMaduro,presidentaz,NikolPashinyan,khamenei_ir,KamalaHarris,WhiteHouse46,VP46archive,PressSec46,NSC_Spox46,WHCEA46archive,IDF,IsraelMFA,mfa_russia,ChineseEmbinUS,NATO,UN,WHO,StateDept,DeptofDefense,jensstoltenberg"

echo ============================================================================
echo  [1/2] 46th presidency (Biden): 2021-01-20 .. 2025-01-20
echo ============================================================================
uv run python x_tweets_retriever.py --handles "%H46%" --since 20210120 --until 20250121

echo.

echo ============================================================================
echo  [2/2] 45th presidency (Trump I): 2017-01-20 .. 2021-01-20
echo ============================================================================
uv run python x_tweets_retriever.py --handles "%H45%" --since 20170120 --until 20210121


echo.
echo ============================================================================
echo  One pass complete. X rate-limits ~600-700 tweets/session, so RE-RUN this
echo  .bat until both windows report mostly "skip (already have ...)".
echo  Covered months are skipped, so re-runs resume forward automatically.
echo ============================================================================
endlocal
