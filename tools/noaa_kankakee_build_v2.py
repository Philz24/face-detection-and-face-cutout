import io, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import psychrolib

STATION_GHCN='USW00004880'
STATION_ISD='72533604880'
STATION_NAME='GREATER KANKAKEE AIRPORT'
ELEVATION_M=191.7
LOCAL_TZ='America/Chicago'
START_LOCAL=pd.Timestamp('2011-08-26 00:00:00', tz=LOCAL_TZ)
END_LOCAL=pd.Timestamp('2026-08-26 23:00:00', tz=LOCAL_TZ)
START_UTC=START_LOCAL.tz_convert('UTC')
END_UTC=END_LOCAL.tz_convert('UTC')
TARGET=pd.date_range(START_UTC,END_UTC,freq='h')
OUT=Path('Kankakee_Kensing_NOAA_Hourly_WetBulb_Weather_2011-08-26_to_2026-08-26.csv')
QA=Path('Kankakee_Kensing_NOAA_Hourly_WetBulb_QA.json')
S=requests.Session(); S.headers.update({'User-Agent':'Mozilla/5.0 NOAA engineering analysis','Accept':'text/csv,text/plain,*/*'})

USER_2024_CANDIDATES=[
'https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/2024/USW00004880.csv',
'https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/2024/LCD_USW00004880_2024.csv',
'https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/by-year/2024/LCD_USW00004880_2024.csv',
'https://www.ncei.noaa.gov/data/local-climatological-data/access/2024/USW00004880.csv',
'https://www.ncei.noaa.gov/data/local-climatological-data-v2/access/2024/LCD_USW00004880_2024.csv',
]

def get(url,timeout=12):
    try:
        r=S.get(url,timeout=timeout,allow_redirects=True)
        return r
    except Exception as e:
        return e

def test_user_candidates():
    log=[]
    for u in USER_2024_CANDIDATES:
        x=get(u,8)
        if isinstance(x,Exception): log.append({'url':u,'status':'EXCEPTION','detail':repr(x)})
        else: log.append({'url':u,'status':x.status_code,'bytes':len(x.content),'content_type':x.headers.get('content-type','')})
    return log

def gh_url(year):
    return f'https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{STATION_ISD}.csv'

def fetch_global_hourly(year):
    u=gh_url(year)
    last=None
    for attempt in range(3):
        x=get(u,60)
        if isinstance(x,Exception):
            last=repr(x)
        else:
            last={'status':x.status_code,'bytes':len(x.content),'content_type':x.headers.get('content-type','')}
            head=x.content[:5000].decode('utf-8','ignore')
            if x.status_code==200 and 'DATE' in head and 'TMP' in head and 'DEW' in head:
                return x.content, {'year':year,'url':u,**last}
        time.sleep(1+attempt)
    raise RuntimeError(f'Failed {u}: {last}')

def parse_isd_temp(series):
    # Global Hourly TMP/DEW: signed tenths C followed by quality code, e.g. +0123,1. +9999 is missing.
    raw=series.astype('string').str.split(',',n=1).str[0]
    v=pd.to_numeric(raw,errors='coerce')/10.0
    v=v.where(raw.str.replace('+','',regex=False).str.replace('-','',regex=False)!='9999')
    return v

def std_pressure_pa(h): return 101325.0*(1-2.25577e-5*h)**5.2559

def select(valid):
    v=valid.reset_index(drop=True).copy(); v['obs_id']=np.arange(len(v),dtype=np.int64)
    a=v.copy(); a['target_hour']=a.obs_ts.dt.floor('h')
    b=v.copy(); b['target_hour']=b.obs_ts.dt.ceil('h')
    c=pd.concat([a,b],ignore_index=True).drop_duplicates(['obs_id','target_hour'])
    c['delta']=(c.obs_ts-c.target_hour).abs().dt.total_seconds()
    c=c[(c.delta<=1800)&c.target_hour.between(START_UTC,END_UTC)]
    c['exact']=c.delta.eq(0)
    c=c.sort_values(['target_hour','exact','delta','obs_ts'],ascending=[True,False,True,True],kind='mergesort')
    used=set(); rows=[]
    for th,g in c.groupby('target_hour',sort=True):
        for row in g.itertuples(index=False):
            if int(row.obs_id) not in used:
                used.add(int(row.obs_id)); rows.append(row); break
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).set_index('target_hour').sort_index()

def main():
    assert len(TARGET)==131520
    candidate_log=test_user_candidates()
    parts=[]; dl=[]
    for y in range(2011,2027):
        content,meta=fetch_global_hourly(y); dl.append(meta)
        d=pd.read_csv(io.BytesIO(content),dtype=str,low_memory=False)
        if not {'DATE','TMP','DEW'}.issubset(d.columns): raise ValueError((y,d.columns.tolist()))
        z=pd.DataFrame({'obs_ts':pd.to_datetime(d.DATE,errors='coerce',utc=True),'temp_c':parse_isd_temp(d.TMP),'dew_c':parse_isd_temp(d.DEW)})
        z['source_year']=y
        # Compute RH directly from NOAA observed dry bulb + dew point with PsychroLib.
        parts.append(z.dropna(subset=['obs_ts']))
        print('YEAR',y,'ROWS',len(z),'BYTES',len(content))
    obs=pd.concat(parts,ignore_index=True).sort_values('obs_ts').reset_index(drop=True)
    obs=obs[obs.obs_ts.between(START_UTC-pd.Timedelta(minutes=30),END_UTC+pd.Timedelta(minutes=30))].copy()
    psychrolib.SetUnitSystem(psychrolib.SI)
    rh=np.full(len(obs),np.nan)
    for i,(t,d) in enumerate(zip(obs.temp_c.to_numpy(float),obs.dew_c.to_numpy(float))):
        if np.isfinite(t) and np.isfinite(d) and d<=t+0.2:
            try: rh[i]=100*psychrolib.GetRelHumFromTDewPoint(float(t),float(d))
            except Exception: pass
    obs['rh_pct']=rh
    valid=obs[obs.temp_c.notna()&obs.dew_c.notna()&obs.rh_pct.between(0,100)].copy()
    sel=select(valid)
    cols=['obs_ts','temp_c','dew_c','rh_pct','source_year']
    if sel.empty:
        sm=pd.DataFrame(columns=cols); sm.index=pd.DatetimeIndex([],tz='UTC',name='target_hour')
    else: sm=sel[cols]
    out=pd.DataFrame(index=TARGET); out.index.name='target_hour'; out=out.join(sm,how='left')
    ok=out.obs_ts.notna()
    p=std_pressure_pa(ELEVATION_M)
    wb=np.full(len(out),np.nan)
    for i,(t,rhval) in enumerate(zip(out.temp_c.to_numpy(float),out.rh_pct.to_numpy(float))):
        if np.isfinite(t) and np.isfinite(rhval):
            wb[i]=psychrolib.GetTWetBulbFromRelHum(float(t),float(rhval)/100,p)
    local=out.index.tz_convert(LOCAL_TZ)
    final=pd.DataFrame({
        'Date':local.strftime('%Y-%m-%d'),
        'Time':local.strftime('%H:%M:%S'),
        'Time_Zone':[x.tzname() for x in local],
        'Wet_Bulb_Temperature_F':wb*1.8+32,
        'Dry_Bulb_Air_Temperature_F':out.temp_c.to_numpy(float)*1.8+32,
        'Relative_Humidity_pct':out.rh_pct.to_numpy(float),
        'Dew_Point_Temperature_F':out.dew_c.to_numpy(float)*1.8+32,
        'Timestamp_UTC':out.index.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'Observation_Timestamp_UTC':out.obs_ts.dt.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy(),
        'Wet_Bulb_Method':np.where(ok,'CALCULATED_NOAA_T_DEWPOINT_STD_PRESSURE','MISSING'),
        'Data_Status':np.where(ok,'OK','MISSING_NOAA_OBSERVATION'),
        'Source_Station_ID':STATION_GHCN,
        'Source_Station_Name':STATION_NAME,
    })
    for c in ['Wet_Bulb_Temperature_F','Dry_Bulb_Air_Temperature_F','Relative_Humidity_pct','Dew_Point_Temperature_F']:
        final[c]=pd.to_numeric(final[c],errors='coerce').round(2)
    final.to_csv(OUT,index=False,lineterminator='\n')
    check=pd.read_csv(OUT,dtype=str)
    wet=final.Wet_Bulb_Temperature_F; dry=final.Dry_Bulb_Air_Temperature_F; dew=final.Dew_Point_Temperature_F; rhv=final.Relative_Humidity_pct
    info={
      'source':'NOAA/NCEI Global Hourly (official NOAA observational archive)',
      'source_file_station_id':STATION_ISD,
      'output_station_id':STATION_GHCN,
      'candidate_2024_url_test_log':candidate_log,
      'download_log':dl,
      'target_start_local':str(START_LOCAL),'target_end_local':str(END_LOCAL),
      'target_start_utc':START_UTC.strftime('%Y-%m-%dT%H:%M:%SZ'),'target_end_utc':END_UTC.strftime('%Y-%m-%dT%H:%M:%SZ'),
      'total_target_rows':len(final),'populated_rows':int(ok.sum()),'missing_rows':int((~ok).sum()),'coverage_pct':round(float(ok.mean()*100),4),
      'duplicate_target_timestamps':int(final.Timestamp_UTC.duplicated().sum()),'source_observation_reuse':int(final.loc[ok,'Observation_Timestamp_UTC'].duplicated().sum()),
      'first_populated_target_utc':final.loc[ok,'Timestamp_UTC'].iloc[0] if ok.any() else None,'last_populated_target_utc':final.loc[ok,'Timestamp_UTC'].iloc[-1] if ok.any() else None,
      'last_available_noaa_observation_utc':obs.obs_ts.max().strftime('%Y-%m-%dT%H:%M:%SZ') if len(obs) else None,
      'rh_outside_0_100':int(((rhv<0)|(rhv>100)).sum()),'dewpoint_gt_drybulb':int((dew>dry+.1).sum()),'wetbulb_gt_drybulb':int((wet>dry+.1).sum()),'wetbulb_lt_dewpoint':int((wet<dew-.1).sum()),
      'csv_rows':len(check),'csv_exact_columns_order':list(check.columns)==list(final.columns),'standard_pressure_pa':round(p,2),
    }
    QA.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k not in ('download_log','candidate_2024_url_test_log')},indent=2))
    assert len(final)==131520 and len(check)==131520
    assert info['duplicate_target_timestamps']==0 and info['source_observation_reuse']==0
    assert info['rh_outside_0_100']==0 and info['dewpoint_gt_drybulb']==0 and info['wetbulb_gt_drybulb']==0 and info['wetbulb_lt_dewpoint']==0
    assert info['csv_exact_columns_order']

if __name__=='__main__': main()
